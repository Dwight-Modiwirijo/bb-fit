#!/usr/bin/env python3
"""
Feature inspectie voor BB-fit LSTM step196.

Onderzoekt welke features en timesteps het model gebruikt voor klasse-2
voorspellingen via:
  1. Gradiënt saliency  — |d logit_kl2 / d input| per feature en timestep
  2. TP vs FP analyse  — feature-distributies van echte kl2 (TP) vs
                          foutief voorspelde kl2 (FP, eigenlijk kl1)

Usage:
    python inspect_bbfit.py \
        --checkpoint /workspace/checkpoints/.../checkpoint_epoch01_step0000196.pt \
        --validation-csv /workspace/data/lstm_validation_sequences.csv \
        --train-csv /workspace/data/lstm_train_sequences.csv \
        --output-json /workspace/data/inspect_step196.json
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


META_COLUMNS = [
    "runId", "split", "windowStartTimestamp", "windowEndTimestamp",
    "targetTimestamp", "sequenceLength",
]
TARGET_COLUMNS = [
    "target_actionTaken", "target_tradeSide", "target_tradeActionRaw",
    "target_lastTrade", "target_netEquity", "target_netEquityDelta",
    "target_isNetEquityUp",
]
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceCsvIterableDataset(IterableDataset):
    def __init__(self, path, feature_columns, sequence_length, per_step_features, limit_rows=None):
        super().__init__()
        self.path = path
        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.per_step_features = per_step_features
        self.limit_rows = limit_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for global_idx, row in enumerate(reader):
                if self.limit_rows is not None and global_idx >= self.limit_rows:
                    break
                if (global_idx % num_workers) != worker_id:
                    continue
                features = [float(row[c]) for c in self.feature_columns]
                x = torch.tensor(features, dtype=torch.float32).view(
                    self.sequence_length, self.per_step_features)
                y = {
                    "action_taken": torch.tensor(int(float(row["target_actionTaken"])), dtype=torch.long),
                    "trade_side": torch.tensor(int(float(row["target_tradeSide"])), dtype=torch.long),
                }
                yield x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHeadLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_size, NUM_CLASSES)
        self.trade_side_head = nn.Linear(hidden_size, NUM_CLASSES)
        self.net_equity_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return {"action_logits": self.action_head(h)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_columns(csv_path):
    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))
    feature_columns = [c for c in header if c not in META_COLUMNS and c not in TARGET_COLUMNS]
    return header, feature_columns


def infer_sequence_shape(feature_columns):
    prefixes = set()
    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            prefixes.add(col[:4])
    seq_len = len(prefixes)
    per_step = len(feature_columns) // seq_len
    return seq_len, per_step


def extract_raw_feature_names(feature_columns, seq_len):
    """Haal de unieke feature-namen op (zonder timestep-prefix)."""
    names = []
    seen = set()
    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            name = col[5:]
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def print_vram():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated(0) / 1024**3
        r = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM allocated={a:.2f} GiB reserved={r:.2f} GiB", flush=True)


# ---------------------------------------------------------------------------
# Analyse 1: Gradiënt saliency
# Berekent |d logit_kl2 / d input| per feature en per timestep,
# apart voor TP (true kl2, predicted kl2) en FP (true kl1, predicted kl2).
# ---------------------------------------------------------------------------

def gradient_saliency(model, loader, device, seq_len, per_step, max_samples=5000):
    model.train()  # cuDNN LSTM vereist train-modus voor backward

    # Accumulatoren: shape [seq_len, per_step]
    grad_sum_tp = torch.zeros(seq_len, per_step)
    grad_sum_fp = torch.zeros(seq_len, per_step)
    count_tp = 0
    count_fp = 0

    for x, y in loader:
        if count_tp >= max_samples and count_fp >= max_samples:
            break

        x = x.to(device)
        x.requires_grad_(True)

        out = model(x)
        logit_kl2 = out["action_logits"][:, 2]  # klasse-2 logit per sample

        # Gradiënt van som van klasse-2 logits naar input
        logit_kl2.sum().backward()
        grad = x.grad.abs().detach().cpu()  # [batch, seq_len, per_step]

        true_action = y["action_taken"]
        pred_action = out["action_logits"].detach().cpu().argmax(dim=1)

        # TP: true=2, pred=2
        tp_mask = (true_action == 2) & (pred_action == 2)
        if tp_mask.any() and count_tp < max_samples:
            grad_sum_tp += grad[tp_mask].sum(dim=0)
            count_tp += int(tp_mask.sum().item())

        # FP: true=1, pred=2
        fp_mask = (true_action == 1) & (pred_action == 2)
        if fp_mask.any() and count_fp < max_samples:
            grad_sum_fp += grad[fp_mask].sum(dim=0)
            count_fp += int(fp_mask.sum().item())

        x.grad = None

    grad_mean_tp = (grad_sum_tp / max(count_tp, 1))  # [seq_len, per_step]
    grad_mean_fp = (grad_sum_fp / max(count_fp, 1))

    # Per feature (gemiddeld over timesteps)
    feat_tp = grad_mean_tp.mean(dim=0)  # [per_step]
    feat_fp = grad_mean_fp.mean(dim=0)

    # Per timestep (gemiddeld over features)
    time_tp = grad_mean_tp.mean(dim=1)  # [seq_len]
    time_fp = grad_mean_fp.mean(dim=1)

    return {
        "count_tp": count_tp,
        "count_fp": count_fp,
        "per_feature_tp": feat_tp.tolist(),
        "per_feature_fp": feat_fp.tolist(),
        "per_timestep_tp": time_tp.tolist(),
        "per_timestep_fp": time_fp.tolist(),
        "grid_tp": grad_mean_tp.tolist(),
        "grid_fp": grad_mean_fp.tolist(),
    }


# ---------------------------------------------------------------------------
# Analyse 2: Feature-distributies TP vs FP
# Vergelijkt gemiddelde feature-waarden (laatste timestep) voor TP en FP.
# ---------------------------------------------------------------------------

def feature_distribution(loader, device, max_samples=5000):
    model_dummy = None  # geen model nodig, alleen labels en input

    sum_tp = None
    sum_fp = None
    count_tp = 0
    count_fp = 0

    for x, y in loader:
        if count_tp >= max_samples and count_fp >= max_samples:
            break

        # Gebruik laatste timestep als representatie
        last = x[:, -1, :]  # [batch, per_step]
        true_action = y["action_taken"]

        # We hebben de voorspellingen niet hier — we laden ze apart.
        # Gebruik true labels: kl2 vs kl1 (ongeacht voorspelling)
        kl2_mask = true_action == 2
        kl1_mask = true_action == 1

        if sum_tp is None:
            per_step = last.shape[1]
            sum_tp = torch.zeros(per_step)
            sum_fp = torch.zeros(per_step)

        if kl2_mask.any() and count_tp < max_samples:
            sum_tp += last[kl2_mask].sum(dim=0)
            count_tp += int(kl2_mask.sum().item())

        if kl1_mask.any() and count_fp < max_samples:
            sum_fp += last[kl1_mask].sum(dim=0)
            count_fp += int(kl1_mask.sum().item())

    mean_tp = (sum_tp / max(count_tp, 1)).tolist()
    mean_fp = (sum_fp / max(count_fp, 1)).tolist()
    diff = [(a - b) for a, b in zip(mean_tp, mean_fp)]

    return {
        "count_class2": count_tp,
        "count_class1_sample": count_fp,
        "mean_last_step_class2": mean_tp,
        "mean_last_step_class1": mean_fp,
        "diff_class2_minus_class1": diff,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Max TP/FP samples voor gradiëntanalyse")
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA vereist.")
    device = torch.device("cuda:0")
    print(f"Using device: {device}", flush=True)
    print_vram()

    _, feature_columns = discover_columns(Path(args.train_csv))
    seq_len, per_step = infer_sequence_shape(feature_columns)
    raw_feature_names = extract_raw_feature_names(feature_columns, seq_len)
    print(f"sequence_length={seq_len}  per_step_features={per_step}", flush=True)
    print(f"Feature namen: {raw_feature_names}", flush=True)

    loader = DataLoader(
        SequenceCsvIterableDataset(
            Path(args.validation_csv), feature_columns, seq_len, per_step),
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    )

    model = MultiHeadLstm(per_step, args.hidden_size, args.num_layers, args.dropout).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    print(f"Checkpoint geladen. Genegeerd: {unexpected}", flush=True)
    print_vram()

    print("\n--- Analyse 1: Gradiënt saliency ---", flush=True)
    saliency = gradient_saliency(model, loader, device, seq_len, per_step, args.max_samples)
    print(f"  TP samples: {saliency['count_tp']}", flush=True)
    print(f"  FP samples: {saliency['count_fp']}", flush=True)

    # Top-10 features voor TP
    feat_tp = list(enumerate(saliency["per_feature_tp"]))
    feat_tp_sorted = sorted(feat_tp, key=lambda x: x[1], reverse=True)
    print("\nTop-10 features voor klasse-2 TP (model let op):")
    for rank, (idx, val) in enumerate(feat_tp_sorted[:10], 1):
        name = raw_feature_names[idx] if idx < len(raw_feature_names) else f"feat_{idx}"
        print(f"  {rank:2d}. [{idx:2d}] {name:<40s} saliency={val:.6f}")

    # Top-10 features voor FP
    feat_fp = list(enumerate(saliency["per_feature_fp"]))
    feat_fp_sorted = sorted(feat_fp, key=lambda x: x[1], reverse=True)
    print("\nTop-10 features voor klasse-2 FP (false positives):")
    for rank, (idx, val) in enumerate(feat_fp_sorted[:10], 1):
        name = raw_feature_names[idx] if idx < len(raw_feature_names) else f"feat_{idx}"
        print(f"  {rank:2d}. [{idx:2d}] {name:<40s} saliency={val:.6f}")

    # Timestep saliency
    time_tp = saliency["per_timestep_tp"]
    time_fp = saliency["per_timestep_fp"]
    peak_tp = max(range(len(time_tp)), key=lambda i: time_tp[i])
    peak_fp = max(range(len(time_fp)), key=lambda i: time_fp[i])
    print(f"\nBelangrijkste timestep TP: t={peak_tp} (van 0..{seq_len-1})")
    print(f"Belangrijkste timestep FP: t={peak_fp}")

    # Laad een nieuwe loader voor distributie-analyse
    loader2 = DataLoader(
        SequenceCsvIterableDataset(
            Path(args.validation_csv), feature_columns, seq_len, per_step),
        batch_size=256, num_workers=args.num_workers, pin_memory=True,
    )

    print("\n--- Analyse 2: Feature-distributies (laatste timestep) ---", flush=True)
    dist = feature_distribution(loader2, device, args.max_samples)
    print(f"  Klasse-2 samples: {dist['count_class2']}")
    print(f"  Klasse-1 samples: {dist['count_class1_sample']}")

    # Top-10 features met grootste verschil kl2 vs kl1
    diff = list(enumerate(dist["diff_class2_minus_class1"]))
    diff_sorted = sorted(diff, key=lambda x: abs(x[1]), reverse=True)
    print("\nTop-10 features met grootste verschil kl2 vs kl1 (laatste timestep):")
    print(f"  {'rank':>4}  {'idx':>3}  {'feature':<40s}  {'kl2_mean':>10}  {'kl1_mean':>10}  {'diff':>10}")
    for rank, (idx, d) in enumerate(diff_sorted[:10], 1):
        name = raw_feature_names[idx] if idx < len(raw_feature_names) else f"feat_{idx}"
        kl2_m = dist["mean_last_step_class2"][idx]
        kl1_m = dist["mean_last_step_class1"][idx]
        print(f"  {rank:>4}  {idx:>3}  {name:<40s}  {kl2_m:>10.4f}  {kl1_m:>10.4f}  {d:>10.4f}")

    report = {
        "checkpoint": args.checkpoint,
        "feature_names": raw_feature_names,
        "sequence_length": seq_len,
        "per_step_features": per_step,
        "saliency": saliency,
        "distribution": dist,
        "top10_saliency_tp": [
            {"rank": r+1, "feature_idx": idx,
             "feature_name": raw_feature_names[idx] if idx < len(raw_feature_names) else f"feat_{idx}",
             "saliency": val}
            for r, (idx, val) in enumerate(feat_tp_sorted[:10])
        ],
        "top10_saliency_fp": [
            {"rank": r+1, "feature_idx": idx,
             "feature_name": raw_feature_names[idx] if idx < len(raw_feature_names) else f"feat_{idx}",
             "saliency": val}
            for r, (idx, val) in enumerate(feat_fp_sorted[:10])
        ],
        "top10_diff_kl2_vs_kl1": [
            {"rank": r+1, "feature_idx": idx,
             "feature_name": raw_feature_names[idx] if idx < len(raw_feature_names) else f"feat_{idx}",
             "mean_class2": dist["mean_last_step_class2"][idx],
             "mean_class1": dist["mean_last_step_class1"][idx],
             "diff": d}
            for r, (idx, d) in enumerate(diff_sorted[:10])
        ],
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nOpgeslagen: {out}", flush=True)

    print("\nKlaar.", flush=True)


if __name__ == "__main__":
    main()
