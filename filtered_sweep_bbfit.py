#!/usr/bin/env python3
"""
Threshold sweep met hard filter op positionValue voor BB-fit LSTM step196.

Vergelijkt twee strategieën:
  A. Alleen threshold op softmax (zoals eerder)
  B. Threshold + hard filter: klasse 2 alleen als positionValue == 0
     op het laatste timestep

positionValue zit op feature-index 20 in het per-step feature-blok,
dus x[63, 20] in een [64, 33] sequentie.

Usage:
    python filtered_sweep_bbfit.py \
        --checkpoint /workspace/checkpoints/.../checkpoint_epoch01_step0000196.pt \
        --validation-csv /workspace/data/lstm_validation_sequences.csv \
        --train-csv /workspace/data/lstm_train_sequences.csv \
        --output-json /workspace/data/filtered_sweep_step196.json
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

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
MAJORITY_CLASS = 1
POSITION_VALUE_FEAT_IDX = 20  # index binnen per-step features
LAST_TIMESTEP = 63            # 0-indexed, sequentielengte 64


# ---------------------------------------------------------------------------
# Dataset — geeft ook positionValue terug
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
                    # positionValue op het laatste timestep
                    "position_value": x[LAST_TIMESTEP, POSITION_VALUE_FEAT_IDX].clone(),
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
        return {
            "action_logits": self.action_head(h),
            "trade_side_logits": self.trade_side_head(h),
        }


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


def safe_div(n, d):
    return n / d if d > 0 else 0.0


def print_vram():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated(0) / 1024**3
        r = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM allocated={a:.2f} GiB reserved={r:.2f} GiB", flush=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_all(model, loader, device):
    all_action_probs, all_trade_probs = [], []
    all_true_action, all_true_trade = [], []
    all_position_value = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            out = model(x)
            all_action_probs.append(F.softmax(out["action_logits"], dim=1).cpu())
            all_trade_probs.append(F.softmax(out["trade_side_logits"], dim=1).cpu())
            all_true_action.append(y["action_taken"])
            all_true_trade.append(y["trade_side"])
            all_position_value.append(y["position_value"])
            if (batch_idx + 1) % 200 == 0:
                print(f"  {batch_idx + 1} batches...", flush=True)

    return {
        "action_probs": torch.cat(all_action_probs),
        "trade_probs": torch.cat(all_trade_probs),
        "true_action": torch.cat(all_true_action),
        "true_trade": torch.cat(all_true_trade),
        "position_value": torch.cat(all_position_value),
    }


# ---------------------------------------------------------------------------
# Predicitons & metrics
# ---------------------------------------------------------------------------

def predict(probs, threshold, position_value=None):
    """
    Vuur klasse 0 of 2 alleen als prob >= threshold.
    Als position_value meegegeven: vuur klasse 2 alleen als positionValue == 0.
    """
    minority_probs = probs.clone()
    minority_probs[:, MAJORITY_CLASS] = -1.0
    best_prob, best_class = minority_probs.max(dim=1)

    preds = torch.full((probs.size(0),), MAJORITY_CLASS, dtype=torch.long)
    fire = best_prob >= threshold

    if position_value is not None:
        # Hard filter: klasse 2 alleen als niet in positie
        not_in_position = position_value.abs() < 1.0
        # Klasse 0 fire: normaal
        fire_kl0 = fire & (best_class == 0)
        # Klasse 2 fire: threshold + positie-filter
        fire_kl2 = fire & (best_class == 2) & not_in_position
        preds[fire_kl0] = 0
        preds[fire_kl2] = 2
    else:
        preds[fire] = best_class[fire]

    return preds


def compute_metrics(preds, labels):
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        confusion[t][p] += 1

    recalls, precisions, f1s = [], [], []
    per_class = []
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(NUM_CLASSES) if r != i)
        fn = sum(confusion[i][c] for c in range(NUM_CLASSES) if c != i)
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        per_class.append({
            "class": i, "support": sum(confusion[i]),
            "tp": confusion[i][i],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        })

    n_kl2_fired = int((preds == 2).sum().item())
    return {
        "accuracy": round(safe_div(correct, total), 4),
        "balanced_accuracy": round(sum(recalls) / NUM_CLASSES, 4),
        "macro_f1": round(sum(f1s) / NUM_CLASSES, 4),
        "n_class2_fired": n_kl2_fired,
        "class2_fire_rate": round(safe_div(n_kl2_fired, total), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def sweep(probs, labels, thresholds, position_value=None):
    return [
        {"threshold": thr,
         **compute_metrics(predict(probs, thr, position_value), labels)}
        for thr in thresholds
    ]


def print_table(results, title):
    print(f"\n{'='*78}")
    print(f"{title}")
    print(f"{'='*78}")
    hdr = f"{'thresh':>7} | {'bal_acc':>7} | {'p2':>7} {'r2':>7} {'f1_2':>7} | {'kl2_fired':>9} | {'fire%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        pc2 = r["per_class"][2]
        print(
            f"{r['threshold']:>7.2f} | {r['balanced_accuracy']:>7.4f} | "
            f"{pc2['precision']:>7.4f} {pc2['recall']:>7.4f} {pc2['f1']:>7.4f} | "
            f"{r['n_class2_fired']:>9d} | {r['class2_fire_rate']*100:>6.2f}"
        )


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
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threshold-min", type=float, default=0.30)
    p.add_argument("--threshold-max", type=float, default=0.95)
    p.add_argument("--threshold-step", type=float, default=0.05)
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
    print(f"sequence_length={seq_len}  per_step_features={per_step}", flush=True)
    print(f"positionValue filter op: x[{LAST_TIMESTEP}, {POSITION_VALUE_FEAT_IDX}]", flush=True)

    loader = DataLoader(
        SequenceCsvIterableDataset(
            Path(args.validation_csv), feature_columns, seq_len, per_step),
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    )

    model = MultiHeadLstm(per_step, args.hidden_size, args.num_layers, args.dropout).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected:
        print(f"Genegeerde keys: {unexpected}", flush=True)
    print(f"Checkpoint geladen: {args.checkpoint}", flush=True)
    print_vram()

    print("\nInference...", flush=True)
    collected = collect_all(model, loader, device)
    n = collected["true_action"].shape[0]
    pv = collected["position_value"]
    n_not_in_pos = int((pv.abs() < 1.0).sum().item())
    print(f"Klaar. {n:,} samples.", flush=True)
    print(f"Niet in positie (positionValue≈0): {n_not_in_pos:,} ({100*n_not_in_pos/n:.1f}%)", flush=True)
    print(f"Wel in positie: {n - n_not_in_pos:,} ({100*(n-n_not_in_pos)/n:.1f}%)", flush=True)

    thresholds = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-9:
        thresholds.append(round(t, 3))
        t += args.threshold_step

    print("\nSweep A: alleen threshold (geen filter)...", flush=True)
    sweep_a = sweep(collected["action_probs"], collected["true_action"], thresholds)

    print("Sweep B: threshold + positionValue==0 filter...", flush=True)
    sweep_b = sweep(collected["action_probs"], collected["true_action"], thresholds,
                    position_value=pv)

    print_table(sweep_a, "A: Alleen threshold (referentie)")
    print_table(sweep_b, "B: Threshold + positionValue==0 filter")

    # Toon verbetering per drempel
    print(f"\n{'='*60}")
    print("VERBETERING filter B vs A (klasse 2 precision)")
    print(f"{'thresh':>7} | {'p2_A':>7} | {'p2_B':>7} | {'delta':>7} | {'r2_A':>7} | {'r2_B':>7}")
    print("-" * 60)
    for a, b in zip(sweep_a, sweep_b):
        p2a = a["per_class"][2]["precision"]
        p2b = b["per_class"][2]["precision"]
        r2a = a["per_class"][2]["recall"]
        r2b = b["per_class"][2]["recall"]
        delta = p2b - p2a
        marker = " <<<" if delta > 0.02 else ""
        print(f"{a['threshold']:>7.2f} | {p2a:>7.4f} | {p2b:>7.4f} | {delta:>+7.4f} | {r2a:>7.4f} | {r2b:>7.4f}{marker}")

    report = {
        "checkpoint": args.checkpoint,
        "n_samples": n,
        "n_not_in_position": n_not_in_pos,
        "thresholds": thresholds,
        "sweep_threshold_only": sweep_a,
        "sweep_threshold_plus_filter": sweep_b,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nOpgeslagen: {out}", flush=True)

    print("\nKlaar.", flush=True)


if __name__ == "__main__":
    main()
