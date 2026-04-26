#!/usr/bin/env python3
"""
Temperature scaling calibration for BB-fit LSTM step196.

Leert één temperatuurparameter T op de validatieset zodat de softmax-
kansen beter gekalibreerd zijn. Na kalibratie wordt een threshold sweep
gedaan op de gecalibreerde kansen.

T > 1 → kansen vlakker (minder zeker) → minder false positives voor kl.2
T < 1 → kansen scherper (meer zeker)

Usage:
    python temperature_scaling_bbfit.py \
        --checkpoint /workspace/checkpoints/lstm_bbfit/classification_warmup_03/checkpoint_epoch01_step0000196.pt \
        --validation-csv /workspace/data/lstm_validation_sequences.csv \
        --train-csv /workspace/data/lstm_train_sequences.csv \
        --output-json /workspace/data/temperature_scaling_step196.json
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def print_vram():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated(0) / 1024**3
        r = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM allocated={a:.2f} GiB reserved={r:.2f} GiB", flush=True)


def safe_div(n, d):
    return n / d if d > 0 else 0.0


# ---------------------------------------------------------------------------
# Collect logits — één inference-pas, bewaar ruwe logits
# ---------------------------------------------------------------------------

def collect_logits(model, loader, device):
    all_action_logits, all_trade_logits = [], []
    all_true_action, all_true_trade = [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            out = model(x)
            all_action_logits.append(out["action_logits"].cpu())
            all_trade_logits.append(out["trade_side_logits"].cpu())
            all_true_action.append(y["action_taken"])
            all_true_trade.append(y["trade_side"])
            if (batch_idx + 1) % 200 == 0:
                print(f"  {batch_idx + 1} batches...", flush=True)

    return {
        "action_logits": torch.cat(all_action_logits),
        "trade_logits": torch.cat(all_trade_logits),
        "true_action": torch.cat(all_true_action),
        "true_trade": torch.cat(all_true_trade),
    }


# ---------------------------------------------------------------------------
# Temperature scaling — leer T via LBFGS op validatieset
# ---------------------------------------------------------------------------

def learn_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimaliseer NLL op logits/T met één scalar T. Geeft optimale T terug."""
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperature.clamp(min=0.01)
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        })

    n_minority = int((preds != MAJORITY_CLASS).sum().item())
    return {
        "accuracy": round(safe_div(correct, total), 4),
        "balanced_accuracy": round(sum(recalls) / NUM_CLASSES, 4),
        "macro_f1": round(sum(f1s) / NUM_CLASSES, 4),
        "n_fired_minority": n_minority,
        "minority_fire_rate": round(safe_div(n_minority, total), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def apply_threshold(probs, threshold):
    """Vuur minderheidsklasse alleen als kans >= drempel, anders klasse 1."""
    minority_probs = probs.clone()
    minority_probs[:, MAJORITY_CLASS] = -1.0
    best_prob, best_class = minority_probs.max(dim=1)
    preds = torch.full((probs.size(0),), MAJORITY_CLASS, dtype=torch.long)
    preds[best_prob >= threshold] = best_class[best_prob >= threshold]
    return preds


def threshold_sweep(logits, labels, temperature, thresholds):
    scaled_probs = F.softmax(logits / max(temperature, 0.01), dim=1)
    results = []
    for thr in thresholds:
        preds = apply_threshold(scaled_probs, thr)
        m = compute_metrics(preds, labels)
        results.append({"threshold": thr, **m})
    return results


def print_table(sweep_results, head):
    print(f"\n{'='*76}")
    print(f"HEAD: {head}")
    print(f"{'='*76}")
    hdr = f"{'thresh':>7} | {'bal_acc':>7} | {'p0':>6} {'r0':>6} | {'p1':>6} {'r1':>6} | {'p2':>6} {'r2':>6} | {'fire%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in sweep_results:
        pc = {d["class"]: d for d in r["per_class"]}
        print(
            f"{r['threshold']:>7.2f} | {r['balanced_accuracy']:>7.4f} | "
            f"{pc[0]['precision']:>6.4f} {pc[0]['recall']:>6.4f} | "
            f"{pc[1]['precision']:>6.4f} {pc[1]['recall']:>6.4f} | "
            f"{pc[2]['precision']:>6.4f} {pc[2]['recall']:>6.4f} | "
            f"{r['minority_fire_rate']*100:>6.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--train-csv", required=True, help="Alleen voor kolomdetectie")
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--validation-limit-rows", type=int, default=None)
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

    loader = DataLoader(
        SequenceCsvIterableDataset(
            Path(args.validation_csv), feature_columns, seq_len, per_step,
            args.validation_limit_rows),
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    )

    model = MultiHeadLstm(per_step, args.hidden_size, args.num_layers, args.dropout).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected:
        print(f"Genegeerde keys: {unexpected}", flush=True)
    print(f"Checkpoint geladen: {args.checkpoint}", flush=True)
    print_vram()

    print("\nInference op validatieset...", flush=True)
    collected = collect_logits(model, loader, device)
    n = collected["true_action"].shape[0]
    print(f"Klaar. {n:,} samples.", flush=True)

    # Leer temperatuur per head
    print("\nTemperatuur leren voor action head...", flush=True)
    T_action = learn_temperature(collected["action_logits"], collected["true_action"])
    print(f"  T_action = {T_action:.4f}", flush=True)

    print("Temperatuur leren voor trade_side head...", flush=True)
    T_trade = learn_temperature(collected["trade_logits"], collected["true_trade"])
    print(f"  T_trade  = {T_trade:.4f}", flush=True)

    # Ongekalibreerde baseline (T=1)
    print("\nOngekalibreerd (T=1.0):", flush=True)
    raw_preds_action = collected["action_logits"].argmax(dim=1)
    raw_m = compute_metrics(raw_preds_action, collected["true_action"])
    print(f"  balanced_acc={raw_m['balanced_accuracy']}  "
          f"kl2_p={raw_m['per_class'][2]['precision']}  "
          f"kl2_r={raw_m['per_class'][2]['recall']}", flush=True)

    # Gekalibreerd argmax (T geleerd)
    print(f"\nGekalibreerd argmax (T={T_action:.4f}):", flush=True)
    cal_preds_action = F.softmax(collected["action_logits"] / T_action, dim=1).argmax(dim=1)
    cal_m = compute_metrics(cal_preds_action, collected["true_action"])
    print(f"  balanced_acc={cal_m['balanced_accuracy']}  "
          f"kl2_p={cal_m['per_class'][2]['precision']}  "
          f"kl2_r={cal_m['per_class'][2]['recall']}", flush=True)

    # Threshold sweep
    thresholds = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-9:
        thresholds.append(round(t, 3))
        t += args.threshold_step

    action_sweep = threshold_sweep(collected["action_logits"], collected["true_action"], T_action, thresholds)
    trade_sweep = threshold_sweep(collected["trade_logits"], collected["true_trade"], T_trade, thresholds)

    print_table(action_sweep, "action_taken (gecalibreerd)")
    print_table(trade_sweep, "trade_side (gecalibreerd)")

    report = {
        "checkpoint": args.checkpoint,
        "n_validation_samples": n,
        "temperature_action": T_action,
        "temperature_trade": T_trade,
        "uncalibrated_argmax": raw_m,
        "calibrated_argmax": cal_m,
        "thresholds": thresholds,
        "action_taken": action_sweep,
        "trade_side": trade_sweep,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nOpgeslagen: {out}", flush=True)

    print("\nKlaar.", flush=True)


if __name__ == "__main__":
    main()
