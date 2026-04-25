#!/usr/bin/env python3
"""
Confidence threshold sweep for BB-fit LSTM.

Runs inference once on the validation set, then sweeps minority-class
confidence thresholds without retraining. A minority prediction (class 0
or 2) is only accepted if its softmax probability >= threshold; otherwise
the sample is classified as class 1 (hold/no-action).

Usage:
    python threshold_sweep_bbfit.py \
        --checkpoint /workspace/checkpoints/lstm_bbfit/classification_warmup_03/checkpoint_epoch01_step0000196.pt \
        --validation-csv /workspace/data/lstm_validation_sequences.csv \
        --train-csv /workspace/data/lstm_train_sequences.csv \
        --output-json /workspace/data/threshold_sweep_step196.json
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
MINORITY_CLASSES = {0, 2}
MAJORITY_CLASS = 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceCsvIterableDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        feature_columns: Sequence[str],
        sequence_length: int,
        per_step_features: int,
        limit_rows: Optional[int] = None,
    ) -> None:
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
                    self.sequence_length, self.per_step_features
                )
                y = {
                    "action_taken": torch.tensor(int(float(row["target_actionTaken"])), dtype=torch.long),
                    "trade_side": torch.tensor(int(float(row["target_tradeSide"])), dtype=torch.long),
                }
                yield x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHeadLstm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_size, NUM_CLASSES)
        self.trade_side_head = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output, _ = self.lstm(x)
        h = self.dropout(output[:, -1, :])
        return {
            "action_logits": self.action_head(h),
            "trade_side_logits": self.trade_side_head(h),
        }


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def discover_columns(csv_path: Path) -> Tuple[List[str], List[str]]:
    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))
    feature_columns = [c for c in header if c not in META_COLUMNS and c not in TARGET_COLUMNS]
    return header, feature_columns


def infer_sequence_shape(feature_columns: Sequence[str]) -> Tuple[int, int]:
    prefixes = set()
    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            prefixes.add(col[:4])
    if not prefixes:
        raise ValueError("Cannot infer sequence prefixes from feature columns.")
    seq_len = len(prefixes)
    per_step = len(feature_columns) // seq_len
    if seq_len * per_step != len(feature_columns):
        raise ValueError("Feature count not divisible by sequence length.")
    return seq_len, per_step


# ---------------------------------------------------------------------------
# Inference — collect raw probs and true labels
# ---------------------------------------------------------------------------

def collect_probs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Returns dict with 'action_probs', 'trade_side_probs', 'true_action', 'true_trade_side'."""
    all_action_probs: List[torch.Tensor] = []
    all_trade_probs: List[torch.Tensor] = []
    all_true_action: List[torch.Tensor] = []
    all_true_trade: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            outputs = model(x)

            action_probs = F.softmax(outputs["action_logits"], dim=1).cpu()
            trade_probs = F.softmax(outputs["trade_side_logits"], dim=1).cpu()

            all_action_probs.append(action_probs)
            all_trade_probs.append(trade_probs)
            all_true_action.append(y["action_taken"])
            all_true_trade.append(y["trade_side"])

            if (batch_idx + 1) % 200 == 0:
                print(f"  collected {batch_idx + 1} batches...", flush=True)

    return {
        "action_probs": torch.cat(all_action_probs, dim=0),
        "trade_side_probs": torch.cat(all_trade_probs, dim=0),
        "true_action": torch.cat(all_true_action, dim=0),
        "true_trade_side": torch.cat(all_true_trade, dim=0),
    }


# ---------------------------------------------------------------------------
# Threshold application and metrics
# ---------------------------------------------------------------------------

def apply_threshold(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    For each sample:
    - If the highest-probability minority class (0 or 2) has prob >= threshold,
      predict that class.
    - Otherwise predict class 1 (majority).
    """
    minority_mask = torch.zeros(NUM_CLASSES, dtype=torch.bool)
    for c in MINORITY_CLASSES:
        minority_mask[c] = True

    minority_probs = probs.clone()
    minority_probs[:, MAJORITY_CLASS] = -1.0  # suppress majority

    best_minority_prob, best_minority_class = minority_probs.max(dim=1)

    predictions = torch.full((probs.size(0),), MAJORITY_CLASS, dtype=torch.long)
    fire = best_minority_prob >= threshold
    predictions[fire] = best_minority_class[fire]

    return predictions


def safe_div(n: float, d: float) -> float:
    return n / d if d > 0 else 0.0


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict:
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
        support = sum(confusion[i])
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = safe_div(2 * p * r, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        per_class.append({
            "class": i,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })

    n_fired_minority = int((preds != MAJORITY_CLASS).sum().item())

    return {
        "accuracy": round(safe_div(correct, total), 4),
        "balanced_accuracy": round(sum(recalls) / NUM_CLASSES, 4),
        "macro_f1": round(sum(f1s) / NUM_CLASSES, 4),
        "n_fired_minority": n_fired_minority,
        "minority_fire_rate": round(safe_div(n_fired_minority, total), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def sweep(
    collected: Dict[str, torch.Tensor],
    thresholds: List[float],
    head: str,
) -> List[Dict]:
    probs = collected[f"{head}_probs"]
    labels = collected[f"true_{head}"]
    results = []
    for thr in thresholds:
        preds = apply_threshold(probs, thr)
        metrics = compute_metrics(preds, labels)
        results.append({"threshold": thr, **metrics})
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sweep_table(sweep_results: List[Dict], head: str) -> None:
    print(f"\n{'='*80}")
    print(f"HEAD: {head}")
    print(f"{'='*80}")
    header = (
        f"{'thresh':>7} | {'bal_acc':>7} | {'mac_f1':>7} | "
        f"{'p0':>6} {'r0':>6} | {'p1':>6} {'r1':>6} | "
        f"{'p2':>6} {'r2':>6} | {'fire_rate':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        pc = {d["class"]: d for d in r["per_class"]}
        print(
            f"{r['threshold']:>7.2f} | {r['balanced_accuracy']:>7.4f} | {r['macro_f1']:>7.4f} | "
            f"{pc[0]['precision']:>6.4f} {pc[0]['recall']:>6.4f} | "
            f"{pc[1]['precision']:>6.4f} {pc[1]['recall']:>6.4f} | "
            f"{pc[2]['precision']:>6.4f} {pc[2]['recall']:>6.4f} | "
            f"{r['minority_fire_rate']:>9.4f}"
        )


def print_vram() -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"VRAM allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold sweep for BB-fit LSTM.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--validation-csv", required=True)
    p.add_argument("--train-csv", required=True, help="Used only for column discovery")
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


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available.")
    device = torch.device("cuda:0")
    print(f"Using device: {device}", flush=True)
    print_vram()

    train_csv = Path(args.train_csv)
    val_csv = Path(args.validation_csv)
    checkpoint_path = Path(args.checkpoint)

    _, feature_columns = discover_columns(train_csv)
    seq_len, per_step = infer_sequence_shape(feature_columns)
    print(f"sequence_length={seq_len}  per_step_features={per_step}", flush=True)

    dataset = SequenceCsvIterableDataset(
        path=val_csv,
        feature_columns=feature_columns,
        sequence_length=seq_len,
        per_step_features=per_step,
        limit_rows=args.validation_limit_rows,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MultiHeadLstm(
        input_size=per_step,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    payload = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected:
        print(f"Ignored extra keys in checkpoint: {unexpected}", flush=True)
    if missing:
        print(f"WARNING missing keys: {missing}", flush=True)
    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    print_vram()

    # Single inference pass — collect all probs
    print("\nRunning inference on validation set...", flush=True)
    collected = collect_probs(model, loader, device)
    n_samples = collected["true_action"].shape[0]
    print(f"Inference done. {n_samples:,} samples collected.", flush=True)
    print_vram()

    # Build threshold list
    thresholds = []
    t = args.threshold_min
    while t <= args.threshold_max + 1e-9:
        thresholds.append(round(t, 3))
        t += args.threshold_step

    # Sweep both heads
    print("\nSweeping thresholds...", flush=True)
    action_results = sweep(collected, thresholds, "action")
    trade_results = sweep(collected, thresholds, "trade_side")

    print_sweep_table(action_results, "action_taken")
    print_sweep_table(trade_results, "trade_side")

    report = {
        "checkpoint": str(checkpoint_path),
        "n_validation_samples": n_samples,
        "thresholds": thresholds,
        "action_taken": action_results,
        "trade_side": trade_results,
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nSaved sweep report to: {out}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
