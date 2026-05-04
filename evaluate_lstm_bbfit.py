#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

# Optional: normalization stats loaded at startup
_NORM_MEAN: Optional[torch.Tensor] = None
_NORM_STD: Optional[torch.Tensor] = None


META_COLUMNS = [
    "runId",
    "split",
    "windowStartTimestamp",
    "windowEndTimestamp",
    "targetTimestamp",
    "sequenceLength",
]

TARGET_COLUMNS = [
    "target_actionTaken",
    "target_tradeSide",
    "target_tradeActionRaw",
    "target_lastTrade",
    "target_netEquity",
    "target_netEquityDelta",
    "target_isNetEquityUp",
]


@dataclass
class DatasetSpec:
    path: Path
    limit_rows: Optional[int] = None


class SequenceCsvIterableDataset(IterableDataset):
    def __init__(
        self,
        spec: DatasetSpec,
        feature_columns: Sequence[str],
        sequence_length: int,
        per_step_features: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.per_step_features = per_step_features

    def _parse_row(self, row: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = [float(row[c]) for c in self.feature_columns]
        x = torch.tensor(features, dtype=torch.float32)
        if _NORM_MEAN is not None and _NORM_STD is not None:
            x = (x - _NORM_MEAN) / _NORM_STD
        x = x.view(self.sequence_length, self.per_step_features)

        y = {
            "action_taken": torch.tensor(int(float(row["target_actionTaken"])), dtype=torch.long),
            "trade_side": torch.tensor(int(float(row["target_tradeSide"])), dtype=torch.long),
            "net_equity_delta": torch.tensor(float(row["target_netEquityDelta"]), dtype=torch.float32),
        }
        return x, y

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        with self.spec.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for global_idx, row in enumerate(reader):
                if self.spec.limit_rows is not None and global_idx >= self.spec.limit_rows:
                    break
                if (global_idx % num_workers) != worker_id:
                    continue
                yield self._parse_row(row)


class MultiHeadLstm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        action_classes: int,
        trade_side_classes: int,
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
        self.action_head = nn.Linear(hidden_size, action_classes)
        self.trade_side_head = nn.Linear(hidden_size, trade_side_classes)
        self.net_equity_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output, _ = self.lstm(x)
        last_hidden = self.dropout(output[:, -1, :])
        return {
            "action_logits": self.action_head(last_hidden),
            "trade_side_logits": self.trade_side_head(last_hidden),
            "net_equity_delta": self.net_equity_head(last_hidden).squeeze(-1),
        }


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation run.")
    return torch.device("cuda:0")


def print_vram(prefix: str) -> None:
    device = torch.device("cuda:0")
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"{prefix} | VRAM allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB")


def discover_columns(csv_path: Path) -> Tuple[List[str], List[str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    feature_columns = [c for c in header if c not in META_COLUMNS and c not in TARGET_COLUMNS]
    return header, feature_columns


def infer_sequence_length(feature_columns: Sequence[str]) -> Tuple[int, int]:
    prefixes = set()
    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            prefixes.add(col[:4])
    if not prefixes:
        raise ValueError("No sequence prefixes found in feature columns.")
    sequence_length = len(prefixes)
    per_step_features = len(feature_columns) // sequence_length
    if sequence_length * per_step_features != len(feature_columns):
        raise ValueError("Feature column count is not divisible by inferred sequence length.")
    return sequence_length, per_step_features


def build_dataloader(
    spec: DatasetSpec,
    feature_columns: Sequence[str],
    sequence_length: int,
    per_step_features: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = SequenceCsvIterableDataset(
        spec=spec,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        per_step_features=per_step_features,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_train_label_stats(csv_path: Path, limit_rows: Optional[int] = None) -> Dict[str, Counter]:
    counters = {
        "target_actionTaken": Counter(),
        "target_tradeSide": Counter(),
    }
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            counters["target_actionTaken"][int(float(row["target_actionTaken"]))] += 1
            counters["target_tradeSide"][int(float(row["target_tradeSide"]))] += 1
            if limit_rows is not None and idx + 1 >= limit_rows:
                break
    return counters


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def confusion_to_metrics(confusion: List[List[int]]) -> Dict[str, object]:
    n = len(confusion)
    per_class = []
    recalls = []
    precisions = []
    f1s = []
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(n))

    for i in range(n):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n) if r != i)
        fn = sum(confusion[i][c] for c in range(n) if c != i)
        support = sum(confusion[i])
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_class.append(
            {
                "class": i,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": safe_div(correct, total),
        "balanced_accuracy": sum(recalls) / n if n else 0.0,
        "macro_precision": sum(precisions) / n if n else 0.0,
        "macro_recall": sum(recalls) / n if n else 0.0,
        "macro_f1": sum(f1s) / n if n else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes_action: int,
    num_classes_trade_side: int,
) -> Dict[str, object]:
    ce_action = nn.CrossEntropyLoss()
    ce_trade_side = nn.CrossEntropyLoss()
    regression_loss = nn.SmoothL1Loss(beta=1.0)

    action_conf = [[0 for _ in range(num_classes_action)] for _ in range(num_classes_action)]
    trade_conf = [[0 for _ in range(num_classes_trade_side)] for _ in range(num_classes_trade_side)]

    total_loss = 0.0
    total_batches = 0
    total_equity_mse = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            action_taken = y["action_taken"].to(device, non_blocking=True)
            trade_side = y["trade_side"].to(device, non_blocking=True)
            raw_net_equity_delta = y["net_equity_delta"].to(device, non_blocking=True)
            net_equity_delta = torch.sign(raw_net_equity_delta) * torch.log1p(torch.abs(raw_net_equity_delta))

            outputs = model(x)
            loss_action = ce_action(outputs["action_logits"], action_taken)
            loss_trade_side = ce_trade_side(outputs["trade_side_logits"], trade_side)
            loss_equity = regression_loss(outputs["net_equity_delta"], net_equity_delta)
            loss = loss_action + loss_trade_side + 0.01 * loss_equity

            total_loss += float(loss.detach().item())
            total_batches += 1
            total_equity_mse += float(torch.mean((outputs["net_equity_delta"] - net_equity_delta) ** 2).item())

            pred_action = outputs["action_logits"].argmax(dim=1).detach().cpu().tolist()
            pred_trade = outputs["trade_side_logits"].argmax(dim=1).detach().cpu().tolist()
            true_action = action_taken.detach().cpu().tolist()
            true_trade = trade_side.detach().cpu().tolist()

            for t, p in zip(true_action, pred_action):
                action_conf[t][p] += 1
            for t, p in zip(true_trade, pred_trade):
                trade_conf[t][p] += 1

    return {
        "loss": safe_div(total_loss, total_batches),
        "net_equity_mse": safe_div(total_equity_mse, total_batches),
        "action": confusion_to_metrics(action_conf),
        "trade_side": confusion_to_metrics(trade_conf),
    }


def collect_probs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """Collect softmax probabilities and true action labels for threshold sweep."""
    all_probs: List[torch.Tensor] = []
    all_true: List[int] = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            outputs = model(x)
            probs = torch.softmax(outputs["action_logits"], dim=1).cpu()
            all_probs.append(probs)
            all_true.extend(y["action_taken"].tolist())
    return torch.cat(all_probs, dim=0), all_true


def threshold_sweep(
    all_probs: torch.Tensor,
    all_true: List[int],
    num_classes: int = 3,
    hold_class: int = 1,
) -> List[Dict]:
    """Sweep hold threshold and report per-class metrics at each point."""
    thresholds = [round(t * 0.05, 2) for t in range(4, 21)]  # 0.20 .. 1.00

    non_hold_classes = [i for i in range(num_classes) if i != hold_class]
    hold_probs = all_probs[:, hold_class]
    non_hold_probs = all_probs[:, non_hold_classes]
    non_hold_argmax = non_hold_probs.argmax(dim=1)
    non_hold_preds = torch.tensor([non_hold_classes[i] for i in non_hold_argmax.tolist()])

    results = []
    for thresh in thresholds:
        preds = torch.where(hold_probs >= thresh,
                            torch.tensor(hold_class),
                            non_hold_preds).tolist()
        conf = [[0] * num_classes for _ in range(num_classes)]
        for t, p in zip(all_true, preds):
            conf[t][p] += 1
        m = confusion_to_metrics(conf)
        pc = {c["class"]: c for c in m["per_class"]}
        results.append({
            "hold_threshold": thresh,
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "short_precision": round(pc[0]["precision"], 4),
            "short_recall": round(pc[0]["recall"], 4),
            "short_support_predicted": sum(conf[r][0] for r in range(num_classes)),
            "long_precision": round(pc[2]["precision"], 4),
            "long_recall": round(pc[2]["recall"], 4),
            "long_support_predicted": sum(conf[r][2] for r in range(num_classes)),
        })
    return results


def evaluate_majority_baseline(
    dataloader: DataLoader,
    majority_action: int,
    majority_trade_side: int,
    num_classes_action: int,
    num_classes_trade_side: int,
) -> Dict[str, object]:
    action_conf = [[0 for _ in range(num_classes_action)] for _ in range(num_classes_action)]
    trade_conf = [[0 for _ in range(num_classes_trade_side)] for _ in range(num_classes_trade_side)]

    for _, y in dataloader:
        true_action = y["action_taken"].tolist()
        true_trade = y["trade_side"].tolist()
        for t in true_action:
            action_conf[t][majority_action] += 1
        for t in true_trade:
            trade_conf[t][majority_trade_side] += 1

    return {
        "action": confusion_to_metrics(action_conf),
        "trade_side": confusion_to_metrics(trade_conf),
    }


def print_distribution(title: str, counter: Counter) -> None:
    total = sum(counter.values())
    print(title)
    for cls, count in sorted(counter.items()):
        share = 100.0 * safe_div(count, total)
        print(f"  class {cls}: {count} ({share:.3f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BB-fit LSTM checkpoint against baseline.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit-rows", type=int, default=None)
    parser.add_argument("--validation-limit-rows", type=int, default=None)
    parser.add_argument("--test-limit-rows", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--normalization-stats", default=None,
                        help="Path to normalization_stats.json for z-score normalization")
    return parser.parse_args()


def main() -> None:
    global _NORM_MEAN, _NORM_STD

    args = parse_args()
    device = require_cuda()
    print(f"Using device: {device}")
    print_vram("Startup")

    train_csv = Path(args.train_csv)
    validation_csv = Path(args.validation_csv)
    test_csv = Path(args.test_csv)
    checkpoint = Path(args.checkpoint)

    _, feature_columns = discover_columns(train_csv)

    if args.normalization_stats:
        stats = json.loads(Path(args.normalization_stats).read_text())
        col_to_mean = dict(zip(stats["feature_columns"], stats["mean"]))
        col_to_std = dict(zip(stats["feature_columns"], stats["std"]))
        _NORM_MEAN = torch.tensor([col_to_mean.get(c, 0.0) for c in feature_columns], dtype=torch.float32)
        _NORM_STD = torch.tensor([col_to_std.get(c, 1.0) for c in feature_columns], dtype=torch.float32)
        print(f"Normalization stats loaded and aligned to {len(feature_columns)} feature columns")
    sequence_length, per_step_features = infer_sequence_length(feature_columns)
    action_classes = 3
    trade_side_classes = 3

    print(
        f"Discovered sequence_length={sequence_length}, per_step_features={per_step_features}, "
        f"action_classes={action_classes}, trade_side_classes={trade_side_classes}"
    )

    train_label_stats = load_train_label_stats(train_csv, args.train_limit_rows)
    majority_action = train_label_stats["target_actionTaken"].most_common(1)[0][0]
    majority_trade_side = train_label_stats["target_tradeSide"].most_common(1)[0][0]

    print_distribution("Train distribution for target_actionTaken:", train_label_stats["target_actionTaken"])
    print_distribution("Train distribution for target_tradeSide:", train_label_stats["target_tradeSide"])
    print(f"Majority baseline action class: {majority_action}")
    print(f"Majority baseline trade_side class: {majority_trade_side}")

    validation_loader = build_dataloader(
        DatasetSpec(validation_csv, args.validation_limit_rows),
        feature_columns,
        sequence_length,
        per_step_features,
        args.batch_size,
        args.num_workers,
    )
    test_loader = build_dataloader(
        DatasetSpec(test_csv, args.test_limit_rows),
        feature_columns,
        sequence_length,
        per_step_features,
        args.batch_size,
        args.num_workers,
    )
    validation_loader_for_baseline = build_dataloader(
        DatasetSpec(validation_csv, args.validation_limit_rows),
        feature_columns,
        sequence_length,
        per_step_features,
        args.batch_size,
        args.num_workers,
    )
    test_loader_for_baseline = build_dataloader(
        DatasetSpec(test_csv, args.test_limit_rows),
        feature_columns,
        sequence_length,
        per_step_features,
        args.batch_size,
        args.num_workers,
    )

    model = MultiHeadLstm(
        input_size=per_step_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        action_classes=action_classes,
        trade_side_classes=trade_side_classes,
    ).to(device)

    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint}")
    print_vram("Checkpoint loaded")

    validation_metrics = evaluate_model(model, validation_loader, device, action_classes, trade_side_classes)
    test_metrics = evaluate_model(model, test_loader, device, action_classes, trade_side_classes)

    print("\n=== THRESHOLD SWEEP (validation) ===")
    val_probs, val_true = collect_probs(model, validation_loader, device)
    val_sweep = threshold_sweep(val_probs, val_true)
    print(f"{'thresh':>7}  {'bal_acc':>7}  {'s_prec':>7}  {'s_rec':>7}  {'s_pred':>7}  {'l_prec':>7}  {'l_rec':>7}  {'l_pred':>7}")
    for r in val_sweep:
        print(f"{r['hold_threshold']:>7.2f}  {r['balanced_accuracy']:>7.4f}  "
              f"{r['short_precision']:>7.4f}  {r['short_recall']:>7.4f}  {r['short_support_predicted']:>7}  "
              f"{r['long_precision']:>7.4f}  {r['long_recall']:>7.4f}  {r['long_support_predicted']:>7}")

    print("\n=== THRESHOLD SWEEP (test) ===")
    test_probs, test_true = collect_probs(model, test_loader, device)
    test_sweep = threshold_sweep(test_probs, test_true)
    print(f"{'thresh':>7}  {'bal_acc':>7}  {'s_prec':>7}  {'s_rec':>7}  {'s_pred':>7}  {'l_prec':>7}  {'l_rec':>7}  {'l_pred':>7}")
    for r in test_sweep:
        print(f"{r['hold_threshold']:>7.2f}  {r['balanced_accuracy']:>7.4f}  "
              f"{r['short_precision']:>7.4f}  {r['short_recall']:>7.4f}  {r['short_support_predicted']:>7}  "
              f"{r['long_precision']:>7.4f}  {r['long_recall']:>7.4f}  {r['long_support_predicted']:>7}")
    validation_baseline = evaluate_majority_baseline(
        validation_loader_for_baseline,
        majority_action,
        majority_trade_side,
        action_classes,
        trade_side_classes,
    )
    test_baseline = evaluate_majority_baseline(
        test_loader_for_baseline,
        majority_action,
        majority_trade_side,
        action_classes,
        trade_side_classes,
    )

    report = {
        "checkpoint": str(checkpoint),
        "majority_baseline": {
            "action_class": majority_action,
            "trade_side_class": majority_trade_side,
            "validation": validation_baseline,
            "test": test_baseline,
        },
        "model": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "threshold_sweep": {
            "validation": val_sweep,
            "test": test_sweep,
        },
    }

    print("\n=== MODEL VALIDATION METRICS ===")
    print(json.dumps(validation_metrics, indent=2))
    print("\n=== MODEL TEST METRICS ===")
    print(json.dumps(test_metrics, indent=2))
    print("\n=== BASELINE VALIDATION METRICS ===")
    print(json.dumps(validation_baseline, indent=2))
    print("\n=== BASELINE TEST METRICS ===")
    print(json.dumps(test_baseline, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nSaved report to: {out_path}")

    print_vram("Done")


if __name__ == "__main__":
    main()
