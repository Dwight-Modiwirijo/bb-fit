#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


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


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    return torch.device("cuda:0")


def print_vram(prefix: str) -> None:
    if not torch.cuda.is_available():
        print(f"{prefix} | CUDA unavailable")
        return
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
    raw_feature_names = set()

    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            prefixes.add(col[:4])
            raw_feature_names.add(col[5:])

    if not prefixes:
        raise ValueError("Could not infer sequence prefixes from feature columns.")

    sequence_length = len(prefixes)
    per_step_features = len(feature_columns) // sequence_length

    if sequence_length * per_step_features != len(feature_columns):
        raise ValueError("Feature column count is not divisible by inferred sequence length.")

    return sequence_length, per_step_features


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
        x = torch.tensor(features, dtype=torch.float32).view(self.sequence_length, self.per_step_features)

        action_taken = int(float(row["target_actionTaken"]))
        trade_side = int(float(row["target_tradeSide"]))
        net_equity_delta = float(row["target_netEquityDelta"])

        y = {
            "action_taken": torch.tensor(action_taken, dtype=torch.long),
            "trade_side": torch.tensor(trade_side, dtype=torch.long),
            "net_equity_delta": torch.tensor(net_equity_delta, dtype=torch.float32),
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
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        return {
            "action_logits": self.action_head(last_hidden),
            "trade_side_logits": self.trade_side_head(last_hidden),
            "net_equity_delta": self.net_equity_head(last_hidden).squeeze(-1),
        }


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


def compute_class_count(csv_path: Path, column_name: str, limit_rows: Optional[int] = None) -> int:
    seen = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            seen.add(int(float(row[column_name])))
            if limit_rows is not None and idx + 1 >= limit_rows:
                break
    return max(seen) + 1 if seen else 1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eval_max_batches: Optional[int],
) -> Dict[str, float]:
    ce_action = nn.CrossEntropyLoss()
    ce_trade_side = nn.CrossEntropyLoss()
    regression_loss = nn.SmoothL1Loss(beta=1.0)

    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_action_correct = 0
    total_trade_side_correct = 0
    total_examples = 0
    total_net_equity_mse = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            action_taken = y["action_taken"].to(device, non_blocking=True)
            trade_side = y["trade_side"].to(device, non_blocking=True)
            raw_net_equity_delta = y["net_equity_delta"].to(device, non_blocking=True)
            net_equity_delta = torch.sign(raw_net_equity_delta) * torch.log1p(torch.abs(raw_net_equity_delta))

            outputs = model(x)

            loss_action = ce_action(outputs["action_logits"], action_taken)
            loss_trade_side = ce_trade_side(outputs["trade_side_logits"], trade_side)
            loss_equity = regression_loss(outputs["net_equity_delta"], net_equity_delta)
            loss = loss_action + loss_trade_side

            total_loss += loss.item()
            total_batches += 1
            total_examples += x.size(0)

            total_action_correct += (outputs["action_logits"].argmax(dim=1) == action_taken).sum().item()
            total_trade_side_correct += (outputs["trade_side_logits"].argmax(dim=1) == trade_side).sum().item()
            total_net_equity_mse += loss_equity.item()

            if eval_max_batches is not None and total_batches >= eval_max_batches:
                break

    if total_batches == 0:
        return {
            "loss": float("nan"),
            "action_accuracy": float("nan"),
            "trade_side_accuracy": float("nan"),
            "net_equity_mse": float("nan"),
        }

    return {
        "loss": total_loss / total_batches,
        "action_accuracy": total_action_correct / total_examples,
        "trade_side_accuracy": total_trade_side_correct / total_examples,
        "net_equity_mse": total_net_equity_mse / total_batches,
    }


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_epoch{epoch:02d}_step{global_step:07d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BB-fit LSTM on generated sequence CSV files.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--train-limit-rows", type=int, default=None)
    parser.add_argument("--validation-limit-rows", type=int, default=None)
    parser.add_argument("--test-limit-rows", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda()

    train_csv = Path(args.train_csv)
    validation_csv = Path(args.validation_csv)
    test_csv = Path(args.test_csv)
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"Using device: {device}")
    print_vram("Startup")

    _, feature_columns = discover_columns(train_csv)
    sequence_length, per_step_features = infer_sequence_length(feature_columns)

    action_classes = compute_class_count(train_csv, "target_actionTaken", args.train_limit_rows)
    trade_side_classes = compute_class_count(train_csv, "target_tradeSide", args.train_limit_rows)

    print(
        f"Discovered sequence_length={sequence_length}, per_step_features={per_step_features}, "
        f"action_classes={action_classes}, trade_side_classes={trade_side_classes}"
    )

    train_loader = build_dataloader(
        DatasetSpec(train_csv, args.train_limit_rows),
        feature_columns,
        sequence_length,
        per_step_features,
        args.batch_size,
        args.num_workers,
    )
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

    model = MultiHeadLstm(
        input_size=per_step_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        action_classes=action_classes,
        trade_side_classes=trade_side_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weights = torch.tensor([1.5, 1.0, 1.5], dtype=torch.float32, device=device)
    ce_action = nn.CrossEntropyLoss(weight=class_weights)
    ce_trade_side = nn.CrossEntropyLoss(weight=class_weights)
    regression_loss = nn.SmoothL1Loss(beta=1.0)

    global_step = 0
    start_epoch = 1

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        payload = torch.load(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        global_step = int(payload.get("global_step", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(f"Resumed from: {resume_path}")
        print(f"Resume start_epoch={start_epoch} global_step={global_step}")
        print_vram("After resume load")

    end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = 0.0
        running_batches = 0

        print(f"Epoch {epoch}/{end_epoch} started")
        print_vram(f"Epoch {epoch} start")

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            action_taken = y["action_taken"].to(device, non_blocking=True)
            trade_side = y["trade_side"].to(device, non_blocking=True)
            raw_net_equity_delta = y["net_equity_delta"].to(device, non_blocking=True)
            net_equity_delta = torch.sign(raw_net_equity_delta) * torch.log1p(torch.abs(raw_net_equity_delta))

            optimizer.zero_grad(set_to_none=True)

            outputs = model(x)
            loss_action = ce_action(outputs["action_logits"], action_taken)
            loss_trade_side = ce_trade_side(outputs["trade_side_logits"], trade_side)
            loss_equity = regression_loss(outputs["net_equity_delta"], net_equity_delta)
            loss = loss_action + loss_trade_side

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            running_loss += float(loss.detach().item())
            running_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={running_loss / running_batches:.6f} "
                    f"loss_action={loss_action.item():.6f} "
                    f"loss_tradeSide={loss_trade_side.item():.6f} "
                    f"loss_equity={loss_equity.item():.6f}"
                )

            if global_step % args.checkpoint_every_steps == 0:
                val_metrics = evaluate(model, validation_loader, device, args.eval_max_batches)
                checkpoint_path = save_checkpoint(
                    checkpoint_dir,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    val_metrics,
                )
                print(f"Checkpoint saved: {checkpoint_path}")
                print(f"Validation metrics at step {global_step}: {json.dumps(val_metrics, indent=2)}")
                print_vram(f"Checkpoint step {global_step}")
                model.train()

        epoch_train_loss = running_loss / max(running_batches, 1)
        validation_metrics = evaluate(model, validation_loader, device, args.eval_max_batches)
        test_metrics = evaluate(model, test_loader, device, args.eval_max_batches)

        print(f"Epoch {epoch} complete")
        print(f"Train loss: {epoch_train_loss:.6f}")
        print(f"Validation: {json.dumps(validation_metrics, indent=2)}")
        print(f"Test: {json.dumps(test_metrics, indent=2)}")
        print_vram(f"Epoch {epoch} end")

        checkpoint_path = save_checkpoint(
            checkpoint_dir,
            model,
            optimizer,
            epoch,
            global_step,
            {
                "train_loss": epoch_train_loss,
                "validation": validation_metrics,
                "test": test_metrics,
            },
        )
        print(f"Epoch checkpoint saved: {checkpoint_path}")

    print("Training complete.")
    print_vram("Training complete")


if __name__ == "__main__":
    main()
