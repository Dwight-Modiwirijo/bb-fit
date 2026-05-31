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
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter


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
    # scalper pipeline targets
    "label_direction",
    "future_return_10",
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
        norm_mean: Optional[torch.Tensor] = None,
        norm_std: Optional[torch.Tensor] = None,
        label_column: str = "target_actionTaken",
    ) -> None:
        super().__init__()
        self.spec = spec
        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.per_step_features = per_step_features
        self.norm_mean = norm_mean  # shape [n_flat_features]
        self.norm_std = norm_std
        self.label_column = label_column

    def _parse_row(self, row: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = [float(row[c]) for c in self.feature_columns]
        x = torch.tensor(features, dtype=torch.float32)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / self.norm_std
        x = x.view(self.sequence_length, self.per_step_features)

        action_taken = int(float(row[self.label_column]))
        trade_side = int(float(row.get("target_tradeSide", 0)))
        net_equity_delta = float(row.get("target_netEquityDelta", 0.0))

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


class FocalLoss(nn.Module):
    """Focal loss for class imbalance: down-weights easy majority examples."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_dataloader(
    spec: DatasetSpec,
    feature_columns: Sequence[str],
    sequence_length: int,
    per_step_features: int,
    batch_size: int,
    num_workers: int,
    norm_mean: Optional[torch.Tensor] = None,
    norm_std: Optional[torch.Tensor] = None,
    label_column: str = "target_actionTaken",
) -> DataLoader:
    dataset = SequenceCsvIterableDataset(
        spec=spec,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        per_step_features=per_step_features,
        norm_mean=norm_mean,
        norm_std=norm_std,
        label_column=label_column,
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
    parser.add_argument("--normalization-stats", default=None,
                        help="Pad naar normalization_stats.json (mean/std per feature).")
    parser.add_argument("--reset-optimizer", action="store_true", default=False,
                        help="Do not load optimizer state from checkpoint (safe when changing LR)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma. 0 = standard cross-entropy.")
    parser.add_argument("--class-weights", type=float, nargs=3, default=[2.0, 1.0, 2.0],
                        metavar=("W0", "W1", "W2"))
    parser.add_argument("--tensorboard-dir", default=None,
                        help="TensorBoard log dir. Defaults to <checkpoint-dir>/tensorboard.")
    parser.add_argument("--lr-scheduler", choices=["plateau", "cosine", "none"], default="plateau",
                        help="LR scheduler: plateau=ReduceLROnPlateau, cosine=CosineAnnealingLR, none.")
    parser.add_argument("--scheduler-patience", type=int, default=2,
                        help="Epochs without improvement before plateau scheduler reduces LR.")
    parser.add_argument("--label-column", default="target_actionTaken",
                        help="CSV column to use as classification label (default: target_actionTaken). "
                             "Use 'label_direction' for scalper pipeline.")
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

    label_column = args.label_column
    action_classes = compute_class_count(train_csv, label_column, args.train_limit_rows)
    trade_side_classes = compute_class_count(train_csv, "target_tradeSide", args.train_limit_rows) \
        if "target_tradeSide" in open(train_csv).readline() else 2

    print(
        f"Discovered sequence_length={sequence_length}, per_step_features={per_step_features}, "
        f"action_classes={action_classes}, trade_side_classes={trade_side_classes}, "
        f"label_column={label_column}"
    )

    norm_mean = norm_std = None
    if args.normalization_stats:
        import json as _json
        stats = _json.loads(Path(args.normalization_stats).read_text())
        col_to_mean = dict(zip(stats["feature_columns"], stats["mean"]))
        col_to_std  = dict(zip(stats["feature_columns"], stats["std"]))
        norm_mean = torch.tensor([col_to_mean.get(c, 0.0) for c in feature_columns], dtype=torch.float32)
        norm_std  = torch.tensor([col_to_std.get(c, 1.0)  for c in feature_columns], dtype=torch.float32)
        print(f"Normalisatie geladen: {args.normalization_stats}")

    train_loader = build_dataloader(
        DatasetSpec(train_csv, args.train_limit_rows),
        feature_columns, sequence_length, per_step_features,
        args.batch_size, args.num_workers, norm_mean, norm_std, label_column,
    )
    validation_loader = build_dataloader(
        DatasetSpec(validation_csv, args.validation_limit_rows),
        feature_columns, sequence_length, per_step_features,
        args.batch_size, args.num_workers, norm_mean, norm_std, label_column,
    )
    test_loader = build_dataloader(
        DatasetSpec(test_csv, args.test_limit_rows),
        feature_columns, sequence_length, per_step_features,
        args.batch_size, args.num_workers, norm_mean, norm_std, label_column,
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

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    else:
        scheduler = None

    tb_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else checkpoint_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard: {tb_dir}")

    best_val_accuracy = -1.0

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    print(f"Loss: FocalLoss(gamma={args.focal_gamma}), class_weights={args.class_weights}")
    ce_action = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    ce_trade_side = FocalLoss(gamma=args.focal_gamma, weight=None)
    regression_loss = nn.SmoothL1Loss(beta=1.0)

    global_step = 0
    start_epoch = 1

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        payload = torch.load(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if not args.reset_optimizer:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        else:
            print("Optimizer state reset (--reset-optimizer)")
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
                avg_loss = running_loss / running_batches
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={avg_loss:.6f} "
                    f"loss_action={loss_action.item():.6f} "
                    f"loss_tradeSide={loss_trade_side.item():.6f} "
                    f"loss_equity={loss_equity.item():.6f}"
                )
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/loss_action", loss_action.item(), global_step)
                writer.add_scalar("train/loss_trade_side", loss_trade_side.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

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
                writer.add_scalar("val/loss", val_metrics["loss"], global_step)
                writer.add_scalar("val/action_accuracy", val_metrics["action_accuracy"], global_step)
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

        writer.add_scalar("epoch/train_loss", epoch_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", validation_metrics["loss"], epoch)
        writer.add_scalar("epoch/val_action_accuracy", validation_metrics["action_accuracy"], epoch)
        writer.add_scalar("epoch/test_action_accuracy", test_metrics["action_accuracy"], epoch)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()
            print(f"LR after scheduler: {optimizer.param_groups[0]['lr']:.2e}")

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

        val_acc = validation_metrics["action_accuracy"]
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_path = checkpoint_dir / "checkpoint_best.pt"
            torch.save(torch.load(checkpoint_path, map_location="cpu"), best_path)
            print(f"New best val_accuracy={val_acc:.4f} → saved {best_path}")

    writer.close()
    print("Training complete.")
    print_vram("Training complete")


if __name__ == "__main__":
    main()
