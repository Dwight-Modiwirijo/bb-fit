#!/usr/bin/env python3
"""
Backtest script voor BB-fit LSTM.

Simuleert kapitaalgroei op de testset door het model live te laten handelen.
TP/SL worden gesimuleerd op basis van het true label:
  - model=LONG + true=LONG  → TP geraakt (+tp_pct)
  - model=LONG + true=SHORT → SL geraakt (-sl_pct)
  - model=SHORT + true=SHORT → TP geraakt (+tp_pct)
  - model=SHORT + true=LONG  → SL geraakt (-sl_pct)
  - model=HOLD of true=HOLD  → geen trade

Naast de standaard argmax run ook een threshold sweep:
voor elke hold_threshold T:
  als P(hold) >= T → hold, anders → short of long op basis van hoogste kans.

Usage:
    python backtest_lstm_bbfit.py \
        --test-csv  /workspace/data/sequences_tpsl_v2/lstm_test_sequences.csv \
        --checkpoint /workspace/checkpoints/lstm_bbfit/tpsl_v2_warmup/checkpoint_epoch05_step0001234.pt \
        --hidden-size 512 --num-layers 3 --dropout 0.1 \
        --initial-capital 10000 --tp-pct 0.036 --sl-pct 0.012 --fee 0.0018 \
        --output-json /workspace/data/backtest_result.json \
        --output-csv  /workspace/data/backtest_equity.csv
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
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


class SequenceCsvIterableDataset(IterableDataset):
    def __init__(self, path: Path, feature_columns: Sequence[str],
                 sequence_length: int, per_step_features: int,
                 limit_rows: Optional[int] = None) -> None:
        super().__init__()
        self.path = path
        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.per_step_features = per_step_features
        self.limit_rows = limit_rows

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, int, str]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if self.limit_rows and idx >= self.limit_rows:
                    break
                if (idx % num_workers) != worker_id:
                    continue
                features = [float(row[c]) for c in self.feature_columns]
                x = torch.tensor(features, dtype=torch.float32).view(
                    self.sequence_length, self.per_step_features)
                true_label = int(float(row["target_actionTaken"]))
                timestamp = row.get("targetTimestamp", str(idx))
                yield x, true_label, timestamp


class MultiHeadLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 action_classes=3, trade_side_classes=3):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=lstm_dropout,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(hidden_size, action_classes)
        self.trade_side_head = nn.Linear(hidden_size, trade_side_classes)
        self.net_equity_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return self.action_head(h)


def discover_columns(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        header = next(csv.reader(f))
    feature_cols = [c for c in header if c not in META_COLUMNS and c not in TARGET_COLUMNS]
    return feature_cols


def infer_sequence_length(feature_columns):
    prefixes = set()
    for col in feature_columns:
        if len(col) > 5 and col[0] == "t" and col[1:4].isdigit() and col[4] == "_":
            prefixes.add(col[:4])
    seq_len = len(prefixes)
    per_step = len(feature_columns) // seq_len
    return seq_len, per_step


def simulate_trades(all_probs: torch.Tensor, all_true: List[int],
                    all_timestamps: List[str], hold_threshold: float,
                    initial_capital: float, tp_pct: float, sl_pct: float,
                    fee: float):
    capital = initial_capital
    equity_curve = []
    trades = []

    hold_probs = all_probs[:, 1]
    non_hold_probs = torch.stack([all_probs[:, 0], all_probs[:, 2]], dim=1)
    non_hold_argmax = non_hold_probs.argmax(dim=1)
    non_hold_classes = [0, 2]

    for i in range(len(all_true)):
        hp = hold_probs[i].item()
        if hp >= hold_threshold:
            pred = 1
        else:
            pred = non_hold_classes[non_hold_argmax[i].item()]

        true = all_true[i]
        ts = all_timestamps[i]
        pnl = 0.0
        outcome = "hold"

        if pred == 2 and true == 2:   # long correct → TP
            pnl = capital * (tp_pct - 2 * fee)
            outcome = "long_win"
        elif pred == 2 and true == 0: # long wrong → SL
            pnl = -capital * (sl_pct + 2 * fee)
            outcome = "long_loss"
        elif pred == 0 and true == 0: # short correct → TP
            pnl = capital * (tp_pct - 2 * fee)
            outcome = "short_win"
        elif pred == 0 and true == 2: # short wrong → SL
            pnl = -capital * (sl_pct + 2 * fee)
            outcome = "short_loss"

        capital += pnl
        equity_curve.append((ts, capital))
        if outcome != "hold":
            trades.append({"timestamp": ts, "outcome": outcome, "pnl": round(pnl, 4), "capital": round(capital, 4)})

    wins = sum(1 for t in trades if "win" in t["outcome"])
    losses = sum(1 for t in trades if "loss" in t["outcome"])
    total = wins + losses
    growth = (capital / initial_capital - 1) * 100

    return {
        "hold_threshold": hold_threshold,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 4),
        "growth_pct": round(growth, 2),
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def collect_probs_and_labels(model, dataloader, device):
    all_probs, all_true, all_ts = [], [], []
    model.eval()
    with torch.no_grad():
        for x, true_labels, timestamps in dataloader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu()
            all_probs.append(probs)
            all_true.extend(true_labels.tolist())
            all_ts.extend(timestamps)
    return torch.cat(all_probs, dim=0), all_true, all_ts


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    timestamps = [b[2] for b in batch]
    return xs, labels, timestamps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--initial-capital", type=float, default=10000.0)
    p.add_argument("--tp-pct", type=float, default=0.036)
    p.add_argument("--sl-pct", type=float, default=0.012)
    p.add_argument("--fee", type=float, default=0.0018)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--limit-rows", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_csv = Path(args.test_csv)
    feature_columns = discover_columns(test_csv)
    seq_len, per_step = infer_sequence_length(feature_columns)
    print(f"sequence_length={seq_len}, per_step_features={per_step}, total_features={len(feature_columns)}")

    dataset = SequenceCsvIterableDataset(test_csv, feature_columns, seq_len, per_step, args.limit_rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=collate_fn)

    model = MultiHeadLstm(per_step, args.hidden_size, args.num_layers, args.dropout).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    print(f"Checkpoint geladen: {args.checkpoint}")

    print("Verzamelen van model output ...")
    all_probs, all_true, all_ts = collect_probs_and_labels(model, loader, device)
    print(f"  {len(all_true):,} sequenties verwerkt.")

    thresholds = [round(t * 0.05, 2) for t in range(4, 21)]  # 0.20 .. 1.00
    results = []
    print(f"\n{'thresh':>7}  {'final_cap':>12}  {'growth%':>8}  {'trades':>7}  {'wins':>6}  {'losses':>6}  {'winrate':>8}")
    for thresh in thresholds:
        r = simulate_trades(all_probs, all_true, all_ts, thresh,
                            args.initial_capital, args.tp_pct, args.sl_pct, args.fee)
        results.append(r)
        print(f"{thresh:>7.2f}  {r['final_capital']:>12,.2f}  {r['growth_pct']:>8.1f}%  "
              f"{r['total_trades']:>7}  {r['wins']:>6}  {r['losses']:>6}  {r['win_rate']:>7.1f}%")

    best = max(results, key=lambda x: x["final_capital"])
    print(f"\nBeste drempel: {best['hold_threshold']} → kapitaal {best['final_capital']:,.2f} ({best['growth_pct']:+.1f}%)")

    if args.output_json:
        out = {"checkpoint": str(args.checkpoint), "settings": vars(args), "sweep": [
            {k: v for k, v in r.items() if k not in ("equity_curve", "trades")}
            for r in results
        ], "best": {k: v for k, v in best.items() if k not in ("equity_curve", "trades")}}
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"JSON opgeslagen: {args.output_json}")

    if args.output_csv:
        best_result = best
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "capital"])
            for ts, cap in best_result["equity_curve"]:
                w.writerow([ts, round(cap, 4)])
        print(f"Equity curve opgeslagen: {args.output_csv}")


if __name__ == "__main__":
    main()
