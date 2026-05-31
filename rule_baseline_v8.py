"""
rule_baseline_v8.py — deterministic rule baseline on the merged features CSV.

Purpose: measure how much of the bot's actionTaken can be predicted by simple rules
derived from the same features the LSTM receives. Run this BEFORE training v8.

  If baseline >> LSTM long_precision → training/model is the bottleneck.
  If baseline ≈ LSTM long_precision  → the target/data itself is the ceiling.

Rule (mirrors bot logic from TAengine):
  LONG  (actionTaken=1):  lastTrade=1 AND inPosition=0  (buy signal, not in position)
  SHORT (actionTaken=-1): lastTrade=2 AND inPosition=1  (sell signal, closing long)
  HOLD  (actionTaken=0):  all other rows

Optionally filter to is_valid_signal_int=1 rows only (--valid-only flag).

Usage:
  python rule_baseline_v8.py --input /workspace/data/lstm_merged_v8.csv
  python rule_baseline_v8.py --input /workspace/data/lstm_merged_v8.csv --valid-only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 300_000


def apply_rule(df: pd.DataFrame) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)  # default: HOLD
    pred[(df["lastTrade"] == 1) & (df["inPosition"] == 0)] = 1   # LONG
    pred[(df["lastTrade"] == 2) & (df["inPosition"] == 1)] = -1  # SHORT (close long)
    return pred


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray, label: int):
    tp = int(((y_pred == label) & (y_true == label)).sum())
    fp = int(((y_pred == label) & (y_true != label)).sum())
    fn = int(((y_pred != label) & (y_true == label)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule baseline for v8")
    parser.add_argument("--input", default="/workspace/data/lstm_merged_v8.csv",
                        help="Merged features CSV (v7 or v8)")
    parser.add_argument("--valid-only", action="store_true",
                        help="Restrict to is_valid_signal_int=1 rows only")
    args = parser.parse_args()

    in_path = Path(args.input)
    print(f"Input:      {in_path}")
    print(f"Valid-only: {args.valid_only}")
    print()

    all_true: list = []
    all_pred: list = []
    total_rows = 0

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
        if args.valid_only:
            if "is_valid_signal_int" in chunk.columns:
                chunk = chunk[chunk["is_valid_signal_int"] == 1]
            elif "is_valid_signal" in chunk.columns:
                chunk = chunk[chunk["is_valid_signal"] == True]

        if chunk.empty or "actionTaken" not in chunk.columns:
            continue

        total_rows += len(chunk)
        all_true.extend(chunk["actionTaken"].astype(int).tolist())
        all_pred.extend(apply_rule(chunk).tolist())

    if not all_true:
        print("No rows found. Check --input path.")
        return

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    # Label distribution
    print(f"Total rows evaluated: {total_rows:,}")
    for val, name in [(-1, "SHORT (actionTaken=-1)"), (0, "HOLD (actionTaken=0)"), (1, "LONG (actionTaken=1)")]:
        count = int((y_true == val).sum())
        print(f"  {name}: {count:,} ({100*count/total_rows:.1f}%)")
    print()

    # Per-class metrics
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>10}")
    print("-" * 54)
    for label, name in [(-1, "SHORT"), (0, "HOLD"), (1, "LONG")]:
        prec, rec, f1, tp, fp, fn = precision_recall(y_true, y_pred, label)
        support = int((y_true == label).sum())
        print(f"  {name:<10} {prec:>10.1%} {rec:>10.1%} {f1:>8.1%} {support:>10,}")
    print()

    # Confusion matrix
    labels = [-1, 0, 1]
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"{'':18s} {'SHORT':>8s} {'HOLD':>8s} {'LONG':>8s}")
    for r_label, r_name in [(-1, "SHORT (actual)"), (0, "HOLD (actual)"), (1, "LONG (actual)")]:
        row = [int(((y_true == r_label) & (y_pred == c_label)).sum()) for c_label in labels]
        print(f"  {r_name:<16s} {row[0]:>8,} {row[1]:>8,} {row[2]:>8,}")
    print()

    long_prec, long_rec, _, _, _, _ = precision_recall(y_true, y_pred, 1)
    short_prec, short_rec, _, _, _, _ = precision_recall(y_true, y_pred, -1)

    print("=" * 54)
    print("Key metrics vs DoD targets:")
    print(f"  Long  precision: {long_prec:.1%}  (DoD: >=75%,  break-even: 69.2%)")
    print(f"  Long  recall:    {long_rec:.1%}  (DoD: >=15%)")
    print(f"  Short precision: {short_prec:.1%}  (DoD: >=10%  — sanity check only)")
    print(f"  Short recall:    {short_rec:.1%}")
    print("=" * 54)
    print()
    if long_prec >= 0.692:
        print(">> Rule baseline beats break-even (69.2%). Gap to LSTM is a TRAINING issue.")
    else:
        print(">> Rule baseline is below break-even. Target ceiling may be in the data.")


if __name__ == "__main__":
    main()
