"""
rule_baseline_v8.py — deterministic rule baseline on the merged features CSV.

Purpose: measure how much of the bot's actionTaken can be predicted by simple rules
derived from the same features the LSTM receives. Run this BEFORE training v8.

  If baseline >> LSTM long_precision → training/model is the bottleneck.
  If baseline ≈ LSTM long_precision  → the target/data itself is the ceiling.

Rule (mirrors bot logic from TAengine):
  LONG  (class 2): lastTrade=1  AND inPosition=0   (buy signal, not yet in position)
  SHORT (class 0): lastTrade=2  AND inPosition=1   (sell signal, closing long)
  HOLD  (class 1): all other rows

Optionally filter to is_valid_signal=True rows only (--valid-only flag).

Usage:
  python rule_baseline_v8.py --input /home/dwyte/bb-fit/lstm_merged_v8.csv
  python rule_baseline_v8.py --input /home/dwyte/bb-fit/lstm_merged_v8.csv --valid-only
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


CHUNK_SIZE = 300_000

# actionTaken values in the raw merged CSV (-1/0/1); remap for display
LABEL_MAP = {-1: "SHORT(0)", 0: "HOLD(1)", 1: "LONG(2)"}


def apply_rule(df: pd.DataFrame) -> pd.Series:
    pred = pd.Series(0, index=df.index, dtype=int)  # default: HOLD
    pred[(df["lastTrade"] == 1) & (df["inPosition"] == 0)] = 1   # LONG signal
    pred[(df["lastTrade"] == 2) & (df["inPosition"] == 1)] = -1  # SHORT signal (close)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule baseline for v8")
    parser.add_argument("--input", default="/home/dwyte/bb-fit/lstm_merged_v8.csv",
                        help="Merged features CSV (v7 or v8)")
    parser.add_argument("--valid-only", action="store_true",
                        help="Restrict to is_valid_signal=True rows only")
    args = parser.parse_args()

    in_path = Path(args.input)
    print(f"Input:      {in_path}")
    print(f"Valid-only: {args.valid_only}")
    print()

    all_true: list[int] = []
    all_pred: list[int] = []
    total_rows = 0
    skipped = 0

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
        if args.valid_only and "is_valid_signal_int" in chunk.columns:
            chunk = chunk[chunk["is_valid_signal_int"] == 1]
        elif args.valid_only and "is_valid_signal" in chunk.columns:
            chunk = chunk[chunk["is_valid_signal"] == True]

        if chunk.empty:
            continue

        # actionTaken = label column (-1/0/1 in merged CSV)
        if "actionTaken" not in chunk.columns:
            skipped += len(chunk)
            continue

        total_rows += len(chunk)
        y_true = chunk["actionTaken"].astype(int)
        y_pred = apply_rule(chunk)

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    if not all_true:
        print("No rows found. Check --input path.")
        return

    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)

    # Label distribution
    print(f"Total rows evaluated: {total_rows:,}")
    for val, name in LABEL_MAP.items():
        count = int((all_true_np == val).sum())
        print(f"  {name}: {count:,} ({100*count/total_rows:.1f}%)")
    print()

    # Classification report (scikit-learn style)
    labels = [-1, 0, 1]
    target_names = ["SHORT(close)", "HOLD", "LONG"]
    print(classification_report(all_true_np, all_pred_np, labels=labels, target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_true_np, all_pred_np, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"{'':15s} {'SHORT':>8s} {'HOLD':>8s} {'LONG':>8s}")
    for i, name in enumerate(["SHORT(true)", "HOLD(true)", "LONG(true)"]):
        print(f"  {name:13s} {cm[i,0]:>8,} {cm[i,1]:>8,} {cm[i,2]:>8,}")
    print()

    # Key metrics summary
    from sklearn.metrics import precision_score, recall_score
    long_prec = precision_score(all_true_np, all_pred_np, labels=[1], average=None, zero_division=0)[0]
    long_rec  = recall_score(   all_true_np, all_pred_np, labels=[1], average=None, zero_division=0)[0]
    short_prec = precision_score(all_true_np, all_pred_np, labels=[-1], average=None, zero_division=0)[0]
    short_rec  = recall_score(   all_true_np, all_pred_np, labels=[-1], average=None, zero_division=0)[0]

    print("=" * 50)
    print("Key metrics (same as orchestrator DoD):")
    print(f"  Long  precision: {long_prec:.1%}  (DoD target: 75.0%,  break-even: 69.2%)")
    print(f"  Long  recall:    {long_rec:.1%}  (DoD target: 15.0%)")
    print(f"  Short precision: {short_prec:.1%}  (DoD target: 10.0%  — sanity check)")
    print(f"  Short recall:    {short_rec:.1%}")
    print("=" * 50)
    print()

    if long_prec >= 0.70:
        print(">> Baseline meets/exceeds break-even (69.2%). Gap vs LSTM is a TRAINING issue.")
    else:
        print(">> Baseline is below break-even. Ceiling may be in data/target quality.")


if __name__ == "__main__":
    main()
