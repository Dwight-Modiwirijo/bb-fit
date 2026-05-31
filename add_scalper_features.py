"""
add_scalper_features.py — filter and normalise scalper JSONL CSV for LSTM training.

Differences vs add_v6_features.py:
  - Default interval: TwoHunderdAndFourty (240-min candles)
  - Passes through future_return_5/10/20 from C# bot
  - Drops rows where future_return_10 is null (last N rows at end of file)
  - Creates label_direction: 0=Down, 1=Flat, 2=Up  (threshold: --threshold, default 2%)
  - Same v6 normalisation for all indicator features

Input:  lstm_merged_scalper.csv  (output of parse_testlog_to_csv.py)
Output: lstm_merged_scalper_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 200_000


def add_features(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sc = df["signalClose"].replace(0.0, np.nan)

    df["sma_r"]        = (df["sma"]        / sc).fillna(1.0).clip(0.5, 2.0)
    df["ema_r"]        = (df["ema"]        / sc).fillna(1.0).clip(0.5, 2.0)
    df["upper_band_r"] = (df["upper_band"] / sc).fillna(1.0).clip(0.5, 2.0)
    df["lower_band_r"] = (df["lower_band"] / sc).fillna(1.0).clip(0.5, 2.0)
    df["deviation_r"]  = (df["deviation"]  / sc).fillna(0.0).clip(0.0, 0.5)
    df["rsi_norm"]     = (df["rsi"] / 100.0).clip(0.0, 1.0)
    df["entryPrice_r"] = (df["long_entry_price"] / sc).fillna(0.0).clip(0.0, 2.0)

    df["bars_since_entry_norm"]      = np.tanh(df["bars_since_entry"]      / 20.0)
    df["bars_since_last_trade_norm"] = np.tanh(df["bars_since_last_trade"] / 20.0)
    df["unrealized_pnl_r"]          = (df["unrealized_pnl"] / sc.abs()).fillna(0.0).clip(-1.0, 1.0)

    r10 = pd.to_numeric(df["future_return_10"], errors="coerce")
    df["label_direction"] = np.where(r10 > threshold, 2,
                            np.where(r10 < -threshold, 0, 1)).astype(int)
    df["future_return_10"] = r10

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add scalper normalised features")
    parser.add_argument("--input",     default="/home/dwyte/bb-fit/lstm_merged_scalper.csv")
    parser.add_argument("--output",    default="/home/dwyte/bb-fit/lstm_merged_scalper_features.csv")
    parser.add_argument("--interval",  default="TwoHunderdAndFourty",
                        help="Keep only rows with this interval value (empty = keep all)")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="future_return_10 threshold for Up/Down label (default 0.02 = 2%%)")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    print(f"Input:     {in_path}")
    print(f"Output:    {out_path}")
    print(f"Interval:  {args.interval or '(all)'}")
    print(f"Threshold: ±{args.threshold*100:.1f}%  (label_direction: 0=Down, 1=Flat, 2=Up)")

    total_in = total_out = null_dropped = warmup_dropped = 0
    label_counts = {0: 0, 1: 0, 2: 0}
    first_chunk = True

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_in += len(chunk)

        if args.interval:
            chunk = chunk[chunk["interval"] == args.interval]
        if chunk.empty:
            continue

        # Drop TAengine warmup rows — rsi not required (scalper bot may not log it)
        before_warmup = len(chunk)
        chunk = chunk[(chunk["sma"] != 0) & (chunk["upper_band"] != 0)]
        warmup_dropped += before_warmup - len(chunk)
        if chunk.empty:
            continue

        # Drop rows where future_return_10 is null (last N rows at end of dataset)
        before_null = len(chunk)
        chunk = chunk[chunk["future_return_10"].notna()]
        null_dropped += before_null - len(chunk)
        if chunk.empty:
            continue

        chunk = add_features(chunk, args.threshold)

        for lbl, cnt in chunk["label_direction"].value_counts().items():
            label_counts[int(lbl)] = label_counts.get(int(lbl), 0) + cnt

        total_out += len(chunk)
        chunk.to_csv(out_path, mode="w" if first_chunk else "a",
                     index=False, header=first_chunk)
        first_chunk = False

    total_labels = sum(label_counts.values()) or 1
    print(f"\nDone. {total_in:,} in → {total_out:,} out")
    print(f"  Warmup dropped:   {warmup_dropped:,}")
    print(f"  Null r10 dropped: {null_dropped:,}")
    print(f"  label_direction:")
    print(f"    Down (0): {label_counts[0]:6,}  ({label_counts[0]/total_labels*100:.1f}%)")
    print(f"    Flat (1): {label_counts[1]:6,}  ({label_counts[1]/total_labels*100:.1f}%)")
    print(f"    Up   (2): {label_counts[2]:6,}  ({label_counts[2]/total_labels*100:.1f}%)")


if __name__ == "__main__":
    main()
