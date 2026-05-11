"""
add_v6_features.py — filter lstm_merged.csv to FiveMinutes and add normalised
TAengine indicator columns for v6 LSTM training.

Input:  lstm_merged.csv  (60 columns, TAengine indicators already present)
Output: lstm_merged_v6.csv  (same rows, extra normalised columns)

New columns added (all LSTM-safe: bounded or tanh-scaled):
  sma_r, ema_r, upper_band_r, lower_band_r, deviation_r
  rsi_norm, stoch_rsi (pass-through), bb_position (pass-through)
  band_width (pass-through), band_width_delta (pass-through)
  bars_since_entry_norm, bars_since_last_trade_norm
  unrealized_pnl_r
  entryPrice_r  (long_entry_price / signalClose)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 200_000


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add v6 normalised features to lstm_merged.csv")
    parser.add_argument("--input",    default="/home/dwyte/bb-fit/lstm_merged.csv")
    parser.add_argument("--output",   default="/home/dwyte/bb-fit/lstm_merged_v6.csv")
    parser.add_argument("--interval", default="FiveMinutes",
                        help="Keep only rows with this interval value (default: FiveMinutes)")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    print(f"Input:    {in_path}")
    print(f"Output:   {out_path}")
    print(f"Interval: {args.interval}")

    total_in = total_out = 0
    first_chunk = True

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_in += len(chunk)

        if args.interval:
            chunk = chunk[chunk["interval"] == args.interval]

        if chunk.empty:
            continue

        # Filter out TAengine warmup rows (indicators not yet stable)
        chunk = chunk[(chunk["sma"] != 0) & (chunk["upper_band"] != 0) & (chunk["rsi"] != 0)]

        chunk = add_features(chunk)
        total_out += len(chunk)

        chunk.to_csv(out_path, mode="w" if first_chunk else "a",
                     index=False, header=first_chunk)
        first_chunk = False

        if total_in % (CHUNK_SIZE * 5) == 0 or total_in >= CHUNK_SIZE:
            print(f"  Processed {total_in:,} in → {total_out:,} out", flush=True)

    print(f"\nDone. {total_in:,} rows read → {total_out:,} rows written to {out_path}")


if __name__ == "__main__":
    main()
