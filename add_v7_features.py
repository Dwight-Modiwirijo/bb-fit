"""
add_v7_features.py — filter lstm_merged_v7_raw.csv and add normalised
TAengine + probe feature columns for v7 LSTM training.

Input:  lstm_merged_v7_raw.csv  (output of parse_testlog_to_csv.py on 14-year JSONL)
Output: lstm_merged_v7.csv

v7 adds over v6:
  is_valid_signal_int   — 0/1: bot considers this candle for trading
  bb_tweak_buy          — BB buy multiplier flag (0/1, pass-through)
  bb_tweak_sell         — BB sell multiplier flag (0/1, pass-through)
  probe_buy_count_norm  — tanh(probe_buy_count / 10)
  probe_sell_count_norm — tanh(probe_sell_count / 10)
  probe_growth_norm     — tanh(probe_growth_per_month / 5)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 200_000


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    sc = df["signalClose"].replace(0.0, np.nan)

    # v6 price-ratio features
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

    # v7 new features
    df["is_valid_signal_int"]   = df["is_valid_signal"].astype(int) if "is_valid_signal" in df.columns else 0
    df["bb_tweak_buy"]          = df["bb_tweak_buy"].fillna(0.0)   if "bb_tweak_buy"   in df.columns else 0.0
    df["bb_tweak_sell"]         = df["bb_tweak_sell"].fillna(0.0)  if "bb_tweak_sell"  in df.columns else 0.0
    df["probe_buy_count_norm"]  = np.tanh(df["probe_buy_count"].fillna(0.0)  / 10.0) if "probe_buy_count"        in df.columns else 0.0
    df["probe_sell_count_norm"] = np.tanh(df["probe_sell_count"].fillna(0.0) / 10.0) if "probe_sell_count"       in df.columns else 0.0
    df["probe_growth_norm"]     = np.tanh(df["probe_growth_per_month"].fillna(0.0) / 5.0) if "probe_growth_per_month" in df.columns else 0.0

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add v7 normalised features")
    parser.add_argument("--input",    default="/home/dwyte/bb-fit/lstm_merged_v7_raw.csv")
    parser.add_argument("--output",   default="/home/dwyte/bb-fit/lstm_merged_v7.csv")
    parser.add_argument("--interval", default="FiveMinutes")
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

        chunk = chunk[(chunk["sma"] != 0) & (chunk["upper_band"] != 0) & (chunk["rsi"] != 0)]
        if chunk.empty:
            continue

        chunk = add_features(chunk)
        total_out += len(chunk)

        chunk.to_csv(out_path, mode="w" if first_chunk else "a",
                     index=False, header=first_chunk)
        first_chunk = False

        if total_in % (CHUNK_SIZE * 5) == 0 or total_in == CHUNK_SIZE:
            print(f"  Processed {total_in:,} in → {total_out:,} out", flush=True)

    print(f"\nDone. {total_in:,} rows read → {total_out:,} rows written to {out_path}")


if __name__ == "__main__":
    main()
