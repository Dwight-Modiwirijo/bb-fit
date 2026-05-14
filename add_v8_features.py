"""
add_v8_features.py — filter lstm_merged_v7_raw.csv and add normalised features for v8.

v8 vs v7:
  - Keeps force_hold_training_row=True rows in output (for sequence context).
    The sequence builder filters them from *targets* via is_valid_signal_int=0.
  - Same 36 features as v7 (kelly/kelly_fraction_used were never in the feature set).
  - Preserves temporal continuity: no gaps from dropped rows in sequence windows.

Input:  lstm_merged_v7_raw.csv
Output: lstm_merged_v8.csv
"""

import argparse
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

    df["is_valid_signal_int"]   = df["is_valid_signal"].astype(int) if "is_valid_signal" in df.columns else 0
    df["bb_tweak_buy"]          = df["bb_tweak_buy"].fillna(0.0)   if "bb_tweak_buy"   in df.columns else 0.0
    df["bb_tweak_sell"]         = df["bb_tweak_sell"].fillna(0.0)  if "bb_tweak_sell"  in df.columns else 0.0
    df["probe_buy_count_norm"]  = np.tanh(df["probe_buy_count"].fillna(0.0)  / 10.0) if "probe_buy_count"        in df.columns else 0.0
    df["probe_sell_count_norm"] = np.tanh(df["probe_sell_count"].fillna(0.0) / 10.0) if "probe_sell_count"       in df.columns else 0.0
    df["probe_growth_norm"]     = np.tanh(df["probe_growth_per_month"].fillna(0.0) / 5.0) if "probe_growth_per_month" in df.columns else 0.0

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Add v8 normalised features (keeps force_hold rows)")
    parser.add_argument("--input",    default="/home/dwyte/bb-fit/lstm_merged_v7_raw.csv")
    parser.add_argument("--output",   default="/home/dwyte/bb-fit/lstm_merged_v8.csv")
    parser.add_argument("--interval", default="FiveMinutes")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    print(f"Input:    {in_path}")
    print(f"Output:   {out_path}")
    print(f"Interval: {args.interval}")
    print("v8: force_hold rows kept for sequence context (sequence builder filters them as targets)")

    total_in = total_out = 0
    force_hold_count = 0
    real_count = 0
    first_chunk = True

    for chunk in pd.read_csv(in_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_in += len(chunk)

        if args.interval:
            chunk = chunk[chunk["interval"] == args.interval]
        if chunk.empty:
            continue

        # Warmup filter: indicators are unreliable before ~30-50 candles of TAengine warmup
        chunk = chunk[(chunk["sma"] != 0) & (chunk["upper_band"] != 0) & (chunk["rsi"] != 0)]
        if chunk.empty:
            continue

        # Track force_hold vs real rows for diagnostics
        if "force_hold_training_row" in chunk.columns:
            force_hold_count += int((chunk["force_hold_training_row"] == True).sum())
            real_count        += int((chunk["force_hold_training_row"] != True).sum())
        elif "is_valid_signal" in chunk.columns:
            force_hold_count += int((chunk["is_valid_signal"] == False).sum())
            real_count        += int((chunk["is_valid_signal"] == True).sum())

        chunk = add_features(chunk)
        total_out += len(chunk)

        chunk.to_csv(out_path, mode="w" if first_chunk else "a",
                     index=False, header=first_chunk)
        first_chunk = False

        if total_in % (CHUNK_SIZE * 5) == 0 or total_in == CHUNK_SIZE:
            print(f"  Processed {total_in:,} in → {total_out:,} out", flush=True)

    pct_force = 100.0 * force_hold_count / total_out if total_out else 0.0
    print(f"\nDone. {total_in:,} rows read → {total_out:,} rows written to {out_path}")
    print(f"  Real bot decisions (is_valid_signal=True):  {real_count:,} ({100-pct_force:.1f}%)")
    print(f"  Force-hold rows   (is_valid_signal=False): {force_hold_count:,} ({pct_force:.1f}%)")
    print(f"  Sequence builder will skip force-hold rows as targets (is_valid_signal_int=0).")


if __name__ == "__main__":
    main()
