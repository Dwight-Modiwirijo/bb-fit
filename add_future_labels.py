"""
add_future_labels.py — adds supervised learning labels to lstm_merged.csv.

New columns logged by C# (v2 TrainingRow):
  sma, ema, upper_band, lower_band, deviation, band_width, band_width_delta,
  rsi, stoch_rsi, shortAssetsHeld, long_entry_price, short_entry_price,
  unrealized_pnl, bars_since_entry, bars_since_last_trade, bb_position,
  longEntryCount, longExitCount, shortEntryCount, shortExitCount,
  long_profit_loss, short_profit_loss

Added by this script (labels only — NEVER use as model input features):
  future_return_3, future_return_6, future_return_12, future_return_24
  future_direction_12
  future_return_label_12
  bot_signal_quality_12

WARNING: future_return_* and future_return_label_* are supervised learning
labels. They encode lookahead information and must NEVER be used as input
features for the model.
"""

import argparse
import pandas as pd
import numpy as np

# Price column used as reference for future return calculation.
# signalClose = the close price of the bar on which the signal was generated
# (look-ahead free: the signal bar is bar j-1, executed on bar j).
PRICE_COL = "signalClose"

# Horizons in candles. For 5-min candles: 12 = 60 min, 24 = 120 min.
DEFAULT_HORIZONS = [3, 6, 12, 24]

# Thresholds after fees/slippage (~0.18% round-trip fee at 0.09% per side).
POSITIVE_THRESHOLD = 0.003
NEGATIVE_THRESHOLD = 0.003

LABEL_HORIZON = 12  # horizon used for direction/label/quality columns


def add_future_labels(
    df: pd.DataFrame,
    horizons: list[int] = DEFAULT_HORIZONS,
    positive_threshold: float = POSITIVE_THRESHOLD,
    negative_threshold: float = NEGATIVE_THRESHOLD,
    label_horizon: int = LABEL_HORIZON,
    price_col: str = PRICE_COL,
) -> pd.DataFrame:
    """
    Adds future return labels to df in-place.

    Assumes df is already sorted by timestamp within each runId group.
    Shifts are computed per runId so that returns never cross run boundaries.
    """
    df = df.copy()
    price = df[price_col].astype(float)

    # Future return per horizon, grouped by runId to prevent cross-run leakage.
    for h in horizons:
        col = f"future_return_{h}"
        df[col] = (
            df.groupby("runId")[price_col]
            .transform(lambda s: (s.shift(-h) - s) / s)
            .astype(float)
        )

    # Direction label for the primary horizon.
    fr = df[f"future_return_{label_horizon}"]
    df[f"future_direction_{label_horizon}"] = np.where(
        fr > positive_threshold, 1,
        np.where(fr < -negative_threshold, -1, 0)
    ).astype("Int8")

    # Classification label (string).
    df[f"future_return_label_{label_horizon}"] = np.where(
        fr > positive_threshold, "long",
        np.where(fr < -negative_threshold, "short", "hold")
    )

    # Bot signal quality: was actionTaken correct in hindsight?
    # Long-only bot: actionTaken=1 → long entry, actionTaken=-1 → long exit.
    # If dataset includes shorts: use inPosition / shortEntryCount to disambiguate.
    action = df["actionTaken"].astype(int)
    has_short_context = "shortEntryCount" in df.columns

    if has_short_context:
        # actionTaken=-1 can mean long-exit OR short-entry; use shortEntryCount delta.
        short_entry_mask = (action == -1) & (df["shortEntryCount"].diff().fillna(0) > 0)
        long_exit_mask = (action == -1) & ~short_entry_mask
    else:
        short_entry_mask = pd.Series(False, index=df.index)
        long_exit_mask = action == -1

    quality = pd.Series("neutral", index=df.index)
    quality[( action == 1) & (fr >  positive_threshold)] = "good_long"
    quality[( action == 1) & (fr < -negative_threshold)] = "bad_long"
    quality[(short_entry_mask) & (fr < -negative_threshold)] = "good_short"
    quality[(short_entry_mask) & (fr >  positive_threshold)] = "bad_short"
    df[f"bot_signal_quality_{label_horizon}"] = quality

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add future-return labels to lstm_merged.csv."
    )
    parser.add_argument("--input",  required=True, help="Path to lstm_merged.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS,
        help="Candle horizons for future return (default: 3 6 12 24)",
    )
    parser.add_argument(
        "--label-horizon", type=int, default=LABEL_HORIZON,
        help="Horizon used for direction/label/quality columns (default: 12)",
    )
    parser.add_argument(
        "--positive-threshold", type=float, default=POSITIVE_THRESHOLD,
        help="Minimum return to label as long (default: 0.003)",
    )
    parser.add_argument(
        "--negative-threshold", type=float, default=NEGATIVE_THRESHOLD,
        help="Minimum negative return to label as short (default: 0.003)",
    )
    parser.add_argument(
        "--price-col", default=PRICE_COL,
        help=f"Price column for return calculation (default: {PRICE_COL})",
    )
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Sort chronologically within each run to ensure correct shift direction.
    df = df.sort_values(["runId", "timestamp"]).reset_index(drop=True)

    df = add_future_labels(
        df,
        horizons=args.horizons,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        label_horizon=args.label_horizon,
        price_col=args.price_col,
    )

    label_cols = (
        [f"future_return_{h}" for h in args.horizons]
        + [f"future_direction_{args.label_horizon}"]
        + [f"future_return_label_{args.label_horizon}"]
        + [f"bot_signal_quality_{args.label_horizon}"]
    )

    print("\nLabel column summary:")
    for col in label_cols:
        n_nan = df[col].isna().sum()
        print(f"  {col}: {n_nan:,} NaN (last {n_nan} rows per run have no future data)")

    print(f"\nSaving to {args.output} ...")
    df.to_csv(args.output, index=False)
    print("Done.")

    print("\n--- Column guide ---")
    print("FEATURES  (safe as model input):")
    feature_cols = [c for c in df.columns if c not in label_cols]
    print(f"  {len(feature_cols)} columns — all original + new C# columns")
    print("\nLABELS  (supervised targets only — never use as model input):")
    for col in label_cols:
        print(f"  {col}")

    print("\nWARNING: future_return_* and future_return_label_* contain")
    print("lookahead information. Never pass them as input features to the model.")


if __name__ == "__main__":
    main()
