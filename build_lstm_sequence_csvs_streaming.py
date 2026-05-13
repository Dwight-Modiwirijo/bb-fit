#!/usr/bin/env python3
"""
Build flattened train/validation/test sequence CSVs for LSTM training from lstm_merged.csv.

This version writes rows incrementally to disk instead of holding every generated sequence row in RAM.
That avoids the massive memory blow-up from accumulating all windows in Python lists.

Comments are intentionally written in English.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_NUMERIC_FEATURES = [
    # Candle timing
    "canonicalFee",
    "intervalMinutes",
    "observedIntervalMinutes",
    # Signal OHLC as intra-candle ratios to signalClose (all ~1.0, bounded)
    # signalClose itself is dropped — already captured by ind_ema55_ratio
    "signalOpen_r",    # signalOpen  / signalClose
    "signalHigh_r",    # signalHigh  / signalClose
    "signalLow_r",     # signalLow   / signalClose
    # Execution prices as ratios to signalClose (captures slippage / spread)
    "execOpen_r",      # executionOpen   / signalClose
    "execHigh_r",      # executionHigh   / signalClose
    "execLow_r",       # executionLow    / signalClose
    "execClose_r",     # executionClose  / signalClose
    "execPrice_r",     # executionPrice  / signalClose
    "entryPrice_r",    # entryPrice      / signalClose  (0 when not in position)
    # Bot signal state (bounded categorical / binary)
    "tradeActionRaw",
    "tradeSide",
    "lastTrade",
    "inPosition",
    # 12 technical indicators (all normalised ratios / bounded deltas)
    "ind_ema55_ratio",
    "ind_ema233_ratio",
    "ind_ema_trend",
    "ind_choppiness14",
    "ind_bb20_upper_ratio",
    "ind_bb20_lower_ratio",
    "ind_bb20_bandwidth",
    "ind_bb20_bw_delta",
    "ind_sma20_delta",
    "ind_sma20_delta2",
    "ind_rs14",
    "ind_rs14_delta",
    # BB-optimalisatie: welke BB-params waren het beste in de afgelopen 12u
    "bb_opt_period_norm",   # beste period (Fibonacci: 8/13/21/34/55) genormaliseerd [0,1]
    "bb_opt_bias_norm",     # beste stddev-multiplier (0.7-1.2) genormaliseerd [0,1]
    "bb_opt_growth",        # groeiperdag van de beste combo (tanh-geschaald ~[-1,1])
    # REMOVED: actionTaken (label leakage — this IS the prediction target)
    # REMOVED: tradingCapital, assetsHeld, positionValue, netEquity,
    #          buyCount, sellCount, wins, losses, totalTradedNotional,
    #          feePerSide, cost  (all grow unboundedly → NaN in LSTM)
    # REMOVED: raw signalClose / executionClose / entryPrice absolute values
    #          (BTC-scale prices ~60k saturate LSTM activations → NaN loss)
]

# v6: uses TAengine indicators directly (sma, ema, BB, RSI, stoch_rsi)
# normalised columns are pre-computed by add_v6_features.py
V6_NUMERIC_FEATURES = [
    "canonicalFee",
    "intervalMinutes",
    "observedIntervalMinutes",
    "signalOpen_r",
    "signalHigh_r",
    "signalLow_r",
    "execOpen_r",
    "execHigh_r",
    "execLow_r",
    "execClose_r",
    "execPrice_r",
    "entryPrice_r",       # long_entry_price / signalClose (added by add_v6_features.py)
    "tradeActionRaw",
    "tradeSide",
    "lastTrade",
    "inPosition",
    "sma_r",              # sma / signalClose
    "ema_r",              # ema / signalClose
    "upper_band_r",       # upper_band / signalClose
    "lower_band_r",       # lower_band / signalClose
    "deviation_r",        # deviation / signalClose
    "band_width",
    "band_width_delta",
    "rsi_norm",           # rsi / 100
    "stoch_rsi",
    "bb_position",
    "bars_since_entry_norm",
    "bars_since_last_trade_norm",
    "unrealized_pnl_r",
]

V6_CATEGORICAL_FEATURES = [
    "interval",
]

# v7: v6 features + is_valid_signal + bb_tweak + probe features (36 total per step)
V7_NUMERIC_FEATURES = [
    "canonicalFee", "intervalMinutes", "observedIntervalMinutes",
    "signalOpen_r", "signalHigh_r", "signalLow_r",
    "execOpen_r", "execHigh_r", "execLow_r", "execClose_r", "execPrice_r",
    "entryPrice_r", "tradeActionRaw", "tradeSide", "lastTrade", "inPosition",
    "sma_r", "ema_r", "upper_band_r", "lower_band_r", "deviation_r",
    "band_width", "band_width_delta", "rsi_norm", "stoch_rsi", "bb_position",
    "bars_since_entry_norm", "bars_since_last_trade_norm", "unrealized_pnl_r",
    "is_valid_signal_int",    # 0/1: bot considers this candle for trading
    "bb_tweak_buy",           # BB buy multiplier flag (0/1)
    "bb_tweak_sell",          # BB sell multiplier flag (0/1)
    "probe_buy_count_norm",   # tanh(count/10)
    "probe_sell_count_norm",  # tanh(count/10)
    "probe_growth_norm",      # tanh(growth_per_month/5)
]

V7_CATEGORICAL_FEATURES = [
    "interval",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "runGroup",
    "sourceFile",
    "interval",
    "splitHint",
]

DEFAULT_TARGETS = [
    "target_actionTaken",
    "target_tradeSide",
    "target_tradeActionRaw",
    "target_lastTrade",
    "target_netEquity",
    "target_netEquityDelta",
    "target_isNetEquityUp",
]

CANONICAL_SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "train": "train",
    "test": "test",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to lstm_merged.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files")
    parser.add_argument("--sequence-length", type=int, default=64, help="Window length")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio when splitHint is absent")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Validation split ratio when splitHint is absent")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio when splitHint is absent")
    parser.add_argument("--ignore-split-hint", action="store_true", default=False,
                        help="Ignore splitHint column and use proportional train/val/test split per run")
    parser.add_argument("--feature-set", default="default", choices=["default", "v6", "v7"],
                        help="Feature set: 'default' (legacy), 'v6' (normalised TAengine), or 'v7' (v6 + probe features)")
    return parser


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {c.lower(): c for c in df.columns}
    wanted = {
        "runid": "runId",
        "rungroup": "runGroup",
        "canonicalfee": "canonicalFee",
        "sourcefile": "sourceFile",
        "rowindexwithinrun": "rowIndexWithinRun",
        "splithint": "splitHint",
        "timestamp": "timestamp",
        "signaltimestamp": "signalTimestamp",
        "executiontimestamp": "executionTimestamp",
        "tradeactionraw": "tradeActionRaw",
        "tradeside": "tradeSide",
        "lasttrade": "lastTrade",
        "actiontaken": "actionTaken",
        "netequity": "netEquity",
    }
    rename_map = {}
    for key, target in wanted.items():
        if key in lookup and lookup[key] != target:
            rename_map[lookup[key]] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["runId", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_share = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_share > 0.95:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["timestamp", "signalTimestamp", "executionTimestamp"]:
        if col in df.columns:
            df[col] = parse_timestamp_series(df[col])
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        raise ValueError(f"{bad_ts} row(s) have invalid timestamp values")
    sort_cols = ["runId", "timestamp"]
    if "rowIndexWithinRun" in df.columns:
        df["rowIndexWithinRun"] = pd.to_numeric(df["rowIndexWithinRun"], errors="coerce")
        sort_cols.append("rowIndexWithinRun")
    return df.sort_values(sort_cols).reset_index(drop=True)


def canonicalize_split(value: object) -> str:
    text = str(value).strip().lower()
    return SPLIT_ALIASES.get(text, text)


def add_categorical_codes(df: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    df = df.copy()
    mappings: Dict[str, Dict[str, int]] = {}
    for col in categorical_features:
        if col not in df.columns:
            continue
        series = df[col].fillna("__MISSING__").astype(str)
        if col == "splitHint":
            series = series.map(canonicalize_split)
        unique_values = sorted(series.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        df[col] = series
        df[f"{col}_code"] = series.map(mapping).astype(int)
        mappings[col] = mapping
    return df, mappings


def prepare_features(df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, Dict[str, int]]]:
    df = df.copy()
    df, mappings = add_categorical_codes(df, categorical_features)
    feature_columns: List[str] = []

    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            feature_columns.append(col)

    for col in categorical_features:
        code_col = f"{col}_code"
        if code_col in df.columns:
            feature_columns.append(code_col)

    if feature_columns:
        df[feature_columns] = df[feature_columns].fillna(0.0)
    return df, feature_columns, mappings


def add_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise OHLC prices to ratios and clip unbounded indicator values.

    Raw BTC-scale prices (~60k) saturate LSTM activations and cause NaN loss.
    ind_rs14 (avgGain/avgLoss) diverges to 10^14 when avgLoss → 0 (long gain
    streaks). Clipping to [0, 100] maps to RSI [0, 99%] without losing signal.
    """
    df = df.copy()
    close = df["signalClose"].replace(0, float("nan"))

    # Price ratios (~1.0, bounded)
    df["signalOpen_r"]  = df["signalOpen"].fillna(0.0)       / close
    df["signalHigh_r"]  = df["signalHigh"].fillna(0.0)       / close
    df["signalLow_r"]   = df["signalLow"].fillna(0.0)        / close
    df["execOpen_r"]    = df["executionOpen"].fillna(0.0)    / close
    df["execHigh_r"]    = df["executionHigh"].fillna(0.0)    / close
    df["execLow_r"]     = df["executionLow"].fillna(0.0)     / close
    df["execClose_r"]   = df["executionClose"].fillna(0.0)   / close
    df["execPrice_r"]   = df["executionPrice"].fillna(0.0)   / close
    if "entryPrice" in df.columns:
        df["entryPrice_r"] = df["entryPrice"].fillna(0.0) / close
        df["entryPrice_r"] = df["entryPrice_r"].fillna(0.0)
    elif "entryPrice_r" not in df.columns:
        df["entryPrice_r"] = 0.0

    # Fill any remaining NaN from close==0 rows
    for col in ["signalOpen_r", "signalHigh_r", "signalLow_r",
                "execOpen_r", "execHigh_r", "execLow_r",
                "execClose_r", "execPrice_r"]:
        df[col] = df[col].fillna(0.0)

    # RS ratio diverges to 10^14 when avgLoss → 0; clip to RSI-equivalent range
    if "ind_rs14" in df.columns:
        df["ind_rs14"] = df["ind_rs14"].clip(0.0, 100.0)
    if "ind_rs14_delta" in df.columns:
        df["ind_rs14_delta"] = df["ind_rs14_delta"].clip(-100.0, 100.0)

    return df


def validate_feature_columns(df: pd.DataFrame, feature_columns: List[str]) -> None:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from dataframe: {missing}")


def segment_bounds(n: int, train_ratio: float, validation_ratio: float, test_ratio: float) -> Dict[str, Tuple[int, int]]:
    total = train_ratio + validation_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    train_end = max(0, min(n, int(math.floor(n * train_ratio))))
    val_end = max(train_end, min(n, int(math.floor(n * (train_ratio + validation_ratio)))))
    return {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, n),
    }


def build_header(sequence_length: int, feature_columns: List[str]) -> List[str]:
    header = [
        "runId",
        "split",
        "windowStartTimestamp",
        "windowEndTimestamp",
        "targetTimestamp",
        "sequenceLength",
    ]
    for t in range(sequence_length):
        for feature in feature_columns:
            header.append(f"t{t:03d}_{feature}")
    header.extend(DEFAULT_TARGETS)
    return header


def stream_segment_rows(
    run_df: pd.DataFrame,
    split_name: str,
    start_idx: int,
    end_idx: int,
    sequence_length: int,
    feature_columns: List[str],
    writer: csv.writer,
) -> int:
    written = 0
    segment = run_df.iloc[start_idx:end_idx].reset_index(drop=True)
    if len(segment) < sequence_length + 1:
        return 0

    segment_features = segment.reindex(columns=feature_columns, fill_value=0.0)

    for end_pos in range(sequence_length - 1, len(segment) - 1):
        window = segment.iloc[end_pos - sequence_length + 1 : end_pos + 1]
        window_features = segment_features.iloc[end_pos - sequence_length + 1 : end_pos + 1]
        target_row = segment.iloc[end_pos + 1]
        last_row = window.iloc[-1]

        current_equity = float(last_row.get("netEquity", 0.0))
        next_equity = float(target_row.get("netEquity", current_equity))
        delta = next_equity - current_equity

        out_row: List[object] = [
            str(last_row["runId"]),
            split_name,
            window.iloc[0]["timestamp"].isoformat(),
            last_row["timestamp"].isoformat(),
            target_row["timestamp"].isoformat(),
            sequence_length,
        ]
        out_row.extend(window_features.to_numpy().reshape(-1).tolist())
        out_row.extend([
            target_row.get("actionTaken", 0),
            target_row.get("tradeSide", 0),
            target_row.get("tradeActionRaw", 0.0),
            target_row.get("lastTrade", 0),
            next_equity,
            delta,
            int(delta > 0.0),
        ])
        writer.writerow(out_row)
        written += 1
    return written


def stream_rows_from_existing_split_hints(
    run_df: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    writers: Dict[str, csv.writer],
) -> Dict[str, int]:
    split_counts = {name: 0 for name in CANONICAL_SPLITS}
    split_series = run_df["splitHint"].fillna("__MISSING__").map(canonicalize_split)
    start = 0
    for idx in range(1, len(run_df) + 1):
        is_boundary = idx == len(run_df) or split_series.iloc[idx] != split_series.iloc[idx - 1]
        if not is_boundary:
            continue
        split_name = split_series.iloc[start]
        if split_name in writers:
            split_counts[split_name] += stream_segment_rows(
                run_df=run_df,
                split_name=split_name,
                start_idx=start,
                end_idx=idx,
                sequence_length=sequence_length,
                feature_columns=feature_columns,
                writer=writers[split_name],
            )
        start = idx
    return split_counts


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged CSV...")
    df = pd.read_csv(input_path)
    print(f"Loaded raw rows: {len(df):,}")

    df = normalize_columns(df)
    ensure_required_columns(df)
    print("Parsing timestamps and sorting...")
    df = parse_and_sort(df)
    print("Computing price ratios (normalise OHLC to signalClose)...")
    df = add_price_ratios(df)
    print("Preparing numeric/categorical features...")
    if args.feature_set == "v7":
        numeric_features     = V7_NUMERIC_FEATURES
        categorical_features = V7_CATEGORICAL_FEATURES
    elif args.feature_set == "v6":
        numeric_features     = V6_NUMERIC_FEATURES
        categorical_features = V6_CATEGORICAL_FEATURES
    else:
        numeric_features     = DEFAULT_NUMERIC_FEATURES
        categorical_features = DEFAULT_CATEGORICAL_FEATURES
    df, feature_columns, mappings = prepare_features(df, numeric_features, categorical_features)
    validate_feature_columns(df, feature_columns)

    header = build_header(args.sequence_length, feature_columns)
    use_split_hint = "splitHint" in df.columns and df["splitHint"].notna().any() and not args.ignore_split_hint
    run_count = int(df["runId"].nunique())
    print(f"Rows ready: {len(df):,}; runs: {run_count:,}; features per step: {len(feature_columns)}; splitHint={'yes' if use_split_hint else 'no'}")

    paths = {
        "train": output_dir / "lstm_train_sequences.csv",
        "validation": output_dir / "lstm_validation_sequences.csv",
        "test": output_dir / "lstm_test_sequences.csv",
    }

    output_counts = {name: 0 for name in CANONICAL_SPLITS}
    run_stats = []

    files = {name: path.open("w", newline="", encoding="utf-8") for name, path in paths.items()}
    try:
        writers = {name: csv.writer(files[name]) for name in CANONICAL_SPLITS}
        for name in CANONICAL_SPLITS:
            writers[name].writerow(header)

        for idx, (run_id, run_df) in enumerate(df.groupby("runId", sort=False), start=1):
            run_df = run_df.sort_values("timestamp").reset_index(drop=True)
            stats_entry = {
                "runId": str(run_id),
                "rows": int(len(run_df)),
                "usedSplitHint": bool(use_split_hint),
                "windows": {name: 0 for name in CANONICAL_SPLITS},
            }

            if use_split_hint:
                counts = stream_rows_from_existing_split_hints(
                    run_df=run_df,
                    sequence_length=args.sequence_length,
                    feature_columns=feature_columns,
                    writers=writers,
                )
                for split_name in CANONICAL_SPLITS:
                    stats_entry["windows"][split_name] = counts[split_name]
                    output_counts[split_name] += counts[split_name]
            else:
                bounds = segment_bounds(
                    len(run_df),
                    args.train_ratio,
                    args.validation_ratio,
                    args.test_ratio,
                )
                for split_name, (start_idx, end_idx) in bounds.items():
                    written = stream_segment_rows(
                        run_df=run_df,
                        split_name=split_name,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        sequence_length=args.sequence_length,
                        feature_columns=feature_columns,
                        writer=writers[split_name],
                    )
                    stats_entry["windows"][split_name] = written
                    output_counts[split_name] += written

            run_stats.append(stats_entry)

            if idx == 1 or idx % 10 == 0 or idx == run_count:
                total_windows = sum(output_counts.values())
                print(f"Processed {idx:,}/{run_count:,} run(s); cumulative windows={total_windows:,}")

    finally:
        for f in files.values():
            f.close()

    summary = {
        "input": str(input_path),
        "sequenceLength": args.sequence_length,
        "featureColumns": feature_columns,
        "categoricalMappings": mappings,
        "usedSplitHint": bool(use_split_hint),
        "rowCount": int(len(df)),
        "runCount": int(run_count),
        "outputCounts": output_counts,
        "runStats": run_stats,
    }
    (output_dir / "lstm_sequence_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    for split_name in CANONICAL_SPLITS:
        print(f"  {split_name}: {output_counts[split_name]:,} sequence row(s)")


if __name__ == "__main__":
    main()
