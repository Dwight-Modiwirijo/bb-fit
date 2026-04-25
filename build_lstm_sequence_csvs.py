#!/usr/bin/env python3
"""
Build flattened train/validation/test sequence CSVs for LSTM training from lstm_merged.csv.

The script preserves per-run ordering:
- group by runId
- sort by timestamp within each run
- split in time order within each run
- never allow windows to cross split boundaries

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
    "intervalMinutes",
    "observedIntervalMinutes",
    "signalOpen",
    "signalHigh",
    "signalLow",
    "signalClose",
    "executionOpen",
    "executionHigh",
    "executionLow",
    "executionClose",
    "executionPrice",
    "tradeActionRaw",
    "tradeSide",
    "tradingCapital",
    "assetsHeld",
    "inPosition",
    "entryPrice",
    "positionValue",
    "netEquity",
    "buyCount",
    "sellCount",
    "wins",
    "losses",
    "totalTradedNotional",
    "feePerSide",
    "cost",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "interval",
    "lastTrade",
    "actionTaken",
]

DEFAULT_TARGETS = [
    "target_actionTaken",
    "target_actionTaken_code",
    "target_tradeSide",
    "target_tradeActionRaw",
    "target_netEquityDelta",
    "target_isNetEquityUp",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to lstm_merged.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files")
    parser.add_argument("--sequence-length", type=int, default=64, help="Window length")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    return parser


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {c.lower(): c for c in df.columns}
    wanted = {
        "runid": "runId",
        "timestamp": "timestamp",
        "tradeactionraw": "tradeActionRaw",
        "tradeside": "tradeSide",
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


def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        raise ValueError(f"{bad_ts} row(s) have invalid timestamp values")
    return df.sort_values(["runId", "timestamp"]).reset_index(drop=True)


def add_categorical_codes(df: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    df = df.copy()
    mappings: Dict[str, Dict[str, int]] = {}
    for col in categorical_features:
        if col not in df.columns:
            continue
        series = df[col].fillna("__MISSING__").astype(str)
        unique_values = sorted(series.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
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
    df[feature_columns] = df[feature_columns].fillna(0.0)
    return df, feature_columns, mappings


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


def build_rows_for_segment(
    run_df: pd.DataFrame,
    split_name: str,
    start_idx: int,
    end_idx: int,
    sequence_length: int,
    feature_columns: List[str],
) -> List[List[object]]:
    rows: List[List[object]] = []
    segment = run_df.iloc[start_idx:end_idx].reset_index(drop=True)
    if len(segment) < sequence_length + 1:
        return rows

    for end_pos in range(sequence_length - 1, len(segment) - 1):
        window = segment.iloc[end_pos - sequence_length + 1 : end_pos + 1]
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

        window_features = window.reindex(columns=feature_columns, fill_value=0.0)
        out_row.extend(window_features.to_numpy().reshape(-1).tolist())

        action_taken = target_row["actionTaken"] if "actionTaken" in target_row.index else ""
        action_taken_code = target_row["actionTaken_code"] if "actionTaken_code" in target_row.index else ""
        trade_side = target_row["tradeSide"] if "tradeSide" in target_row.index else 0
        trade_action_raw = target_row["tradeActionRaw"] if "tradeActionRaw" in target_row.index else 0.0

        out_row.extend(
            [
                action_taken,
                action_taken_code,
                trade_side,
                trade_action_raw,
                delta,
                int(delta > 0.0),
            ]
        )
        rows.append(out_row)
    return rows


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = normalize_columns(df)
    ensure_required_columns(df)
    df = parse_and_sort(df)
    df, feature_columns, mappings = prepare_features(df, DEFAULT_NUMERIC_FEATURES, DEFAULT_CATEGORICAL_FEATURES)

    missing_feature_cols = [c for c in feature_columns if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"Feature columns missing from dataframe: {missing_feature_cols}")

    header = build_header(args.sequence_length, feature_columns)

    split_rows = {
        "train": [],
        "validation": [],
        "test": [],
    }
    run_stats = []

    for run_id, run_df in df.groupby("runId", sort=False):
        run_df = run_df.sort_values("timestamp").reset_index(drop=True)
        bounds = segment_bounds(
            len(run_df),
            args.train_ratio,
            args.validation_ratio,
            args.test_ratio,
        )

        stats_entry = {"runId": str(run_id), "rows": int(len(run_df)), "windows": {}}
        for split_name, (start_idx, end_idx) in bounds.items():
            rows = build_rows_for_segment(
                run_df=run_df,
                split_name=split_name,
                start_idx=start_idx,
                end_idx=end_idx,
                sequence_length=args.sequence_length,
                feature_columns=feature_columns,
            )
            split_rows[split_name].extend(rows)
            stats_entry["windows"][split_name] = len(rows)
        run_stats.append(stats_entry)

    write_csv(output_dir / "lstm_train_sequences.csv", header, split_rows["train"])
    write_csv(output_dir / "lstm_validation_sequences.csv", header, split_rows["validation"])
    write_csv(output_dir / "lstm_test_sequences.csv", header, split_rows["test"])

    summary = {
        "input": str(input_path),
        "sequenceLength": args.sequence_length,
        "featureColumns": feature_columns,
        "categoricalMappings": mappings,
        "splitCounts": {k: len(v) for k, v in split_rows.items()},
        "runStats": run_stats,
    }
    with (output_dir / "lstm_sequence_build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
