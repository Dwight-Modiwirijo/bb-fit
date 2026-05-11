"""
parse_testlog_to_csv.py — convert .jsonl training logs to lstm_merged.csv.

Flow:
  MaxProfitTest (C#) → one JSON object per line → .jsonl file
  → this script → lstm_merged.csv

Supports both:
  - .jsonl files (new format): one JSON object per line
  - .testlog files (legacy): JSON embedded in [INFO] lines

New columns logged by C# v2 (vs old version):
  sma, ema, upper_band, lower_band, deviation, band_width, band_width_delta,
  rsi, stoch_rsi, shortAssetsHeld, long_entry_price, short_entry_price,
  unrealized_pnl, bars_since_entry, bars_since_last_trade, bb_position,
  longEntryCount, longExitCount, shortEntryCount, shortExitCount,
  long_profit_loss, short_profit_loss

Added by this script:
  runGroup, canonicalFee, sourceFile, rowIndexWithinRun, splitHint
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Regex to extract JSON object from a log line.
_JSON_RE = re.compile(r'\{.*\}')

# Train/val/test split ratios (chronological per runId).
SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15
# test = remainder


def extract_run_metadata(run_id: str, source_file: str) -> dict:
    """
    Derive runGroup, canonicalFee from runId.

    Expected runId pattern examples:
      XBTUSD_OneMinute_LongOnly_0.0018_20260511_abc123
      XBTUSD_FiveMinutes_0.0018_20260417_xyz
    """
    parts = run_id.split("_")
    fee = "unknown"
    interval_label = parts[1] if len(parts) > 1 else "Unknown"

    for p in parts:
        try:
            val = float(p)
            if 0 < val < 1:
                fee = p
                break
        except ValueError:
            pass

    interval_minutes = {
        "OneMinute": 1, "FiveMinutes": 5, "FifteenMinutes": 15,
        "ThirtyMinutes": 30, "OneHour": 60,
    }.get(interval_label, 0)

    run_group = f"{interval_label}_{interval_minutes}m_fee_{fee}"
    return {"runGroup": run_group, "canonicalFee": fee, "sourceFile": source_file}


def assign_split(n: int) -> list[str]:
    """Return a list of splitHint values ('train'/'validation'/'test') for n rows."""
    train_end = int(n * SPLIT_TRAIN)
    val_end   = int(n * (SPLIT_TRAIN + SPLIT_VAL))
    hints = ["train"] * train_end
    hints += ["validation"] * (val_end - train_end)
    hints += ["test"] * (n - val_end)
    return hints


def parse_jsonl(path: Path) -> list[dict]:
    """Parse .jsonl file — one JSON object per line."""
    rows = []
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "runId" in obj and "timestamp" in obj:
                    rows.append(obj)
            except json.JSONDecodeError:
                pass
    return rows


def parse_testlog(path: Path) -> list[dict]:
    """Parse legacy .testlog — JSON embedded in [INFO] lines."""
    rows = []
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            m = _JSON_RE.search(line)
            if m:
                try:
                    obj = json.loads(m.group())
                    if "runId" in obj and "timestamp" in obj:
                        rows.append(obj)
                except json.JSONDecodeError:
                    pass
    return rows


def parse_file(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return parse_jsonl(path)
    return parse_testlog(path)


def build_dataframe(all_rows: list[dict], source_file: str) -> pd.DataFrame:
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Sort chronologically within each run.
    df = df.sort_values(["runId", "timestamp"]).reset_index(drop=True)

    # Drop duplicate (runId, timestamp) pairs — can occur when append-mode reruns
    # write the same candle range into the same .jsonl file.
    before = len(df)
    df = df.drop_duplicates(subset=["runId", "timestamp"], keep="first").reset_index(drop=True)
    if len(df) < before:
        print(f"  [dedup] Removed {before - len(df):,} duplicate (runId, timestamp) rows")

    # Add metadata columns.
    meta_cache: dict[str, dict] = {}
    run_groups, fees, sources = [], [], []
    for rid in df["runId"]:
        if rid not in meta_cache:
            meta_cache[rid] = extract_run_metadata(rid, source_file)
        m = meta_cache[rid]
        run_groups.append(m["runGroup"])
        fees.append(m["canonicalFee"])
        sources.append(m["sourceFile"])

    df.insert(1, "runGroup",      run_groups)
    df.insert(2, "canonicalFee",  fees)
    df.insert(3, "sourceFile",    sources)

    # rowIndexWithinRun and splitHint per runId.
    row_indices = []
    split_hints = []
    for _, group in df.groupby("runId", sort=False):
        n = len(group)
        row_indices.extend(range(n))
        split_hints.extend(assign_split(n))

    df.insert(4, "rowIndexWithinRun", row_indices)
    df.insert(5, "splitHint",         split_hints)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Parse NUnit testlog(s) → lstm_merged.csv"
    )
    parser.add_argument(
        "testlogs", nargs="+",
        help="One or more .testlog files produced by MaxProfitTest"
    )
    parser.add_argument(
        "--output", default="lstm_merged.csv",
        help="Output CSV path (default: lstm_merged.csv)"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing output CSV instead of overwriting"
    )
    args = parser.parse_args()

    frames = []
    for log_path_str in args.testlogs:
        log_path = Path(log_path_str)
        if not log_path.exists():
            print(f"[WARN] Not found: {log_path}", file=sys.stderr)
            continue
        print(f"Parsing {log_path} ...")
        rows = parse_file(log_path)
        print(f"  Found {len(rows):,} training rows")
        if rows:
            frames.append(build_dataframe(rows, log_path.name))

    if not frames:
        print("No training rows found in any testlog.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    split_counts = df["splitHint"].value_counts().to_dict()
    print(f"Split: {split_counts}")

    new_cols = [c for c in df.columns if c not in {
        "runId","runGroup","canonicalFee","sourceFile","rowIndexWithinRun","splitHint",
        "timestamp","interval","intervalMinutes","observedIntervalMinutes",
        "signalTimestamp","executionTimestamp",
        "signalOpen","signalHigh","signalLow","signalClose",
        "executionOpen","executionHigh","executionLow","executionClose",
        "executionPrice","tradeActionRaw","tradeSide","lastTrade","actionTaken",
        "tradingCapital","assetsHeld","inPosition","entryPrice","positionValue","netEquity",
        "buyCount","sellCount","wins","losses","totalTradedNotional","feePerSide","cost",
    }]
    if new_cols:
        print(f"\nNew columns vs old format: {new_cols}")

    output = Path(args.output)
    if args.append and output.exists():
        existing = pd.read_csv(output)
        df = pd.concat([existing, df], ignore_index=True)
        print(f"Appended. Total rows now: {len(df):,}")

    df.to_csv(output, index=False)
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()
