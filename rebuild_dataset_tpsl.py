#!/usr/bin/env python3
"""
Rebuild lstm_merged.csv with TP/SL exit rules.

For each long position:
  - TP triggered when 1-min High  >= entry_price * (1 + tp_pct)
  - SL triggered when 1-min Low   <= entry_price * (1 - sl_pct)
  - Exit = earliest of: TP, SL, or original exit signal (-1)
  - If both TP and SL trigger in the same 1-min candle: SL wins (conservative)

Position features (inPosition, assetsHeld, entryPrice, positionValue, netEquity)
are updated consistently for all rows affected by an early exit.

Usage:
    python rebuild_dataset_tpsl.py \\
        --input-csv  ~/bb-fit/lstm_merged.csv \\
        --onemin-csv ~/bb-fit/btcusd_1-min_data.csv \\
        --output-csv ~/bb-fit/lstm_merged_tpsl.csv \\
        --tp-pct 0.024 \\
        --sl-pct 0.012
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1-min data loader
# ---------------------------------------------------------------------------

def load_onemin(path: Path) -> Dict[int, Tuple[float, float]]:
    """Return {unix_ts_int: (high, low)} for every 1-min candle."""
    print("Loading 1-min candle data ...", flush=True)
    data: Dict[int, Tuple[float, float]] = {}
    with path.open(encoding="utf-8") as f:
        next(f)  # skip header: Timestamp,Open,High,Low,Close,Volume
        for line in f:
            parts = line.split(",")
            ts = int(float(parts[0]))
            high = float(parts[2])
            low = float(parts[3])
            data[ts] = (high, low)
    print(f"  {len(data):,} candles loaded.", flush=True)
    return data


# ---------------------------------------------------------------------------
# TP/SL scanner
# ---------------------------------------------------------------------------

def find_first_tpsl(
    start_ts: int,
    end_ts: int,
    tp_price: float,
    sl_price: float,
    onemin: Dict[int, Tuple[float, float]],
) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """
    Scan 1-min candles from start_ts to end_ts (inclusive).
    Returns (hit_ts, exit_price, reason) or (None, None, None).
    SL takes priority over TP within the same candle.
    """
    ts = start_ts
    while ts <= end_ts:
        candle = onemin.get(ts)
        if candle is not None:
            high, low = candle
            if low <= sl_price:
                return ts, sl_price, "sl"
            if high >= tp_price:
                return ts, tp_price, "tp"
        ts += 60
    return None, None, None


# ---------------------------------------------------------------------------
# Per-run processor
# ---------------------------------------------------------------------------

def process_run(
    rows: List[Dict],
    onemin: Dict[int, Tuple[float, float]],
    tp_pct: float,
    sl_pct: float,
) -> Tuple[List[Dict], int, int]:
    """
    Apply TP/SL logic to a single run.
    Returns (modified_rows, n_early_exits, n_total_exits).
    """
    result: List[Dict] = []

    # Position state
    in_position = False
    entry_price = 0.0
    entry_net_equity = 0.0
    entry_assets = 0.0
    tp_price = 0.0
    sl_price = 0.0
    fee = 0.0
    last_scanned_ts = 0

    # Early-exit state
    early_exit_pending = False   # TP/SL found, not yet applied to a row
    early_exit_price = 0.0
    early_exit_reason = ""
    exited_early = False         # applied early exit; nulling rows until orig exit
    post_exit_net_equity = 0.0

    n_early = 0
    n_total = 0

    for row in rows:
        row = dict(row)
        action = int(float(row["actionTaken"]))
        row_ts = int(float(row["executionTimestamp"]))
        fee = float(row["canonicalFee"])

        # ------------------------------------------------------------------
        # While in position: scan 1-min data up to this row's timestamp
        # ------------------------------------------------------------------
        if in_position and not early_exit_pending and not exited_early:
            scan_start = last_scanned_ts + 60
            if scan_start <= row_ts:
                hit_ts, hit_price, hit_reason = find_first_tpsl(
                    scan_start, row_ts, tp_price, sl_price, onemin
                )
                if hit_ts is not None:
                    early_exit_pending = True
                    early_exit_price = hit_price
                    early_exit_reason = hit_reason
            last_scanned_ts = row_ts

        # ------------------------------------------------------------------
        # Apply pending early exit at this row
        # ------------------------------------------------------------------
        if in_position and early_exit_pending:
            exit_net_equity = (
                entry_net_equity * (early_exit_price / entry_price) * (1.0 - fee)
                if entry_price > 0 else entry_net_equity
            )
            post_exit_net_equity = exit_net_equity

            row["actionTaken"]    = "-1"
            row["tradeSide"]      = "-1"
            row["tradeActionRaw"] = str(early_exit_price)
            row["lastTrade"]      = str(early_exit_price)
            row["executionPrice"] = str(early_exit_price)
            row["inPosition"]     = "0"
            row["assetsHeld"]     = "0"
            row["entryPrice"]     = "0"
            row["positionValue"]  = "0"
            row["netEquity"]      = str(exit_net_equity)
            result.append(row)

            in_position = False
            early_exit_pending = False
            exited_early = True
            n_early += 1
            n_total += 1
            continue

        # ------------------------------------------------------------------
        # Null out rows between early exit and original exit
        # ------------------------------------------------------------------
        if exited_early:
            row["inPosition"]     = "0"
            row["assetsHeld"]     = "0"
            row["entryPrice"]     = "0"
            row["positionValue"]  = "0"
            row["netEquity"]      = str(post_exit_net_equity)
            if action == -1:
                # This is the original exit — convert to hold
                row["actionTaken"]    = "0"
                row["tradeSide"]      = "0"
                row["tradeActionRaw"] = "0"
                exited_early = False
            result.append(row)
            continue

        # ------------------------------------------------------------------
        # Normal processing
        # ------------------------------------------------------------------
        if action == 1:
            in_position = True
            entry_price       = float(row["executionPrice"])
            entry_net_equity  = float(row["netEquity"])
            entry_assets      = float(row["assetsHeld"])
            tp_price          = entry_price * (1.0 + tp_pct)
            sl_price          = entry_price * (1.0 - sl_pct)
            last_scanned_ts   = row_ts
            early_exit_pending = False
            exited_early       = False
            result.append(row)

        elif action == -1:
            in_position = False
            early_exit_pending = False
            exited_early = False
            n_total += 1
            result.append(row)

        else:
            # Hold row — keep in-position features consistent
            if in_position:
                current_price = float(row.get("signalClose") or 0) or entry_price
                if entry_price > 0 and current_price > 0:
                    row["positionValue"] = str(entry_assets * current_price)
                    row["netEquity"]     = str(entry_net_equity * current_price / entry_price)
                row["inPosition"]  = "1"
                row["assetsHeld"]  = str(entry_assets)
                row["entryPrice"]  = str(entry_price)
            result.append(row)

    return result, n_early, n_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild lstm_merged.csv with TP/SL exits.")
    p.add_argument("--input-csv",  required=True)
    p.add_argument("--onemin-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--tp-pct", type=float, default=0.024, help="Take-profit fraction (default 0.024 = 2.4%%)")
    p.add_argument("--sl-pct", type=float, default=0.012, help="Stop-loss fraction (default 0.012 = 1.2%%)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onemin = load_onemin(Path(args.onemin_csv))

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    # Read all rows, group by runId (preserving insertion order)
    print("Reading input CSV ...", flush=True)
    run_rows: Dict[str, List[Dict]] = defaultdict(list)
    fieldnames = None
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            run_rows[row["runId"]].append(row)

    total_rows = sum(len(v) for v in run_rows.values())
    print(f"  {total_rows:,} rows across {len(run_rows):,} run(s).", flush=True)
    print(f"TP={args.tp_pct*100:.1f}%  SL={args.sl_pct*100:.1f}%", flush=True)

    # Process each run and write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grand_early = 0
    grand_total = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_idx, (run_id, rows) in enumerate(run_rows.items()):
            print(f"  Run {run_idx+1}/{len(run_rows)}: {run_id[:60]} ({len(rows):,} rows)", flush=True)
            modified, n_early, n_total = process_run(rows, onemin, args.tp_pct, args.sl_pct)
            grand_early += n_early
            grand_total += n_total
            for row in modified:
                writer.writerow(row)
            print(f"    exits: {n_total:,} total, {n_early:,} early (TP/SL), {n_total-n_early:,} original", flush=True)

    print(f"\nDone.", flush=True)
    print(f"Total exits : {grand_total:,}", flush=True)
    print(f"Early (TP/SL): {grand_early:,} ({100*grand_early/grand_total:.1f}% of exits)", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
