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
    pnl_filter: bool = False,
) -> Tuple[List[Dict], int, int, int]:
    """
    Apply TP/SL logic to a single run.

    PnL filtering: if a trade exits via SL (loss), the entry row is relabeled
    as hold (actionTaken=0, tradeSide=0, tradeActionRaw=0) so the LSTM does not
    learn to open losing positions.

    Returns (modified_rows, n_early_tp, n_early_sl, n_total_exits).
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
    entry_result_idx = -1  # index in result[] where entry row was appended

    # Early-exit state
    early_exit_pending = False
    early_exit_price = 0.0
    early_exit_reason = ""
    exited_early = False
    post_exit_net_equity = 0.0

    # PnL filter: indices of entry rows that resulted in SL (losing) trades
    sl_entry_indices: List[int] = []

    n_early_tp = 0
    n_early_sl = 0
    n_total = 0

    for row in rows:
        row = dict(row)
        row["exit_reason"] = ""
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
            row["exit_reason"]    = early_exit_reason  # "tp" or "sl"
            result.append(row)

            # PnL filter: SL exit → mark the entry row for relabeling
            if early_exit_reason == "sl" and entry_result_idx >= 0:
                sl_entry_indices.append(entry_result_idx)
                n_early_sl += 1
            else:
                n_early_tp += 1

            in_position = False
            early_exit_pending = False
            exited_early = True
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
                # Original exit — convert to hold (position already closed)
                row["actionTaken"]    = "0"
                row["tradeSide"]      = "0"
                row["tradeActionRaw"] = "0"
                row["exit_reason"]    = "orig_suppressed"
                exited_early = False
            result.append(row)
            continue

        # ------------------------------------------------------------------
        # Normal processing
        # ------------------------------------------------------------------
        if action == 1:
            in_position = True
            entry_result_idx  = len(result)   # record where this entry lands
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
            row["exit_reason"] = "orig"
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

    # ------------------------------------------------------------------
    # PnL filter post-pass: relabel SL entry rows as hold (only if enabled)
    # ------------------------------------------------------------------
    if pnl_filter:
        for idx in sl_entry_indices:
            result[idx]["actionTaken"]    = "0"
            result[idx]["tradeSide"]      = "0"
            result[idx]["tradeActionRaw"] = "0"
            result[idx]["exit_reason"]    = "sl_entry_relabeled"

    return result, n_early_tp, n_early_sl, n_total


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
    p.add_argument("--pnl-filter", action="store_true", default=False,
                   help="Relabel SL-exit entry rows as hold (PnL filtering). Default: off.")
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
    grand_total = 0

    # Add exit_reason to fieldnames if not present
    if "exit_reason" not in fieldnames:
        fieldnames = fieldnames + ["exit_reason"]

    grand_tp = 0
    grand_sl = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_idx, (run_id, rows) in enumerate(run_rows.items()):
            print(f"  Run {run_idx+1}/{len(run_rows)}: {run_id[:60]} ({len(rows):,} rows)", flush=True)
            modified, n_tp, n_sl, n_total = process_run(rows, onemin, args.tp_pct, args.sl_pct, args.pnl_filter)
            grand_tp += n_tp
            grand_sl += n_sl
            grand_total += n_total
            for row in modified:
                writer.writerow(row)
            n_orig = n_total - n_tp - n_sl
            print(f"    exits: {n_total:,} total | TP={n_tp:,} | SL={n_sl:,} (relabeled→hold) | orig={n_orig:,}", flush=True)

    n_orig_total = grand_total - grand_tp - grand_sl
    print(f"\nDone.", flush=True)
    print(f"Total exits : {grand_total:,}", flush=True)
    print(f"  TP exits  : {grand_tp:,}  ({100*grand_tp/max(grand_total,1):.1f}%)", flush=True)
    print(f"  SL exits  : {grand_sl:,}  ({100*grand_sl/max(grand_total,1):.1f}%) → entry relabeled as hold", flush=True)
    print(f"  Orig exits: {n_orig_total:,}  ({100*n_orig_total/max(grand_total,1):.1f}%)", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
