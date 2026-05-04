#!/usr/bin/env python3
"""
Voeg technische indicatoren toe aan lstm_merged.csv.

Originele indicators (4):
  ind_ema55_ratio    — close / EMA(55)
  ind_ema233_ratio   — close / EMA(233)
  ind_ema_trend      — EMA(55) / EMA(233)
  ind_choppiness14   — Choppiness Index(14)

BB-fit bot indicators (8) — dit zijn de signalen die de bot gebruikt:
  ind_bb20_upper_ratio  — close / BB(20) upper band  [>1 = boven upper, koopsignaal reversed]
  ind_bb20_lower_ratio  — close / BB(20) lower band  [<1 = onder lower, koopsignaal]
  ind_bb20_bandwidth    — (upper-lower)/SMA(20): genormaliseerde bandbreedte
  ind_bb20_bw_delta     — verandering in bandbreedte (bwDelta in bot: keert om bij reversal)
  ind_sma20_delta       — SMA(20) helling genormaliseerd door close (dSma in bot)
  ind_sma20_delta2      — versnelling van SMA helling (dSma - dSma_1 in bot)
  ind_rs14              — RS ratio voor RSI(14): avgGain/avgLoss
  ind_rs14_delta        — verandering in RS ratio (drs in bot: keert om bij momentum shift)

Totaal: 12 indicators, 45 features na sequentie-bouw.

NaN-afhandeling:
  BB-ratios     → 1.0  (neutraal: prijs op de band)
  bandwidth     → 0.0
  deltas        → 0.0
  rs            → 1.0  (neutraal: gelijke winsten en verliezen)

Usage:
    python add_indicators.py \\
        --input-csv  ~/bb-fit/lstm_merged_tpsl_v2.csv \\
        --output-csv ~/bb-fit/lstm_merged_tpsl_v2_indicators.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def compute_bollinger_features(close: pd.Series, period: int = 20):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = sma + 2 * std
    lower = sma - 2 * std

    upper_ratio = (close / upper.replace(0, np.nan)).fillna(1.0)
    lower_ratio = (close / lower.replace(0, np.nan)).fillna(1.0)
    bandwidth = ((upper - lower) / sma.replace(0, np.nan)).fillna(0.0)
    bw_delta = bandwidth.diff().fillna(0.0)
    sma_delta = (sma.diff() / close.replace(0, np.nan)).fillna(0.0)
    sma_delta2 = sma_delta.diff().fillna(0.0)
    return upper_ratio, lower_ratio, bandwidth, bw_delta, sma_delta, sma_delta2


def compute_rs(close: pd.Series, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = (avg_gain / avg_loss.replace(0, np.nan)).fillna(1.0)
    rs_delta = rs.diff().fillna(0.0)
    return rs, rs_delta


def compute_choppiness(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    sum_tr   = tr.rolling(period).sum()
    highest  = high.rolling(period).max()
    lowest   = low.rolling(period).min()
    range_hl = (highest - lowest).replace(0, np.nan)

    chop = np.log10(sum_tr / range_hl) / np.log10(period)
    return chop.fillna(0.618).clip(0.0, 1.0)


def add_indicators_to_run(run_df: pd.DataFrame) -> pd.DataFrame:
    run_df = run_df.copy()

    close = run_df["signalClose"].astype(float)
    high  = run_df["signalHigh"].astype(float)
    low   = run_df["signalLow"].astype(float)

    ema55  = compute_ema(close, 55)
    ema233 = compute_ema(close, 233)

    run_df["ind_ema55_ratio"]  = (close / ema55.replace(0, np.nan)).fillna(1.0)
    run_df["ind_ema233_ratio"] = (close / ema233.replace(0, np.nan)).fillna(1.0)
    run_df["ind_ema_trend"]    = (ema55 / ema233.replace(0, np.nan)).fillna(1.0)
    run_df["ind_choppiness14"] = compute_choppiness(high, low, close, period=14)

    upper_ratio, lower_ratio, bandwidth, bw_delta, sma_delta, sma_delta2 = compute_bollinger_features(close, period=20)
    rs, rs_delta = compute_rs(close, period=14)

    run_df["ind_bb20_upper_ratio"] = upper_ratio
    run_df["ind_bb20_lower_ratio"] = lower_ratio
    run_df["ind_bb20_bandwidth"]   = bandwidth
    run_df["ind_bb20_bw_delta"]    = bw_delta
    run_df["ind_sma20_delta"]      = sma_delta
    run_df["ind_sma20_delta2"]     = sma_delta2
    run_df["ind_rs14"]             = rs
    run_df["ind_rs14_delta"]       = rs_delta

    return run_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv",  required=True)
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    vol_lookup = None

    print("Laden input CSV ...", flush=True)
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    print(f"  {len(df):,} rijen, {df['runId'].nunique()} runs.", flush=True)

    new_cols = [
        "ind_ema55_ratio", "ind_ema233_ratio", "ind_ema_trend", "ind_choppiness14",
        "ind_bb20_upper_ratio", "ind_bb20_lower_ratio", "ind_bb20_bandwidth",
        "ind_bb20_bw_delta", "ind_sma20_delta", "ind_sma20_delta2",
        "ind_rs14", "ind_rs14_delta",
    ]

    parts = []
    for run_id, group in df.groupby("runId", sort=False):
        print(f"  Verwerken run: {run_id[:60]} ({len(group):,} rijen)", flush=True)
        parts.append(add_indicators_to_run(group))

    result = pd.concat(parts, ignore_index=True)

    print(f"\nNieuwe kolommen: {new_cols}", flush=True)
    print(result[new_cols].describe().round(4).to_string(), flush=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"\nOpgeslagen: {out}  ({len(result):,} rijen, {len(result.columns)} kolommen)", flush=True)


if __name__ == "__main__":
    main()
