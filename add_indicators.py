#!/usr/bin/env python3
"""
Voeg technische indicatoren toe aan lstm_merged.csv.

Alleen features die informatie toevoegen buiten het 64-stap LSTM-venster:

  ind_ema55_ratio    — close / EMA(55):  positie t.o.v. middellange trend  [~1.0]
  ind_ema233_ratio   — close / EMA(233): positie t.o.v. langetermijntrend  [~1.0]
  ind_ema_trend      — EMA(55) / EMA(233): golden/death cross positie      [~1.0]
  ind_choppiness14   — Choppiness Index(14): trending=laag, zijwaarts=hoog [0..1]

EMA(55) en EMA(233) vereisen 55 resp. 233 bars history — buiten het 64-stap venster.
Choppiness comprimeert trend/range-context die het model niet vanzelf leert.

NaN-afhandeling (begin van run, te weinig history):
  EMA-ratios   → 1.0  (neutraal: prijs op de EMA)
  ind_ema_trend → 1.0  (neutraal)
  Choppiness   → 0.618 (neutraal/licht zijwaarts)

Usage:
    python add_indicators.py \\
        --input-csv  ~/bb-fit/lstm_merged.csv \\
        --output-csv ~/bb-fit/lstm_merged_indicators.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


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

    new_cols = ["ind_ema55_ratio", "ind_ema233_ratio", "ind_ema_trend", "ind_choppiness14"]

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
