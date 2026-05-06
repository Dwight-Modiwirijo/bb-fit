#!/usr/bin/env python3
"""
Voeg technische indicatoren toe aan lstm_merged.csv.

Originele indicators (4):
  ind_ema55_ratio    - close / EMA(55)
  ind_ema233_ratio   - close / EMA(233)
  ind_ema_trend      - EMA(55) / EMA(233)
  ind_choppiness14   - Choppiness Index(14)

BB-fit bot indicators (8) - de signalen die de bot gebruikt:
  ind_bb20_upper_ratio  - close / BB(20) upper band
  ind_bb20_lower_ratio  - close / BB(20) lower band
  ind_bb20_bandwidth    - (upper-lower)/SMA(20)
  ind_bb20_bw_delta     - verandering in bandbreedte
  ind_sma20_delta       - SMA(20) helling genormaliseerd
  ind_sma20_delta2      - versnelling van SMA helling
  ind_rs14              - RS ratio voor RSI(14)
  ind_rs14_delta        - verandering in RS ratio

BB-optimalisatie features (3) - welke BB-parameters historisch het beste werkten:
  bb_opt_period_norm  - beste BB-periode [0..1]: 8->0.0, 13->0.25, 21->0.5, 34->0.75, 55->1.0
  bb_opt_bias_norm    - beste stddev-multiplier [0..1]: 0.7->0.0, 1.2->1.0
  bb_opt_growth       - groeiperdag van de beste combo (tanh-geschaald, ~[-1,1])

  Berekening: elke BB_DAY_BARS=1440 rijen wordt de beste (period, bias) combo bepaald
  op de afgelopen BB_LOOKBACK=720 candles (12u terugkijken, dagelijks updaten).
  Forward-filled voor de volgende dag. Zelfde logica als de bot (Optimizor.cs).

Totaal: 15 indicators, 48 features na sequentie-bouw.

Usage:
    python add_indicators.py \\
        --input-csv  ~/bb-fit/lstm_merged_tpsl_v4.csv \\
        --output-csv ~/bb-fit/lstm_merged_tpsl_v4_indicators.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# -- BB-optimalisatie constanten (zelfde als Optimizor.cs) ------------------
_BB_PERIODS  = [8, 13, 21, 34, 55]
_BB_BIASES   = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
_BB_TP       = 0.024   # take-profit 2.4%
_BB_SL       = 0.012   # stop-loss   1.2%
_BB_FEE      = 0.0006  # 0.06% per side
BB_LOOKBACK  = 720     # candles terugkijken voor parameter-selectie
BB_DAY_BARS  = 1440    # update-frequentie (1 trading dag in minuten)

_PERIOD_NORM = {8: 0.0, 13: 0.25, 21: 0.5, 34: 0.75, 55: 1.0}
_BIAS_MIN    = 0.7
_BIAS_RANGE  = 0.5     # 1.2 - 0.7


def _simulate_bb(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    """Simuleer BB-trading met TP/SL op gegeven bands. Geeft groei-per-dag terug."""
    n = len(close)
    capital = 1.0
    in_pos = False
    entry = 0.0
    n_trades = 0

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        c = close[i]
        if not in_pos:
            if c <= lower[i]:
                in_pos = True
                entry = c * (1.0 + _BB_FEE)
        else:
            tp_p = entry * (1.0 + _BB_TP)
            sl_p = entry * (1.0 - _BB_SL)
            if c >= tp_p or c >= upper[i]:
                sell = tp_p if c >= tp_p else c
                capital *= sell * (1.0 - _BB_FEE) / entry
                in_pos = False
                n_trades += 1
            elif c <= sl_p:
                capital *= sl_p * (1.0 - _BB_FEE) / entry
                in_pos = False
                n_trades += 1

    if n_trades == 0 or capital <= 0:
        return 0.0
    days = n / 1440.0
    return capital ** (1.0 / max(days, 1.0 / 1440.0)) - 1.0


def _best_bb_params(close_window: np.ndarray):
    """Zoek de beste (period, bias, growth) voor een prijsvenster."""
    s = pd.Series(close_window.astype(float))
    best_g = -np.inf
    best_p = 21
    best_b = 1.0

    for period in _BB_PERIODS:
        if len(close_window) < period + 2:
            continue
        sma = s.rolling(period).mean().to_numpy()
        std = s.rolling(period).std(ddof=0).to_numpy()
        for bias in _BB_BIASES:
            g = _simulate_bb(
                close_window.astype(float),
                sma + bias * std,
                sma - bias * std,
            )
            if g > best_g:
                best_g, best_p, best_b = g, period, bias

    return best_p, best_b, best_g


def add_bb_opt_features(run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Voeg bb_opt_period_norm, bb_opt_bias_norm, bb_opt_growth toe.
    Elke BB_DAY_BARS rijen: kijk terug op BB_LOOKBACK candles en kies beste params.
    Forward-fill voor de volgende dag. Zelfde mechanisme als Optimizor.AnalyseMaxProfit().
    """
    n = len(run_df)
    close_arr = run_df["signalClose"].astype(float).to_numpy()

    period_col = np.full(n, _PERIOD_NORM[21], dtype=float)  # standaard: period=21
    bias_col   = np.full(n, (1.0 - _BIAS_MIN) / _BIAS_RANGE, dtype=float)  # standaard: bias=1.0
    growth_col = np.zeros(n, dtype=float)

    n_computed = 0
    for day_start in range(0, n, BB_DAY_BARS):
        lb_start = max(0, day_start - BB_LOOKBACK)
        lb_end   = day_start
        if lb_end - lb_start < 30:
            continue

        window = close_arr[lb_start:lb_end]
        p, b, g = _best_bb_params(window)

        day_end = min(day_start + BB_DAY_BARS, n)
        period_col[day_start:day_end] = _PERIOD_NORM[p]
        bias_col[day_start:day_end]   = (b - _BIAS_MIN) / _BIAS_RANGE
        # tanh-schaling: groei van 0.1/dag -> tanh(1.0) ~= 0.76; negatief mogelijk
        growth_col[day_start:day_end] = float(np.tanh(g * 10.0))
        n_computed += 1

    print(f"    BB-opt: {n_computed} dagperiodes berekend ({n // BB_DAY_BARS} verwacht)", flush=True)

    run_df = run_df.copy()
    run_df["bb_opt_period_norm"] = period_col
    run_df["bb_opt_bias_norm"]   = bias_col
    run_df["bb_opt_growth"]      = growth_col
    return run_df


# ---------------------------------------------------------------------------


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

    run_df = add_bb_opt_features(run_df)

    return run_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv",  required=True)
    p.add_argument("--output-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    print("Laden input CSV ...", flush=True)
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig", low_memory=False)
    print(f"  {len(df):,} rijen, {df['runId'].nunique()} runs.", flush=True)

    new_cols = [
        "ind_ema55_ratio", "ind_ema233_ratio", "ind_ema_trend", "ind_choppiness14",
        "ind_bb20_upper_ratio", "ind_bb20_lower_ratio", "ind_bb20_bandwidth",
        "ind_bb20_bw_delta", "ind_sma20_delta", "ind_sma20_delta2",
        "ind_rs14", "ind_rs14_delta",
        "bb_opt_period_norm", "bb_opt_bias_norm", "bb_opt_growth",
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
