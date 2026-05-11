# Prompt: Build lstm_merged.csv from Crypteum Bot JSONL Logs

## Context

A C# trading bot (Crypteum) logs one JSON object per candle to `.jsonl` files
during backtesting via `MaxProfitTest`. These files are the source of truth for
LSTM training data.

Each JSON row contains everything the bot knew at decision time (no lookahead):
- OHLC of the signal bar and execution bar
- Technical indicators computed by TAengine: `sma`, `ema`, `upper_band`,
  `lower_band`, `deviation`, `band_width`, `band_width_delta`, `rsi`, `stoch_rsi`
- BB position: `bb_position` = (close - lower_band) / (upper_band - lower_band)
- Position context: `inPosition`, `unrealized_pnl`, `bars_since_entry`,
  `bars_since_last_trade`, `long_entry_price`, `short_entry_price`
- Bot decision: `actionTaken` (1=buy/long-entry, -1=sell/long-exit, 0=hold)
- Trade side: `tradeSide`, `lastTrade`
- Account state: `tradingCapital`, `assetsHeld`, `shortAssetsHeld`, `netEquity`
- P&L counters: `wins`, `losses`, `long_profit_loss`, `short_profit_loss`
- Trade counts: `longEntryCount`, `longExitCount`, `shortEntryCount`, `shortExitCount`
- Fees: `feePerSide`, `cost`

## JSONL File Format

One JSON object per line. Filename encodes metadata:

```
XBTUSD_FiveMinutes_LongOnly_0.0018_20260511120408_<uuid>.jsonl
symbol_interval_mode_fee_datetime_uuid.jsonl
```

Example row (abbreviated):
```json
{"runId":"XBTUSD_FiveMinutes_LongOnly_0.0018_20260511...","timestamp":1776078360.0,
 "interval":"FiveMinutes","intervalMinutes":5,"signalClose":70855.0,
 "sma":71200.0,"ema":71150.0,"upper_band":72000.0,"lower_band":70400.0,
 "rsi":42.3,"stoch_rsi":0.18,"bb_position":0.29,
 "actionTaken":1,"tradeSide":1,"inPosition":1,...}
```

## What lstm_merged.csv Is

`lstm_merged.csv` is the merged training dataset. It is built from one or more
`.jsonl` files and adds these metadata columns:

| Column | Source | Description |
|---|---|---|
| `runGroup` | derived from filename | e.g. "FiveMinutes_5m_fee_0.0018" |
| `canonicalFee` | derived from filename | e.g. "0.0018" |
| `sourceFile` | filename | source .jsonl filename |
| `rowIndexWithinRun` | computed | sequential index per runId |
| `splitHint` | computed | "train" / "validation" / "test" (70/15/15 chronological) |

All 60 columns of the final CSV:
- 6 metadata columns (above)
- 54 columns directly from the JSON rows

## Script: parse_testlog_to_csv.py

Located at `/home/dwyte/Github/bb-fit/parse_testlog_to_csv.py`.

**Usage:**
```bash
python3 parse_testlog_to_csv.py \
  /path/to/logs/*.jsonl \
  --output lstm_merged.csv
```

**What it does:**
1. Reads each `.jsonl` file line by line
2. Parses each JSON object, keeps rows with `runId` + `timestamp`
3. Sorts chronologically per `runId`
4. Adds metadata columns (`runGroup`, `canonicalFee`, `sourceFile`,
   `rowIndexWithinRun`, `splitHint`)
5. Concatenates all runs and saves as CSV

## What Is NOT Needed Anymore

The new `.jsonl` format makes these old pipeline steps **obsolete**:

- `rebuild_dataset_tpsl.py` — used to recompute TP/SL exits; no longer needed
  because future return labels replace this (see below)
- `add_indicators.py` — used to recompute BB/RSI from OHLC; no longer needed
  because the bot now logs these directly

## Next Step: Future Return Labels

After building `lstm_merged.csv`, run `add_future_labels.py` to add supervised
learning targets:

```bash
python3 add_future_labels.py \
  --input  lstm_merged.csv \
  --output lstm_merged_labeled.csv \
  --horizons 3 6 12 24 \
  --label-horizon 12 \
  --positive-threshold 0.003 \
  --negative-threshold 0.003
```

This adds:
- `future_return_3/6/12/24` — % price change after N candles
- `future_direction_12` — 1 / 0 / -1
- `future_return_label_12` — "long" / "hold" / "short"
- `bot_signal_quality_12` — was the bot's action correct in hindsight?

**WARNING:** `future_return_*` columns are labels only. Never use as model input.

## Training Objective Options

Two approaches for the LSTM:

### Option A: Imitation Learning
Train model to predict what the bot did:
```
X = features (OHLC, BB, RSI, position context)
y = actionTaken (1=long, 0=hold, -1=short)
```
Strength: directly replicates bot behavior.
Weakness: if bot behavior is suboptimal, model inherits that.

### Option B: Outcome Learning
Train model to predict what was profitable:
```
X = features (OHLC, BB, RSI, position context)
y = future_return_label_12 ("long"/"hold"/"short")
```
Strength: model learns when entries are actually profitable.
Weakness: requires enough data for labels to be meaningful.

## Full Pipeline

```
MaxProfitTest (C#, Windows)
  └─ writes *.jsonl to ~/bb-fit/Trader/logs/

parse_testlog_to_csv.py *.jsonl
  └─ lstm_merged.csv  (60 columns, all features + actionTaken)

add_future_labels.py
  └─ lstm_merged_labeled.csv  (+ future_return_* labels)

build_lstm_sequence_csvs_streaming.py
  └─ sequences/lstm_train_sequences.csv
  └─ sequences/lstm_validation_sequences.csv
  └─ sequences/lstm_test_sequences.csv

train_lstm_bbfit.py
  └─ checkpoint_epoch*.pt
```

## Important Notes

- Bot is **Long-only** in current setup (`allowShorts=False` in MaxProfitTest)
- `actionTaken=-1` means "close Long position", NOT "open Short"
- Early rows have `sma=0.0`, `rsi=0.0` etc. — TAengine needs warmup period
  (typically first 30-50 candles per run). Filter these out before training.
- Sort always by `timestamp` within `runId` before computing sequences
- Never mix train/val/test splits randomly — always chronological
