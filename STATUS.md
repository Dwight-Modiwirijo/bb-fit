# BB-Fit LSTM — Project Status

## Doel
Een LSTM trainen die leert wanneer de BB-fit strategie een positie moet openen (short/hold/long) op basis van marktdata + technische indicators.

---

## Pipeline (volledige volgorde bij herstart)

```
1. rebuild_dataset_tpsl.py   → lstm_merged_tpsl.csv      (optioneel, TP/SL exits)
2. add_indicators.py          → lstm_merged_indicators_v3.csv
3. build_lstm_sequence_csvs_streaming.py → sequences_indicators_v3/
4. remap_labels_fast.py       → labels -1/0/1 → 0/1/2 (train + val + test)
5. build_balanced_warmup_csv.py → lstm_train_balanced_warmup.csv (1:2:1 ratio)
6. train_lstm_bbfit.py        → checkpoints/
```

Scripts om dit uit te voeren:
- `run_build_indicators_v3.sh` — stap 2 t/m 5
- `run_indicators_warmup_01.sh` — stap 6

---

## Data

| Bestand | Locatie | Grootte |
|---|---|---|
| `lstm_merged.csv` | `/home/dwyte/logs/` | 1.2 GB |
| `btcusd_1-min_data.csv` | `/home/dwyte/logs/` | 377 MB |
| `lstm_merged_indicators_v3.csv` | `/home/dwyte/bb-fit/` | 1.3 GB |
| `sequences_indicators_v3/` | `/home/dwyte/bb-fit/` | ~22 GB |
| Checkpoints | `/home/dwyte/checkpoints/lstm_bbfit/` | ~50-100 MB/stuk |

**Backup op Google Drive:** `lstm_merged.csv` + `btcusd_1-min_data.csv` + checkpoints na elke epoch.

---

## Features (37 totaal)

**Origineel (33):** canonicalFee, intervalMinutes, observedIntervalMinutes, signalOHLC (4), executionOHLC (4), executionPrice, tradeActionRaw, tradeSide, lastTrade, actionTaken, tradingCapital, assetsHeld, inPosition, entryPrice, positionValue, netEquity, buyCount, sellCount, wins, losses, totalTradedNotional, feePerSide, cost, runGroup_code, sourceFile_code, interval_code, splitHint_code

**Nieuw toegevoegd (4):**
- `ind_ema55_ratio` — close / EMA(55), middellange trend
- `ind_ema233_ratio` — close / EMA(233), langetermijntrend
- `ind_ema_trend` — EMA(55) / EMA(233), golden/death cross
- `ind_choppiness14` — Choppiness Index(14), trending vs zijwaarts

---

## Labels

| Waarde in CSV | Betekenis |
|---|---|
| 0 | Short (was -1 in brondata) |
| 1 | Hold (was 0 in brondata) |
| 2 | Long (was 1 in brondata) |

**Let op:** `remap_labels_fast.py` moet altijd op train/val/test gedraaid worden na het bouwen van sequences. Zonder remap crasht de training met CUDA label assertion (`t >= 0 && t < n_classes`).

---

## Dataset statistieken (v3)

| Split | Sequences |
|---|---|
| Train (full) | 1,924,495 |
| Train (balanced warmup, 1:2:1) | 99,036 |
| Validation | 412,291 |
| Test | 412,290 |

Klasse-verdeling validation: short=4,145 / hold=402,710 / long=5,436 (sterk ongebalanceerd → model beoordeeld op balanced accuracy)

---

## Model architectuur

```
LSTM hidden_size=512, num_layers=3, dropout=0.1
Input: sequence_length=64, features=37
Output heads:
  - action_taken (3 klassen: short/hold/long)
  - trade_side   (3 klassen)
  - net_equity_delta (regressie)
```

---

## Huidige run: `indicators_warmup_01`

**Doel:** Model opwarmen op balanced data (1:2:1) zodat het alle 3 klassen leert.

**Parameters:**
```
--hidden-size 512 --num-layers 3 --dropout 0.1
--lr 3e-4 --epochs 5 --batch-size 256
--class-weights 1.5 1.0 1.5
--checkpoint-every-steps 200
```

**Status:** Gestart (opnieuw na DGX crash van mei 2026)

**Checkpoint locatie:** `/home/dwyte/checkpoints/lstm_bbfit/indicators_warmup_01/`

---

## Eerdere runs (voor de crash, mei 2026)

Volgorde op basis van gitignore history:

1. **`indicators_warmup_01`** — warmup, getraind tot epoch 6 step 2328
2. **`indicators_finetune_01`** — finetune op de warmup
3. **`balanced_ft`** — finetune met balanced data, meerdere checkpoints (step 400 t/m 2556)
4. **`norm_focal_01`** — aparte tak met normalisatie + focal loss, tot epoch 5 step 1800

Metrics zijn verloren gegaan (lokaal opgeslagen, niet in GitHub of Google Drive).

---

## Bekende problemen & oplossingen

| Probleem | Oorzaak | Oplossing |
|---|---|---|
| `CUDA assertion t >= 0 && t < n_classes` | Validation/test labels zijn -1/0/1 i.p.v. 0/1/2 | `remap_labels_fast.py` draaien op val + test CSV's |
| Docker permission denied | User niet in docker groep | `sudo usermod -aG docker dwyte && newgrp docker` |
| tmux sessie verdwijnt meteen | Complexe command string mislukt in `new-session` | `send-keys` gebruiken i.p.v. command inline meegeven |

---

## Volgende stappen na warmup

1. Evalueren met `evaluate_lstm_bbfit.py` + threshold sweep
2. Finetune op volledige (ongebalanceerde) dataset
3. Eventueel TP/SL rebuild toevoegen: `--tp-pct X --sl-pct 0.10`
4. Backtest op testset

---

*Laatst bijgewerkt: 2026-05-03*
