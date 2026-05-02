#!/usr/bin/env bash
set -e

echo "=== Sequences bouwen (v3, 37 features) ==="
python /home/dwyte/bb-fit/build_lstm_sequence_csvs_streaming.py \
  --input      /home/dwyte/bb-fit/lstm_merged_indicators_v3.csv \
  --output-dir /home/dwyte/bb-fit/sequences_indicators_v3

echo ""
echo "=== Balanced warmup CSV (1:2:1) ==="
python /home/dwyte/bb-fit/build_balanced_warmup_csv.py \
  --input  /home/dwyte/bb-fit/sequences_indicators_v3/lstm_train_sequences.csv \
  --output /home/dwyte/bb-fit/sequences_indicators_v3/lstm_train_balanced_warmup.csv \
  --majority-factor 2 \
  --seed 42

echo ""
echo "=== Klaar ==="
