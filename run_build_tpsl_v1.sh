#!/usr/bin/env bash
# Pipeline: TP/SL exits → indicators → sequences → remap → balanced warmup
# SL=1.2%  TP=3.6%  (ratio 3:1, break-even precision = 25%)
SESSION="build_tpsl_v1"

tmux new-session -d -s "$SESSION" "bash -c '
set -e
DOCKER=\"docker run --rm --ipc=host \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/logs:/workspace/logs:ro \
  -v /home/dwyte/bb-fit:/workspace/data \
  nvcr.io/nvidia/pytorch:25.06-py3\"

echo \"=== Stap 1: TP/SL exits toepassen (SL=1.2%  TP=3.6%) ===\"
\$DOCKER python /workspace/scripts/rebuild_dataset_tpsl.py \
  --input-csv  /workspace/logs/lstm_merged.csv \
  --onemin-csv /workspace/logs/btcusd_1-min_data.csv \
  --output-csv /workspace/data/lstm_merged_tpsl_v1.csv \
  --tp-pct 0.036 \
  --sl-pct 0.012

echo \"\"
echo \"=== Stap 2: Indicators berekenen ===\"
\$DOCKER python /workspace/scripts/add_indicators.py \
  --input-csv  /workspace/data/lstm_merged_tpsl_v1.csv \
  --output-csv /workspace/data/lstm_merged_tpsl_v1_indicators.csv

echo \"\"
echo \"=== Stap 3: Sequences bouwen ===\"
\$DOCKER python /workspace/scripts/build_lstm_sequence_csvs_streaming.py \
  --input      /workspace/data/lstm_merged_tpsl_v1_indicators.csv \
  --output-dir /workspace/data/sequences_tpsl_v1

echo \"\"
echo \"=== Stap 4: Labels remappen (-1/0/1 → 0/1/2) ===\"
\$DOCKER python /workspace/scripts/remap_labels_fast.py \
  /workspace/data/sequences_tpsl_v1/lstm_train_sequences.csv \
  /workspace/data/sequences_tpsl_v1/lstm_validation_sequences.csv \
  /workspace/data/sequences_tpsl_v1/lstm_test_sequences.csv

echo \"\"
echo \"=== Stap 5: Balanced warmup CSV (1:2:1) ===\"
\$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_tpsl_v1/lstm_train_sequences.csv \
  --output /workspace/data/sequences_tpsl_v1/lstm_train_balanced_warmup.csv \
  --majority-factor 2 \
  --seed 42

echo \"\"
echo \"=== Klaar ===\"
'"

echo "Build gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
