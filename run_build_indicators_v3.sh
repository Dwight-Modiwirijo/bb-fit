#!/usr/bin/env bash
SESSION="build_indicators_v3"

tmux new-session -d -s "$SESSION" "bash -c '
set -e
DOCKER=\"docker run --rm --ipc=host \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/logs:/workspace/logs:ro \
  -v /home/dwyte/bb-fit:/workspace/data \
  nvcr.io/nvidia/pytorch:25.06-py3\"

echo \"=== Stap 1: Indicators berekenen ===\"
\$DOCKER python /workspace/scripts/add_indicators.py \
  --input-csv  /workspace/logs/lstm_merged.csv \
  --output-csv /workspace/data/lstm_merged_indicators_v3.csv

echo \"\"
echo \"=== Stap 2: Sequences bouwen ===\"
\$DOCKER python /workspace/scripts/build_lstm_sequence_csvs_streaming.py \
  --input      /workspace/data/lstm_merged_indicators_v3.csv \
  --output-dir /workspace/data/sequences_indicators_v3

echo \"\"
echo \"=== Stap 3: Labels remappen (-1/0/1 → 0/1/2) ===\"
\$DOCKER python /workspace/scripts/remap_labels_fast.py \
  /workspace/data/sequences_indicators_v3/lstm_train_sequences.csv \
  /workspace/data/sequences_indicators_v3/lstm_validation_sequences.csv \
  /workspace/data/sequences_indicators_v3/lstm_test_sequences.csv

echo \"\"
echo \"=== Stap 4: Balanced warmup CSV (1:2:1) ===\"
\$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_indicators_v3/lstm_train_sequences.csv \
  --output /workspace/data/sequences_indicators_v3/lstm_train_balanced_warmup.csv \
  --majority-factor 2 \
  --seed 42

echo \"\"
echo \"=== Klaar ===\"
'"

echo "Build gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
