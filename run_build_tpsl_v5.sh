#!/usr/bin/env bash
# Pipeline v5: BB-opt features, TP=0.8% SL=1.8% (inverted ratio 0.44:1)
# Reden: backtesting toont 73.9% win rate bij TP=0.8%/SL=1.8% vs 27.9% bij TP=2.4%/SL=1.2%.
# Break-even precision = 69.2%. Zelfs zonder modelfiltering al winstgevend.
#
# TP=0.8%  SL=1.8%  (ratio 0.44:1, break-even precision = 69.2%)
SESSION="build_tpsl_v5"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "bash -c '
set -e
DOCKER=\"docker run --rm --ipc=host \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/logs:/workspace/logs:ro \
  -v /home/dwyte/bb-fit:/workspace/data \
  nvcr.io/nvidia/pytorch:25.06-py3\"

echo \"=== Stap 1: TP/SL exits (TP=0.8% SL=1.8%) ===\"
\$DOCKER python /workspace/scripts/rebuild_dataset_tpsl.py \
  --input-csv  /workspace/logs/lstm_merged.csv \
  --onemin-csv /workspace/logs/btcusd_1-min_data.csv \
  --output-csv /workspace/data/lstm_merged_tpsl_v5.csv \
  --tp-pct 0.008 \
  --sl-pct 0.018

echo \"\"
echo \"=== Stap 2: Indicators + BB-optimalisatie features ===\"
\$DOCKER python /workspace/scripts/add_indicators.py \
  --input-csv  /workspace/data/lstm_merged_tpsl_v5.csv \
  --output-csv /workspace/data/lstm_merged_tpsl_v5_indicators.csv

echo \"\"
echo \"=== Stap 3: Sequences bouwen ===\"
\$DOCKER python /workspace/scripts/build_lstm_sequence_csvs_streaming.py \
  --input      /workspace/data/lstm_merged_tpsl_v5_indicators.csv \
  --output-dir /workspace/data/sequences_tpsl_v5

echo \"\"
echo \"=== Stap 4: Labels remappen ===\"
\$DOCKER python /workspace/scripts/remap_labels_fast.py \
  /workspace/data/sequences_tpsl_v5/lstm_train_sequences.csv \
  /workspace/data/sequences_tpsl_v5/lstm_validation_sequences.csv \
  /workspace/data/sequences_tpsl_v5/lstm_test_sequences.csv

echo \"\"
echo \"=== Stap 5: Balanced warmup CSV (1:2:1) ===\"
\$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_tpsl_v5/lstm_train_sequences.csv \
  --output /workspace/data/sequences_tpsl_v5/lstm_train_balanced_warmup.csv \
  --majority-factor 2 \
  --seed 42

echo \"\"
echo \"=== Klaar ===\"
' 2>&1 | tee /home/dwyte/bb-fit/build_tpsl_v5.log" Enter

echo "Build v5 gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
echo "Log: /home/dwyte/bb-fit/build_tpsl_v5.log"
