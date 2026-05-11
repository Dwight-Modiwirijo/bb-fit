#!/usr/bin/env bash
# Pipeline v4b: BB-opt features, ZONDER PnL-filtering
# Reden: PnL-filtering creëert tegenstrijdige labels (zelfde condities -> TP=long, SL=hold)
# waardoor het model collapsed naar Hold. BB-opt features zijn de innovatie, niet PnL-filter.
#
# TP=2.4%  SL=1.2%  (ratio 2:1, break-even precision = 33%)
SESSION="build_tpsl_v4b"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "bash -c '
set -e
DOCKER=\"docker run --rm --ipc=host \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/logs:/workspace/logs:ro \
  -v /home/dwyte/bb-fit:/workspace/data \
  nvcr.io/nvidia/pytorch:25.06-py3\"

echo \"=== Stap 1: TP/SL exits (ZONDER PnL-filtering) ===\"
\$DOCKER python /workspace/scripts/rebuild_dataset_tpsl.py \
  --input-csv  /workspace/logs/lstm_merged.csv \
  --onemin-csv /workspace/logs/btcusd_1-min_data.csv \
  --output-csv /workspace/data/lstm_merged_tpsl_v4b.csv \
  --tp-pct 0.024 \
  --sl-pct 0.012

echo \"\"
echo \"=== Stap 2: Indicators + BB-optimalisatie features ===\"
\$DOCKER python /workspace/scripts/add_indicators.py \
  --input-csv  /workspace/data/lstm_merged_tpsl_v4b.csv \
  --output-csv /workspace/data/lstm_merged_tpsl_v4b_indicators.csv

echo \"\"
echo \"=== Stap 3: Sequences bouwen ===\"
\$DOCKER python /workspace/scripts/build_lstm_sequence_csvs_streaming.py \
  --input      /workspace/data/lstm_merged_tpsl_v4b_indicators.csv \
  --output-dir /workspace/data/sequences_tpsl_v4b

echo \"\"
echo \"=== Stap 4: Labels remappen ===\"
\$DOCKER python /workspace/scripts/remap_labels_fast.py \
  /workspace/data/sequences_tpsl_v4b/lstm_train_sequences.csv \
  /workspace/data/sequences_tpsl_v4b/lstm_validation_sequences.csv \
  /workspace/data/sequences_tpsl_v4b/lstm_test_sequences.csv

echo \"\"
echo \"=== Stap 5: Balanced warmup CSV (1:2:1) ===\"
\$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_tpsl_v4b/lstm_train_sequences.csv \
  --output /workspace/data/sequences_tpsl_v4b/lstm_train_balanced_warmup.csv \
  --majority-factor 2 \
  --seed 42

echo \"\"
echo \"=== Klaar ===\"
' 2>&1 | tee /home/dwyte/bb-fit/build_tpsl_v4b.log" Enter

echo "Build v4b gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
echo "Log: /home/dwyte/bb-fit/build_tpsl_v4b.log"
