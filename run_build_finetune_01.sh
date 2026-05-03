#!/usr/bin/env bash
SESSION="build_finetune_01"

tmux new-session -d -s "$SESSION" "bash -c '
set -e
DOCKER=\"docker run --rm --ipc=host \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/bb-fit:/workspace/data \
  nvcr.io/nvidia/pytorch:25.06-py3\"

echo \"=== Balanced finetune CSV (1:10:1) ===\"
\$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_indicators_v3/lstm_train_sequences.csv \
  --output /workspace/data/sequences_indicators_v3/lstm_train_balanced_finetune_01.csv \
  --majority-factor 10 \
  --seed 42

echo \"=== Klaar ===\"
'"

echo "Build gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
