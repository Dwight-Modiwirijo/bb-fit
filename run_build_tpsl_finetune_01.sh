#!/usr/bin/env bash
# Bouw 1:10:1 balanced finetune dataset voor tpsl_v1
SESSION="build_tpsl_finetune_01"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "docker run --rm --ipc=host \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_tpsl_v1/lstm_train_sequences.csv \
  --output /workspace/data/sequences_tpsl_v1/lstm_train_balanced_finetune_01.csv \
  --majority-factor 10 \
  --seed 42 \
  2>&1 | tee /workspace/data/build_tpsl_finetune_01.log" Enter

echo "Build gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
