#!/usr/bin/env bash
# Bouw 1:1:1 balanced dataset voor tpsl_v1 (short=hold=long=19820)
SESSION="build_tpsl_111"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "docker run --rm --ipc=host \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/build_balanced_warmup_csv.py \
  --input  /workspace/data/sequences_tpsl_v1/lstm_train_sequences.csv \
  --output /workspace/data/sequences_tpsl_v1/lstm_train_balanced_111.csv \
  --majority-factor 1 \
  --seed 42 \
  2>&1 | tee /workspace/data/build_tpsl_111.log && \
  echo DONE_111" Enter

echo "Build 1:1:1 gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
