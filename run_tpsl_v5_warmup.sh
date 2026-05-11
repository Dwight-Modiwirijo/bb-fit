#!/usr/bin/env bash
# Warmup training v5: BB-opt features, TP=0.8%/SL=1.8%, 35 features, 1:2:1 balanced
# lr=3e-4, 5 epochs, class-weights 1.5/1.0/1.5
SESSION="tpsl_v5_warmup"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_tpsl_v5/lstm_train_balanced_warmup.csv \
  --validation-csv /workspace/data/sequences_tpsl_v5/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_tpsl_v5/lstm_test_sequences.csv \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 3e-4 --epochs 5 --batch-size 256 \
  --class-weights 1.5 1.0 1.5 \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/v5_warmup \
  --checkpoint-every-steps 200 \
  --lr-scheduler plateau --scheduler-patience 2 \
  2>&1 | tee /workspace/data/v5_warmup.log" Enter

echo "V5 warmup gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
echo "Log: /home/dwyte/bb-fit/v5_warmup.log"
