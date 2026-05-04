#!/usr/bin/env bash
SESSION="tpsl_warmup_01"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_tpsl_v1/lstm_train_balanced_warmup.csv \
  --validation-csv /workspace/data/sequences_tpsl_v1/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_tpsl_v1/lstm_test_sequences.csv \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 3e-4 --epochs 5 --batch-size 256 \
  --class-weights 1.5 1.0 1.5 \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/tpsl_warmup_01 \
  --checkpoint-every-steps 200 \
  2>&1 | tee /workspace/data/tpsl_warmup_01.log" Enter

echo "Warmup training gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
