#!/usr/bin/env bash
SESSION="eval_finetune_01"

tmux new-session -d -s "$SESSION" \
  "docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/evaluate_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_indicators_v3/lstm_train_balanced_finetune_01.csv \
  --validation-csv /workspace/data/sequences_indicators_v3/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_indicators_v3/lstm_test_sequences.csv \
  --checkpoint /workspace/checkpoints/lstm_bbfit/finetune_01/checkpoint_epoch10_step0007760.pt \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --output-json /workspace/data/eval_finetune_01_ep10.json \
  2>&1 | tee /workspace/data/eval_finetune_01.log"

echo "Evaluatie gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
