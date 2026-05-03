#!/usr/bin/env bash
SESSION="finetune_01"

tmux new-session -d -s "$SESSION" \
  "docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_indicators_v3/lstm_train_balanced_finetune_01.csv \
  --validation-csv /workspace/data/sequences_indicators_v3/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_indicators_v3/lstm_test_sequences.csv \
  --resume-checkpoint /workspace/checkpoints/lstm_bbfit/indicators_warmup_01/checkpoint_epoch05_step0001940.pt \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 1e-5 --epochs 5 --batch-size 256 \
  --focal-gamma 2.0 --class-weights 4.0 1.0 3.0 \
  --reset-optimizer \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/finetune_01 \
  --checkpoint-every-steps 200 \
  2>&1 | tee /workspace/data/finetune_01.log"

echo "Finetune gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
