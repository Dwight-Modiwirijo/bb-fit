#!/usr/bin/env bash
# Finetune 04: verder trainen vanuit finetune_03 epoch14, 10 extra epochs, zelfde 1:1:1 data
SESSION="tpsl_finetune_04"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/scripts/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_tpsl_v1/lstm_train_balanced_111.csv \
  --validation-csv /workspace/data/sequences_tpsl_v1/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_tpsl_v1/lstm_test_sequences.csv \
  --resume-checkpoint /workspace/checkpoints/lstm_bbfit/tpsl_finetune_03/checkpoint_epoch14_step0003384.pt \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 2e-5 --epochs 10 --batch-size 256 \
  --focal-gamma 2.0 --class-weights 1.0 1.0 1.0 \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/tpsl_finetune_04 \
  --checkpoint-every-steps 100 \
  2>&1 | tee /workspace/data/tpsl_finetune_04.log" Enter

echo "Finetune_04 gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
