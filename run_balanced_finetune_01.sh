#!/usr/bin/env bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/data/train_lstm_bbfit.py \
  --train-csv      /workspace/data/lstm_train_balanced_finetune_01.csv \
  --validation-csv /workspace/data/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/lstm_test_sequences.csv \
  --resume-checkpoint /workspace/checkpoints/lstm_bbfit/classification_warmup_03/checkpoint_epoch01_step0000196.pt \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 1e-5 --epochs 5 --batch-size 256 \
  --focal-gamma 2.0 --class-weights 3.0 1.0 3.0 \
  --reset-optimizer \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/balanced_finetune_01 \
  --checkpoint-every-steps 200 \
  2>&1 | tee /home/dwyte/bb-fit/balanced_finetune_01.log
