#!/usr/bin/env bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/data/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_indicators_v2/lstm_train_balanced_warmup.csv \
  --validation-csv /workspace/data/sequences_indicators_v2/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_indicators_v2/lstm_test_sequences.csv \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 3e-4 --epochs 5 --batch-size 256 \
  --class-weights 1.5 1.0 1.5 \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/indicators_warmup_01 \
  --checkpoint-every-steps 200 \
  2>&1 | tee /workspace/data/indicators_warmup_01.log
