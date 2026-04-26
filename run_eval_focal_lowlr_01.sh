#!/usr/bin/env bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/data/evaluate_lstm_bbfit.py \
  --train-csv /workspace/data/lstm_train_recalibrated.csv \
  --validation-csv /workspace/data/lstm_validation_sequences.csv \
  --test-csv /workspace/data/lstm_test_sequences.csv \
  --checkpoint /workspace/checkpoints/lstm_bbfit/focal_lowlr_01/checkpoint_epoch04_step0006112.pt \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --output-json /workspace/data/eval_focal_lowlr_01_ep4.json \
  2>&1 | tee /workspace/data/eval_focal_lowlr_01.log
