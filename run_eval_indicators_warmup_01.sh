#!/usr/bin/env bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python /workspace/data/evaluate_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_indicators_v2/lstm_train_balanced_warmup.csv \
  --validation-csv /workspace/data/sequences_indicators_v2/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_indicators_v2/lstm_test_sequences.csv \
  --checkpoint /workspace/checkpoints/lstm_bbfit/indicators_warmup_01/checkpoint_epoch06_step0002328.pt \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --output-json /workspace/data/eval_indicators_warmup_01_ep6.json \
  2>&1 | tee /home/dwyte/bb-fit/eval_indicators_warmup_01.log
