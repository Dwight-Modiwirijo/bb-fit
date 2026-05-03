#!/usr/bin/env bash
for CKPT in \
  checkpoint_epoch01_step0000200.pt \
  checkpoint_epoch01_step0000388.pt \
  checkpoint_epoch02_step0000400.pt \
  checkpoint_epoch02_step0000776.pt \
  checkpoint_epoch03_step0001164.pt; do

  STEP=$(echo $CKPT | grep -oP 'step\K[0-9]+')
  echo "=== Evaluating $CKPT ==="
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/dwyte/bb-fit:/workspace/data \
    -v /home/dwyte/checkpoints:/workspace/checkpoints \
    nvcr.io/nvidia/pytorch:25.06-py3 \
    python /workspace/data/evaluate_lstm_bbfit.py \
    --train-csv      /workspace/data/sequences_indicators_v3/lstm_train_balanced_warmup.csv \
    --validation-csv /workspace/data/sequences_indicators_v3/lstm_validation_sequences.csv \
    --test-csv       /workspace/data/sequences_indicators_v3/lstm_test_sequences.csv \
    --checkpoint /workspace/checkpoints/lstm_bbfit/indicators_warmup_01/${CKPT} \
    --hidden-size 512 --num-layers 3 --dropout 0.1 \
    --output-json /workspace/data/eval_indicators_warmup_step${STEP}.json \
    2>&1 | grep -E "balanced_accuracy|class.*recall|class.*precision|Loaded checkpoint"
  echo ""
done
