#!/usr/bin/env bash
for CKPT in \
  checkpoint_epoch02_step0000400.pt \
  checkpoint_epoch02_step0000668.pt \
  checkpoint_epoch03_step0001140.pt \
  checkpoint_epoch04_step0001612.pt \
  checkpoint_epoch06_step0002556.pt; do

  STEP=$(echo $CKPT | grep -oP 'step\K[0-9]+')
  echo "=== $CKPT ==="
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /home/dwyte/bb-fit:/workspace/data \
    -v /home/dwyte/checkpoints:/workspace/checkpoints \
    nvcr.io/nvidia/pytorch:25.06-py3 \
    python /workspace/data/evaluate_lstm_bbfit.py \
    --train-csv      /workspace/data/lstm_train_balanced_finetune_01.csv \
    --validation-csv /workspace/data/lstm_validation_sequences.csv \
    --test-csv       /workspace/data/lstm_test_sequences.csv \
    --checkpoint /workspace/checkpoints/lstm_bbfit/balanced_finetune_01/${CKPT} \
    --hidden-size 512 --num-layers 3 --dropout 0.1 \
    --output-json /workspace/data/eval_balanced_ft_step${STEP}.json \
    2>&1 | grep -E "balanced_accuracy|\"recall\"|\"precision\"|Loaded checkpoint"
  echo ""
done
