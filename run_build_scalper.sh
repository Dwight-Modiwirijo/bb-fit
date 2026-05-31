#!/usr/bin/env bash
# Scalper pipeline: parse JSONL → add features → sequences → balanced warmup → train
#
# Input:  Trader/logs/XXBTZEUR_TwoHunderdAndFourty_LongOnly_*.jsonl  (with future_return fields)
# Target: label_direction (0=Down, 1=Flat, 2=Up) from future_return_10 ± 2%
# Seq:    64 × 30 features (v6 feature set, 240-min candles)
#
# Attach: tmux attach -t build_scalper

SESSION="build_scalper"

VENV_PYTHON="/tmp/venv_bbfit/bin/python3"

DOCKER="docker run --rm --ipc=host \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3"

DOCKER_GPU="docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/dwyte/Github/bb-fit:/workspace/scripts \
  -v /home/dwyte/bb-fit:/workspace/data \
  -v /home/dwyte/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.06-py3"

# Newest JSONL with future_return fields (null-fixed version)
JSONL_DIR="/home/dwyte/bb-fit/Trader/logs"
JSONL_PATTERN="XXBTZEUR_TwoHunderdAndFourty_LongOnly_0,0005_20260531183054_*.jsonl"

tmux new-session -d -s "$SESSION"

tmux send-keys -t "$SESSION" "bash -c '
set -e

echo \"=== Step 1: Parse JSONL → lstm_merged_scalper.csv ===\"
$VENV_PYTHON /home/dwyte/Github/bb-fit/parse_testlog_to_csv.py \
  $JSONL_DIR/$JSONL_PATTERN \
  --output /home/dwyte/bb-fit/lstm_merged_scalper.csv
echo \"Step 1 done.\"

echo \"\"
echo \"=== Step 2: Add normalised features + label_direction ===\"
$VENV_PYTHON /home/dwyte/Github/bb-fit/add_scalper_features.py \
  --input     /home/dwyte/bb-fit/lstm_merged_scalper.csv \
  --output    /home/dwyte/bb-fit/lstm_merged_scalper_features.csv \
  --interval  TwoHunderdAndFourty \
  --threshold 0.02
echo \"Step 2 done.\"

echo \"\"
echo \"=== Step 3: Build sequences (scalper feature set, seq_len=64) ===\"
$DOCKER python /workspace/scripts/build_lstm_sequence_csvs_streaming.py \
  --input      /workspace/data/lstm_merged_scalper_features.csv \
  --output-dir /workspace/data/sequences_scalper \
  --feature-set scalper \
  --sequence-length 64
echo \"Step 3 done.\"

echo \"\"
echo \"=== Step 4: Build balanced warmup CSV (Down:Flat:Up = 1:2:1) ===\"
$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input          /workspace/data/sequences_scalper/lstm_train_sequences.csv \
  --output         /workspace/data/sequences_scalper/lstm_train_balanced_warmup.csv \
  --label-column   label_direction \
  --majority-factor 2 \
  --seed 42
echo \"Step 4 done.\"

echo \"\"
echo \"=== Step 5: Warmup training ===\"
$DOCKER_GPU python /workspace/scripts/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_scalper/lstm_train_balanced_warmup.csv \
  --validation-csv /workspace/data/sequences_scalper/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_scalper/lstm_test_sequences.csv \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 3e-4 --epochs 5 --batch-size 256 \
  --label-column   label_direction \
  --class-weights 1.5 1.0 1.5 \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/scalper_warmup \
  --checkpoint-every-steps 200 \
  --lr-scheduler plateau --scheduler-patience 2 \
  2>&1 | tee /home/dwyte/bb-fit/scalper_warmup.log
echo \"Warmup done.\"

echo \"\"
echo \"=== Pipeline complete ===\"

' 2>&1 | tee /home/dwyte/bb-fit/build_scalper_pipeline.log" Enter

echo "Scalper pipeline started in tmux session '$SESSION'."
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Log:     /home/dwyte/bb-fit/build_scalper_pipeline.log"
echo "Warmup:  /home/dwyte/bb-fit/scalper_warmup.log"
