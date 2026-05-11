#!/usr/bin/env bash
# Full v6 pipeline: add_v6_features → sequences → remap → balanced warmup → warmup train → orchestrator
#
# One command to run everything. Attach to tmux session to watch progress.
# tmux attach -t build_v6
#
# TP=0.8%  SL=1.8%  Long-only  FiveMinutes only  30 normalised TAengine features
SESSION="build_v6"

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

tmux new-session -d -s "$SESSION"

tmux send-keys -t "$SESSION" "bash -c '
set -e

echo \"=== Step 1: Filter FiveMinutes + add normalised v6 features ===\"
$VENV_PYTHON /home/dwyte/Github/bb-fit/add_v6_features.py \
  --input  /home/dwyte/bb-fit/lstm_merged.csv \
  --output /home/dwyte/bb-fit/lstm_merged_v6.csv \
  --interval FiveMinutes
echo \"Step 1 done.\"

echo \"\"
echo \"=== Step 2: Build sequences (v6 feature set, seq_len=64) ===\"
$DOCKER python /workspace/scripts/build_lstm_sequence_csvs_streaming.py \
  --input      /workspace/data/lstm_merged_v6.csv \
  --output-dir /workspace/data/sequences_tpsl_v6 \
  --feature-set v6 \
  --sequence-length 64
echo \"Step 2 done.\"

echo \"\"
echo \"=== Step 3: Remap labels (-1/0/1 → 0/1/2) ===\"
$DOCKER python /workspace/scripts/remap_labels_fast.py \
  /workspace/data/sequences_tpsl_v6/lstm_train_sequences.csv \
  /workspace/data/sequences_tpsl_v6/lstm_validation_sequences.csv \
  /workspace/data/sequences_tpsl_v6/lstm_test_sequences.csv
echo \"Step 3 done.\"

echo \"\"
echo \"=== Step 4: Build balanced warmup CSV (Hold:Long:Short = 2:1:1) ===\"
$DOCKER python /workspace/scripts/build_balanced_warmup_csv.py \
  --input          /workspace/data/sequences_tpsl_v6/lstm_train_sequences.csv \
  --output         /workspace/data/sequences_tpsl_v6/lstm_train_balanced_warmup.csv \
  --majority-factor 2 \
  --seed 42
echo \"Step 4 done.\"

echo \"\"
echo \"=== Pipeline complete. Starting warmup training... ===\"
$DOCKER_GPU python /workspace/scripts/train_lstm_bbfit.py \
  --train-csv      /workspace/data/sequences_tpsl_v6/lstm_train_balanced_warmup.csv \
  --validation-csv /workspace/data/sequences_tpsl_v6/lstm_validation_sequences.csv \
  --test-csv       /workspace/data/sequences_tpsl_v6/lstm_test_sequences.csv \
  --hidden-size 512 --num-layers 3 --dropout 0.1 \
  --lr 3e-4 --epochs 5 --batch-size 256 \
  --class-weights 1.5 1.0 1.5 \
  --checkpoint-dir /workspace/checkpoints/lstm_bbfit/v6_warmup \
  --checkpoint-every-steps 200 \
  --lr-scheduler plateau --scheduler-patience 2 \
  2>&1 | tee /home/dwyte/bb-fit/v6_warmup.log
echo \"Warmup done.\"

echo \"\"
echo \"=== Starting orchestrator (finetune -> eval -> DoD loop) ===\"
python3 /home/dwyte/Github/bb-fit/orchestrate_tpsl_v6.py \
  --skip-warmup-wait \
  2>&1 | tee /home/dwyte/bb-fit/v6_orchestrator.log

' 2>&1 | tee /home/dwyte/bb-fit/build_v6_pipeline.log" Enter

echo "V6 pipeline started in tmux session '$SESSION'."
echo ""
echo "Attach:    tmux attach -t $SESSION"
echo "Full log:  /home/dwyte/bb-fit/build_v6_pipeline.log"
echo "Warmup:    /home/dwyte/bb-fit/v6_warmup.log"
echo "Orch:      /home/dwyte/bb-fit/v6_orchestrator.log"
echo "Report:    /home/dwyte/bb-fit/orchestrator_v6_report.md"
