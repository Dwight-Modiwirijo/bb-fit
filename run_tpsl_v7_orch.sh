#!/usr/bin/env bash
# Standalone v7 orchestrator (crash recovery — warmup checkpoint must exist).
#
# Resume from specific round:
#   ROUND=3 CKPT=/workspace/checkpoints/lstm_bbfit/v7_orch_r02/checkpoint_epoch05.pt bash run_tpsl_v7_orch.sh
SESSION="tpsl_v7_orch"

ROUND="${ROUND:-1}"
CKPT="${CKPT:-}"

EXTRA_ARGS=""
if [ -n "$CKPT" ]; then
  EXTRA_ARGS="--skip-warmup-wait --start-round $ROUND --start-checkpoint $CKPT"
fi

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "python3 /home/dwyte/Github/bb-fit/orchestrate_tpsl_v7.py \
  $EXTRA_ARGS \
  2>&1 | tee /home/dwyte/bb-fit/v7_orchestrator.log" Enter

echo "V7 orchestrator started in tmux '$SESSION'."
echo "Attach:  tmux attach -t $SESSION"
echo "Log:     /home/dwyte/bb-fit/v7_orchestrator.log"
echo "Report:  /home/dwyte/bb-fit/orchestrator_v7_report.md"
echo ""
echo "Resume after crash:"
echo "  ROUND=3 CKPT=/workspace/checkpoints/lstm_bbfit/v7_orch_r02/checkpoint_epoch05.pt bash run_tpsl_v7_orch.sh"
