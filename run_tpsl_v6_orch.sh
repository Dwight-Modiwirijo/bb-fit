#!/usr/bin/env bash
# Standalone orchestrator for v6 (use this if run_build_v6.sh crashed after warmup).
# Warmup checkpoint must already exist in /home/dwyte/checkpoints/lstm_bbfit/v6_warmup/
#
# To resume from a specific round:
#   ROUND=3 CKPT=/workspace/checkpoints/lstm_bbfit/v6_orch_r02/checkpoint_epoch05.pt bash run_tpsl_v6_orch.sh
SESSION="tpsl_v6_orch"

ROUND="${ROUND:-1}"
CKPT="${CKPT:-}"

EXTRA_ARGS=""
if [ -n "$CKPT" ]; then
  EXTRA_ARGS="--skip-warmup-wait --start-round $ROUND --start-checkpoint $CKPT"
fi

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "python3 /home/dwyte/Github/bb-fit/orchestrate_tpsl_v6.py \
  $EXTRA_ARGS \
  2>&1 | tee /home/dwyte/bb-fit/v6_orchestrator.log" Enter

echo "V6 orchestrator started in tmux session '$SESSION'."
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Log:     /home/dwyte/bb-fit/v6_orchestrator.log"
echo "Report:  /home/dwyte/bb-fit/orchestrator_v6_report.md"
echo ""
echo "To resume after crash:"
echo "  ROUND=3 CKPT=/workspace/checkpoints/lstm_bbfit/v6_orch_r02/checkpoint_epoch05.pt bash run_tpsl_v6_orch.sh"
