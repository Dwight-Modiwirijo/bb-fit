#!/usr/bin/env bash
# Start de v5 orchestrator. Wacht automatisch op warmup, dan finetune -> eval -> DoD loop.
# TP=0.8% SL=1.8%, break-even precision=69.2%, DoD: long_prec>=75%
SESSION="tpsl_v5_orch"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "python3 /home/dwyte/Github/bb-fit/orchestrate_tpsl_v5.py \
  2>&1 | tee /home/dwyte/bb-fit/v5_orchestrator.log" Enter

echo "V5 orchestrator gestart in tmux sessie '$SESSION'."
echo "Wacht op v5_warmup checkpoint, daarna automatisch finetune -> eval -> DoD."
echo ""
echo "Attachen met: tmux attach -t $SESSION"
echo "Log:          /home/dwyte/bb-fit/v5_orchestrator.log"
echo "Rapport:      /home/dwyte/bb-fit/orchestrator_v5_report.md"
