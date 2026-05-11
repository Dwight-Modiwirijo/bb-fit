#!/usr/bin/env bash
# Start de v4b orchestrator. Wacht automatisch op warmup, dan finetune -> eval -> DoD loop.
# BB-opt features, ZONDER PnL-filtering.
# Als warmup al klaar is: gewoon starten, orchestrator detecteert het checkpoint.
SESSION="tpsl_v4b_orch"

tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "python3 /home/dwyte/Github/bb-fit/orchestrate_tpsl_v4b.py \
  2>&1 | tee /home/dwyte/bb-fit/v4b_orchestrator.log" Enter

echo "V4b orchestrator gestart in tmux sessie '$SESSION'."
echo "Wacht op v4b_warmup checkpoint, daarna automatisch finetune -> eval -> DoD."
echo ""
echo "Attachen met: tmux attach -t $SESSION"
echo "Log:          /home/dwyte/bb-fit/v4b_orchestrator.log"
echo "Rapport:      /home/dwyte/bb-fit/orchestrator_v4b_report.md"
