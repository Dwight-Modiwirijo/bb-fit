#!/usr/bin/env bash
# Draait in tmux, uploadt automatisch nieuwe checkpoints naar Google Drive.
# Start met: bash run_watch_drive.sh
SESSION="watch_drive"
REMOTE="gdrive:DGX/bb-fit"
CKPT_DIR="/home/dwyte/checkpoints/lstm_bbfit"
INTERVAL=300  # elke 5 minuten checken

tmux new-session -d -s "$SESSION" "bash -c '
while true; do
  echo \"[\$(date +\"%H:%M:%S\")] Syncing checkpoints...\"
  rclone copy \"$CKPT_DIR/\" \"$REMOTE/checkpoints/\" --include \"*.pt\"

  echo \"[\$(date +\"%H:%M:%S\")] Syncing evals...\"
  rclone copy \"/home/dwyte/Github/bb-fit/\" \"$REMOTE/evals/\" \
    --include \"eval_*.json\" \
    --include \"threshold_sweep_*.json\" \
    --include \"sweep_*.json\"

  rclone copy \"/home/dwyte/Github/bb-fit/STATUS.md\" \"$REMOTE/\"

  echo \"[\$(date +\"%H:%M:%S\")] Done. Wachten $INTERVAL seconden...\"
  sleep $INTERVAL
done
'"

echo "Drive watcher gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
