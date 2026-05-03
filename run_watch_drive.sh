#!/usr/bin/env bash
# Draait in tmux, uploadt automatisch bestanden naar Google Drive.
# Vereist: inotify-tools (sudo apt-get install inotify-tools)
# Start met: bash run_watch_drive.sh
SESSION="watch_drive"
REMOTE="gdrive:DGX/bb-fit"
CKPT_DIR="/home/dwyte/checkpoints/lstm_bbfit"
SLOW_INTERVAL=300   # evals, logs, memory elke 5 min

tmux new-session -d -s "$SESSION" "bash -c '

# Eenmalig: brondata uploaden als die er nog niet op staat
echo \"[\$(date +\"%H:%M:%S\")] Eenmalig: brondata uploaden...\"
rclone copy \"/home/dwyte/logs/lstm_merged.csv\"        \"$REMOTE/brondata/\" &
rclone copy \"/home/dwyte/logs/btcusd_1-min_data.csv\" \"$REMOTE/brondata/\" &
wait
echo \"[\$(date +\"%H:%M:%S\")] Brondata klaar.\"

# Achtergrond: checkpoint watcher via inotifywait
(
  mkdir -p \"$CKPT_DIR\"
  while true; do
    inotifywait -q -e close_write -r \"$CKPT_DIR\" --include \".*\\.pt\" 2>/dev/null
    echo \"[\$(date +\"%H:%M:%S\")] Nieuw checkpoint gevonden! Uploaden...\"
    rclone copy \"$CKPT_DIR/\" \"$REMOTE/checkpoints/\" --include \"*.pt\"
    echo \"[\$(date +\"%H:%M:%S\")] Checkpoint geupload.\"
  done
) &

# Hoofdloop: evals, logs, memory elke 5 min
while true; do
  sleep $SLOW_INTERVAL

  echo \"[\$(date +\"%H:%M:%S\")] Syncing evals + logs + memory...\"

  rclone copy \"/home/dwyte/Github/bb-fit/\" \"$REMOTE/evals/\" \
    --include \"eval_*.json\" \
    --include \"threshold_sweep_*.json\" \
    --include \"sweep_*.json\"
  rclone copy \"/home/dwyte/bb-fit/\" \"$REMOTE/evals/\" \
    --include \"eval_*.json\" \
    --include \"threshold_sweep_*.json\" \
    --include \"sweep_*.json\"

  rclone copy \"/home/dwyte/bb-fit/\" \"$REMOTE/logs/\" \
    --include \"*.log\"

  rclone copy \"/home/dwyte/Github/bb-fit/STATUS.md\" \"$REMOTE/\"

  rclone copy \"/home/dwyte/.claude/projects/-home-dwyte-Github-bb-fit/memory/\" \"$REMOTE/claude-memory/\"

  echo \"[\$(date +\"%H:%M:%S\")] Sync klaar.\"
done
'"

echo "Drive watcher gestart in tmux sessie '$SESSION'."
echo "Attachen met: tmux attach -t $SESSION"
