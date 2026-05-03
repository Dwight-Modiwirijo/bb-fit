#!/usr/bin/env bash
# Upload checkpoints, eval JSONs en STATUS.md naar Google Drive DGX/bb-fit/
REMOTE="gdrive:DGX/bb-fit"

echo "=== Uploading checkpoints ==="
rclone copy /home/dwyte/checkpoints/lstm_bbfit/ "$REMOTE/checkpoints/" \
  --include "*.pt" \
  --progress

echo ""
echo "=== Uploading eval JSONs ==="
rclone copy /home/dwyte/Github/bb-fit/ "$REMOTE/evals/" \
  --include "eval_*.json" \
  --include "threshold_sweep_*.json" \
  --include "sweep_*.json" \
  --progress

echo ""
echo "=== Uploading STATUS.md ==="
rclone copy /home/dwyte/Github/bb-fit/STATUS.md "$REMOTE/" --progress

echo ""
echo "=== Klaar ==="
