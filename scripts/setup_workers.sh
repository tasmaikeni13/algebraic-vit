#!/usr/bin/env bash
# Sync source to the same absolute path on the named TPU slice workers.
# Run from worker zero. Set up SSH aliases and install requirements-tpu.txt
# plus this package on each worker first. Copy datasets separately.
set -euo pipefail
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi
VISION_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VISION_HOSTS=("$@")
if [ ${#VISION_HOSTS[@]} -eq 0 ]; then VISION_HOSTS=(w0 w1 w2 w3); fi
for worker in "${VISION_HOSTS[@]}"; do
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '+ ssh %s mkdir -p %s (hostname check + rsync skipped in --dry-run)\n' \
      "$worker" "$VISION_ROOT"
    continue
  fi
  ssh "$worker" "mkdir -p '$VISION_ROOT'"
  # Worker zero already has the authoritative checkout.
  if [ "$(ssh "$worker" hostname)" = "$(hostname)" ]; then continue; fi
  rsync -a --delete --exclude .git --exclude .lake --exclude runs --exclude /data \
    --exclude .venv --exclude '*.egg-info' --exclude '__pycache__' --exclude .pytest_cache \
    "$VISION_ROOT/" "$worker:$VISION_ROOT/"
done
if [ "$DRY_RUN" -eq 1 ]; then printf 'Dry run: no workers touched.\n'; else printf 'Workers synced.\n'; fi
