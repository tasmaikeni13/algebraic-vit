#!/usr/bin/env bash
# Run Python on one TPU chip without joining the multi-host slice.
# Usage: bash scripts/tpu_python.sh <chip 0-3> <python arguments...>
set -euo pipefail
chip=${1:?specify a TPU chip (0-3)}
shift
case "$chip" in 0|1|2|3) ;; *) echo 'chip must be 0, 1, 2, or 3' >&2; exit 2 ;; esac
VISION_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS="$chip"
export TPU_PROCESS_ADDRESSES="localhost:$((8476+chip))" TPU_PROCESS_PORT="$((8476+chip))" CLOUD_TPU_TASK_ID=0
export JAX_PLATFORMS=tpu JAX_ENABLE_X64=0
export PYTHONPATH="$VISION_ROOT/src:$VISION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -x "$VISION_ROOT/.venv/bin/python" ]; then
  VISION_DEFAULT_PYTHON="$VISION_ROOT/.venv/bin/python"
elif [ -x "$HOME/venv/bin/python" ]; then
  VISION_DEFAULT_PYTHON="$HOME/venv/bin/python"
else
  VISION_DEFAULT_PYTHON=python3
fi
cd "$VISION_ROOT"
exec "${VISION_PYTHON:-$VISION_DEFAULT_PYTHON}" "$@"
