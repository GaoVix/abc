#!/usr/bin/env bash
set -e
cd "$(dirname "$0")" || exit 1

export TRAIN_CACHE_ROOT="${RETARGET_OUTPUT_ROOT:-cache/dishrack_retargeted_official_timing}"
export TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-output/dit_joint_retargeted_official_timing_$(date +%Y%m%d_%H%M%S)}"

.venv/bin/python scripts/export_retargeted_joints_official_timing.py \
  --source "${RETARGET_SOURCE_ROOT:-cache/dishrack_real}" \
  --output "$TRAIN_CACHE_ROOT" \
  --workers "${RETARGET_WORKERS:-16}"

exec bash train.sh
