#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
WORK_DIR="${WORK_DIR:-$REPO/outputs/right_anchor_sequence_ik_$(date +%Y%m%d_%H%M%S)}"
CALIBRATION_DIR="${CALIBRATION_DIR:-$REPO/outputs/cape_calibration}"
GRID="${GRID:-/media/ps/T7/Datasets/ABC-130k/eef_pose_grid_margin10_gpu_20261029}"
GRID_INDEX="${GRID_INDEX:-/media/ps/T7/Datasets/ABC-130k/eef_pose_grid_margin10_index_20261029}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

"$REPO/.venv/bin/python" -m scripts.convert_right_anchor --solve-only \
  --state-sources "${STATE_SOURCES:-$REPO/outputs/all_motion_states_margin15_median_20261101/source}" \
  --action-sources "${ACTION_SOURCES:-$REPO/outputs/median_base_grid_fallback_bounded_ik_20261026}" \
  --calibration-path "$CALIBRATION_DIR/calibrations.json.gz" --model-dir "$CALIBRATION_DIR/models" \
  --grid "$GRID" --grid-index "$GRID_INDEX" --work-dir "$WORK_DIR" \
  --workers "${DP_WORKERS:-6}" --equality-workers "${EQUALITY_WORKERS:-12}"
"$REPO/.venv/bin/python" -m scripts.pack_right_anchor_results \
  --work-dir "$WORK_DIR" --output "$WORK_DIR/results.tar.gz"
echo "Ready to transfer: $WORK_DIR/results.tar.gz"
