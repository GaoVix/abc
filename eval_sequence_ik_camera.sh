#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ARM_BASE_SOURCE=training_camera
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/mnt/afs/L202500703/abc_storage/output/eval_median_sequence_ik_pink_60k_left_camera_right_planar_$(date +%Y%m%d_%H%M%S)}"
exec bash "$REPO/eval_sequence_ik.sh" "$@"
