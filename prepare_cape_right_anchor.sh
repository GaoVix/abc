#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
STORAGE="${STORAGE:-/mnt/afs/L202500703/abc_storage}"
SOURCE_ROOT="${SOURCE_ROOT:-$STORAGE/dishrack_original_1261}"
RESULT_ROOT="${RESULT_ROOT:-$STORAGE/right_anchor_ik_results}"
DATA_ROOT="${DATA_ROOT:-$STORAGE/dishrack_right_anchor_left_ik_raw_sim_right_average}"
CALIBRATION_PATH="${CALIBRATION_PATH:-$STORAGE/cape_calibration/calibrations.json.gz}"
MODEL_DIR="${MODEL_DIR:-$STORAGE/cape_calibration/models}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

# Local IK results must use the same calibration as the physical RGB geometry.
"$REPO/.venv/bin/python" - "$RESULT_ROOT" "$CALIBRATION_PATH" <<'PY'
import hashlib, json, sys
from pathlib import Path
settings = json.loads((Path(sys.argv[1]) / "settings.json").read_text())
assert not settings["pilot"], "Use the complete IK result bundle, not the pilot"
assert settings["calibration_sha256"] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert settings["retarget_sides"] == ["left"]
PY

# No grid or IK solver is needed on the server. Keep the original video export.
"$REPO/.venv/bin/python" -m scripts.convert_right_anchor --export-only \
  --source-root "$SOURCE_ROOT" --work-dir "$RESULT_ROOT" --output "$DATA_ROOT" \
  --workers "${EXPORT_WORKERS:-6}"
"$REPO/.venv/bin/python" -m scripts.prepare_camera_parameters \
  "$DATA_ROOT/train_real" "$DATA_ROOT/val_real" "$DATA_ROOT/train_sim" "$DATA_ROOT/val_sim" \
  --calibration-path "$CALIBRATION_PATH" --model-dir "$MODEL_DIR" \
  --reference-frame right_arm --source-root "$SOURCE_ROOT" \
  --workers "${PREPARE_WORKERS:-8}"
"$REPO/.venv/bin/python" -m scripts.average_camera_pose \
  "$DATA_ROOT/train_real" "$DATA_ROOT/train_sim" \
  --real-frames 100000 --sim-frames 25000 --seed 0 \
  --reference-frame right_arm --camera-keys top left right \
  --output "$DATA_ROOT/camera_average.json"
echo "Prepared right-anchored IK dataset with three-camera right-base-average CaPE: $DATA_ROOT"
