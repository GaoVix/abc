#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

STORAGE="${STORAGE:-/mnt/afs/L202500703/abc_storage}"
DATA_ROOT="${DATA_ROOT:-$STORAGE/dishrack_xml_default_real_ik_raw_sim}"
SOURCE_ROOT="${SOURCE_ROOT:-$STORAGE/dishrack_original_1261}"
CALIBRATION_PATH="${CALIBRATION_PATH:-$STORAGE/cape_calibration/calibrations.json.gz}"
MODEL_DIR="${MODEL_DIR:-$STORAGE/cape_calibration/models}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$REPO/cache/abc_dit_xl_200k_model.pt}"
BATCH_SIZE="${BATCH_SIZE:-88}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$STORAGE/output/dit_three_stage_filtered_cape_wrists_topavg_noise40_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=online

# Match the filtered three-stage XML-base dataset used by the old retarget run.
"$REPO/.venv/bin/python" - "$DATA_ROOT" "$SOURCE_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
config = json.loads((root / "config.json").read_text())
assert config["base_mode"] == "xml", config
assert config["filters"] == dict(position_tolerance_mm=.1, rotation_tolerance_deg=.1,
                                 jump_threshold_deg=30., chunk_length=30), config
summary = json.loads((root / "summary.json").read_text())
assert summary["status"] == "complete" and summary["failed"] == 0, summary
assert summary["base_mode"] == "xml" and summary["sim_handling"] == "original", summary
assert summary["totals"]["usable_chunks"] == 5488467, summary["totals"]
for split in ("train_real", "val_real"):
    path = next(iter(sorted((root / split).glob("episode_*/episode_metadata.json"))), None)
    if path is None:
        raise SystemExit(f"No converted episodes in {root / split}")
    metadata = json.loads(path.read_text())
    ik = metadata["sequence_ik"]
    assert ik["base_mode"] == "xml", path
    assert ik["state_solver"] == ik["action_solver"] == "trust-constr", path
    assert "source_frame_range" in metadata, path
    source = source_root / split / ik["source_episode"] / "states_actions.bin"
    if not source.is_file():
        raise SystemExit(f"Original RGB-aligned states missing (needed for camera FK): {source}")
    result = Path(ik["state_result"])
    if not result.is_absolute():
        result = path.parent / result
    if not result.is_file():
        raise SystemExit(f"Full-sequence IK state result missing (needed for noise): {result}")
print(f"Filtered three-stage XML IK: {root}")
print("RGB: top/left/right; CaPE: left/right; real joint-state noise: 40 dB")
PY

if [[ "${SKIP_CAMERA_PREPARE:-0}" != 1 ]]; then
  # Regenerate beside the converted states. The original RGB videos are untouched.
  "$REPO/.venv/bin/python" -m scripts.prepare_camera_parameters \
    "$DATA_ROOT/train_real" "$DATA_ROOT/val_real" \
    "$DATA_ROOT/train_sim" "$DATA_ROOT/val_sim" \
    --calibration-path "$CALIBRATION_PATH" --model-dir "$MODEL_DIR" \
    --reference-frame top --source-root "$SOURCE_ROOT" \
    --workers "${PREPARE_WORKERS:-8}" --overwrite

  # One shared reference, calculated from training frames only and reused by every split.
  "$REPO/.venv/bin/python" -m scripts.average_camera_pose \
    "$DATA_ROOT/train_real" "$DATA_ROOT/train_sim" \
    --real-frames 100000 --sim-frames 25000 --seed 0 \
    --output "$DATA_ROOT/camera_average.json"
else
  echo "Reusing prepared camera parameters and camera_average.json"
fi

if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

[[ -f "$PRETRAINED_CKPT" ]] || { echo "Checkpoint not found: $PRETRAINED_CKPT" >&2; exit 1; }
"$REPO/.venv/bin/wandb" login --host https://api.wandb.ai --verify

# Check this exact wrist-only/noisy setup once before launching the full run.
if [[ "${PROBE_BATCH_SIZE:-1}" == 1 ]]; then
  "$REPO/.venv/bin/python" -m scripts.probe_cape_batch \
    --cache-root "$DATA_ROOT" --pretrained-ckpt-name "$PRETRAINED_CKPT" \
    --output-dir "$TRAIN_OUTPUT_DIR/batch_probe" --num-gpus 4 --num-workers 16 \
    --start-batch "$BATCH_SIZE" --max-batch "$BATCH_SIZE" --single-trial --steps 5 \
    --cape-camera-keys left right --cape-reference-frame top_average --state-noise-snr 40
fi

exec "$REPO/.venv/bin/torchrun" --standalone --nproc-per-node 4 train.py \
  --policy dit --load-pretrained --model.use-cape --model.cape-camera-keys left right \
  --model.cape-reference-frame top_average \
  --cache-root "$DATA_ROOT" --output-dir "$TRAIN_OUTPUT_DIR" \
  --pretrained-ckpt-name "$PRETRAINED_CKPT" \
  --mixture-preset load_the_plates_into_the_dish_rack --flow.state-noise-snr 40 \
  --batch-size "$BATCH_SIZE" --num-workers 16 --train-steps 100000 \
  --log-every 5 --log-wandb --wandb-project abc-dishrack \
  --val-every 1000 --ckpt-every 30000
