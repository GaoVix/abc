#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

: "${DATA_ROOT:?Set DATA_ROOT to the final processed-state dataset directory}"
: "${CALIBRATION_PATH:?Set CALIBRATION_PATH to the complete three-camera calibrations.json.gz}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$REPO/cache/abc_dit_xl_200k_model.pt}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$REPO/output/dit_cape_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$PRETRAINED_CKPT" ]] || { echo "Checkpoint not found: $PRETRAINED_CKPT" >&2; exit 1; }
PRETRAINED_CKPT="$(realpath -- "$PRETRAINED_CKPT")"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
camera_args=(--calibration-path "$CALIBRATION_PATH")
if [[ -n "${MODEL_DIR:-}" ]]; then
  camera_args+=(--model-dir "$MODEL_DIR")
fi

if [[ "${SKIP_CAMERA_PREPARE:-0}" != 1 ]]; then
  "$REPO/.venv/bin/python" -m scripts.prepare_camera_parameters \
    "$DATA_ROOT/train_real" "$DATA_ROOT/val_real" \
    "$DATA_ROOT/train_sim" "$DATA_ROOT/val_sim" \
    "${camera_args[@]}" --workers "${PREPARE_WORKERS:-8}" --overwrite
fi

exec "$REPO/.venv/bin/torchrun" --standalone --nproc-per-node "${NPROC_PER_NODE:-4}" train.py \
  --load-pretrained --model.use-cape \
  --cache-root "$DATA_ROOT" --output-dir "$TRAIN_OUTPUT_DIR" \
  --pretrained-ckpt-name "$PRETRAINED_CKPT" \
  --mixture-preset load_the_plates_into_the_dish_rack \
  --batch-size "${BATCH_SIZE:-64}" --num-workers "${NUM_WORKERS:-16}" \
  --train-steps "${TRAIN_STEPS:-100000}" \
  --log-every 5 --log-wandb --wandb-project abc-dishrack \
  --val-every 1000 --ckpt-every 10000
