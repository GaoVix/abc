#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

DATA_ROOT="${DATA_ROOT:-/mnt/afs/L202500703/abc_storage/dishrack_original_1261}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$REPO/cache/abc_dit_xl_200k_model.pt}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/mnt/afs/L202500703/abc_storage/output/dit_baseline_real_joint_noise_snr40_bs90_$(TZ=Asia/Taipei date +%Y%m%d_%H%M%S)}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=online

exec "$REPO/.venv/bin/torchrun" --standalone --nproc-per-node 4 train.py \
  --policy dit --load-pretrained \
  --cache-root "$DATA_ROOT" --output-dir "$TRAIN_OUTPUT_DIR" \
  --pretrained-ckpt-name "$PRETRAINED_CKPT" \
  --mixture-preset load_the_plates_into_the_dish_rack \
  --flow.state-noise-snr 40 \
  --batch-size 90 --num-workers 16 --train-steps 100000 \
  --log-every 5 --log-wandb --wandb-project abc-dishrack \
  --val-every 1000 --ckpt-every 30000
