#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Server experiment using the separately pretrained 14D EEF adapters.
# Copy the local outputs/eef_adapters_rotvec_fixed_stats_350k/best.pt to this path first.
test -f output/eef_adapters_rotvec_fixed_stats_350k/best.pt

uv run torchrun --standalone --nproc-per-node 4 train.py \
  --load-pretrained \
  --eef-adapters output/eef_adapters_rotvec_fixed_stats_350k/best.pt \
  --cache-root cache/dishrack_real \
  --output-dir output/dit_eef_rotvec_pretrained_flow_20261021 \
  --pretrained-ckpt-name "$PWD/cache/abc_dit_xl_200k_model.pt" \
  --mixture-preset load_the_plates_into_the_dish_rack \
  --batch-size 90 \
  --num-workers 16 \
  --train-steps 100000 \
  --log-every 5 \
  --log-wandb \
  --wandb-project abc-dishrack \
  --val-every 1000 \
  --ckpt-every 10000
