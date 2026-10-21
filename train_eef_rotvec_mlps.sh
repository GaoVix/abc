#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Same pretraining setup as the rotation6D adapters; only the EEF representation changes.
uv run python -u -m scripts.train_eef_adapters \
  --pose-norm-stats cache/eef_rotvec_norm_stats.json \
  --dit-checkpoint "$PWD/cache/abc_dit_xl_200k_model.pt" \
  --output outputs/eef_adapters_rotvec_fixed_stats_350k \
  --steps 350000 \
  --batch-size 40960 \
  --wandb \
  "$@"

uv run python -u -m scripts.compare_eef_adapters \
  --checkpoints \
    outputs/eef_adapters_fixed_stats_350k/best.pt \
    outputs/eef_adapters_rotvec_fixed_stats_350k/best.pt \
    outputs/eef_adapters_rotvec_fixed_stats_350k/last.pt \
  --output outputs/eef_adapters_rotvec_fixed_stats_350k/comparison.json
