#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
unset MUJOCO_GL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

# Pass a 60k checkpoint as the first argument (or CHECKPOINT). Otherwise choose
# exactly 60000.pt in the latest default median-sequence training run.
CHECKPOINT="${CHECKPOINT:-${1:-}}"
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(.venv/bin/python - <<'PY'
from pathlib import Path

roots = [Path('/mnt/afs/L202500703/abc_storage/output'), Path('output')]
runs = {path.resolve() for root in roots
        for path in root.glob('dit_median_sequence_ik_*') if path.is_dir()}
if not runs:
    raise SystemExit('No median-sequence run found. Pass its checkpoint as the first argument.')
run = max(runs, key=lambda path: path.name)
checkpoint = run / '60000.pt'
if not checkpoint.is_file():
    raise SystemExit(f'Missing {checkpoint}. Pass a completed 60k checkpoint explicitly.')
print(checkpoint)
PY
)"
fi

# Verify the selected file before creating a policy or allocating GPU memory.
.venv/bin/python - "$CHECKPOINT" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
if checkpoint.get('global_step') != 60000:
    raise SystemExit(f'Expected step 60000; got {checkpoint.get("global_step")!r}')
cache_root = checkpoint.get('train_config', {}).get('cache_root', '')
if 'median_sequence_ik' not in str(cache_root):
    raise SystemExit(f'Expected a median-sequence IK checkpoint; cache_root={cache_root!r}')
print(f'Checkpoint: {path}', flush=True)
print(f'Step: {checkpoint.get("global_step")}; data: {cache_root}', flush=True)
PY

exec "$REPO/.venv/bin/python" eval_policy.py \
  --checkpoint "$CHECKPOINT" \
  --policy dit --task dishrack \
  --gpu-id "${GPU_ID:-0}" \
  --num-worlds 100 --parallel-worlds 100 \
  --seed 20260511 --policy-seed 0 \
  --num-chunks 236 --diffusion-steps 10 --execute-chunk-dim 15 \
  --rtc --rtc-prefix-length 4 --rtc-inference-lead-steps 4 \
  --compensate-arm-bases \
  --base-compensation-ik pink \
  --base-compensation-reference median \
  --randomization '{"plate_count":3,"plate_variant":"plate_0","dish_rack_variant":"dish_rack_0","randomize_variants":false,"randomize_scales":false,"randomize_arm_bases":true}' \
  --log-every-chunk \
  --output-dir "${EVAL_OUTPUT_DIR:-/mnt/afs/L202500703/abc_storage/output/eval_median_sequence_ik_pink_60k_$(date +%Y%m%d_%H%M%S)}"
