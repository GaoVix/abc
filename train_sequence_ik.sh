#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/afs/L202500703/abc_storage/dishrack_median_sequence_ik}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
cd "$REPO"

"$REPO/.venv/bin/python" - "$DATA_ROOT/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert report['status'] == 'complete' and report['selected'] == 4514 and report['failed'] == 0, report
assert report['totals']['usable_chunks'] == 5181517, report['totals']
PY

if [[ -n "${RESUME_FROM:-}" ]]; then
  init_args=(--resume-from "$RESUME_FROM")
  default_output="$(dirname -- "$RESUME_FROM")"
else
  init_args=(--load-pretrained)
  default_output="/mnt/afs/L202500703/abc_storage/output/dit_median_sequence_ik_$(date +%Y%m%d_%H%M%S)"
fi

exec "$REPO/.venv/bin/torchrun" --standalone --nproc-per-node 4 train.py \
  "${init_args[@]}" \
  --cache-root "$DATA_ROOT" \
  --output-dir "${TRAIN_OUTPUT_DIR:-$default_output}" \
  --pretrained-ckpt-name "$REPO/cache/abc_dit_xl_200k_model.pt" \
  --mixture-preset load_the_plates_into_the_dish_rack \
  --batch-size 90 --num-workers 16 --train-steps 100000 \
  --log-every 5 --log-wandb --wandb-project abc-dishrack \
  --val-every 1000 --ckpt-every 30000
