#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
STORAGE="${STORAGE:-/mnt/afs/L202500703/abc_storage}"
DATA_ROOT="${DATA_ROOT:-$STORAGE/dishrack_right_anchor_left_ik_raw_sim_right_average}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$REPO/cache/abc_dit_xl_200k_model.pt}"
BATCH_SIZE="${BATCH_SIZE:-88}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$STORAGE/output/dit_cape_right_anchor_rightavg_noise40_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 WANDB_MODE=online

"$REPO/.venv/bin/python" - "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
from abc_minimal.camera_parameters import load_camera_average
root = Path(sys.argv[1])
anchor = json.loads((root / "right_anchor.json").read_text())
assert anchor["status"] == "complete" and not anchor["pilot"]
assert anchor["real_sequences"] == 1261 and anchor["sim_sequences"] == 3253
assert anchor["anchor_frame"] == "right_arm" and anchor["retarget_sides"] == ["left"]
assert anchor["sim_handling"] == "original"
summary = json.loads((root / "summary.json").read_text())
assert summary["status"] == "complete" and summary["failed"] == 0
config = json.loads((root / "config.json").read_text())
assert config["base_mode"] == "right_anchor"
assert config["filters"] == dict(position_tolerance_mm=.1, rotation_tolerance_deg=.1,
                                 jump_threshold_deg=30., chunk_length=30)
load_camera_average(root / "camera_average.json", camera_keys=("top", "left", "right"), reference_frame="right_arm")
for split in ("train_real", "val_real", "train_sim", "val_sim"):
    folders = sorted((root / split).glob("episode_*"))
    assert folders, f"Empty split: {split}"
    for folder in folders:
        camera = json.loads((folder / "camera_parameters.json").read_text())
        assert camera["reference_frame"] == "right_arm", folder
        assert camera["camera_keys"] == ["top", "left", "right"], folder
        if split.endswith("real"):
            assert camera["pose_source"] == "original_rgb_state", folder
print("Right-anchored IK: original right joints; all three cameras centered on the right-base mean; 40 dB noise")
PY

[[ -f "$PRETRAINED_CKPT" ]] || { echo "Checkpoint not found: $PRETRAINED_CKPT" >&2; exit 1; }
"$REPO/.venv/bin/wandb" login --host https://api.wandb.ai --verify
exec "$REPO/.venv/bin/torchrun" --standalone --nproc-per-node 4 train.py \
  --policy dit --load-pretrained --model.use-cape --model.cape-camera-keys top left right \
  --model.cape-reference-frame right_average \
  --cache-root "$DATA_ROOT" --output-dir "$TRAIN_OUTPUT_DIR" \
  --pretrained-ckpt-name "$PRETRAINED_CKPT" \
  --mixture-preset load_the_plates_into_the_dish_rack --flow.state-noise-snr 40 \
  --batch-size "$BATCH_SIZE" --num-workers 16 --train-steps 100000 \
  --log-every 5 --log-wandb --wandb-project abc-dishrack \
  --val-every 1000 --ckpt-every 30000
