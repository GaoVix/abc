# ABC Training (`abc_minimal`)

**Contents**

- [Training data](#training-data)
  - [Exporting a task from ABC-130k](#exporting-a-task-from-abc-130k)
  - [Converting local MCAPs & the episode format](#converting-local-mcaps--the-episode-format)
- [Multi-node training](#multi-node-training)
- [Plücker camera conditioning](#plücker-camera-conditioning)
- [Subtask & operator conditioning](#subtask--operator-conditioning)
- [Visualizing episodes and policies](#visualizing-episodes-and-policies)


`abc_minimal` is the training and inference package for ABC-DiT: the model and
its DINOv3/CLIP backbones (`dit.py`), the episode dataloader (`dataloader.py`,
`episode_io.py`, `preprocess.py`), the training loop and checkpointing
(`train_loop.py`, `checkpointing.py`), DiT and VLA policy inference
(`policy.py`), the sim-eval glue (`sim_env.py`, `eval_policy.py`), and
the episode/policy visualizers. The scripts at the repository root —
`train.py`, `eval_policy.py`, `viz_episode.py`, `viz_policy.py` — are thin
wrappers around this package.

Setup, the data download, single-node training, and the evaluation quickstart
are covered in the [top-level README](../README.md). Simulator environments and
the sim-eval task catalogue are covered in the
[abc_sim README](../abc_sim/README.md). This document is the training-side
reference: multi-node jobs, the episode data format and converters, and prompt
conditioning options.

## Training data

While we host a single task in training format, there are many more in the ABC
Dataset. The ABC-130k MCAPs are hosted on Hugging Face at
[`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k). The dataset
is gated, so accept access on the dataset page and set `HF_TOKEN` before
downloading.

### Exporting a task from ABC-130k

Download all MCAPs for one task and convert them in place:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles
```

By default this downloads both `train` and `val`, stages raw MCAPs under
`$ABC_CACHE/hf_tasks/<task>/`, runs the MCAP converter, writes converted episodes
to `$ABC_CACHE/train_real/` and `$ABC_CACHE/val_real/`, then deletes the staged
raw MCAPs after each successful split conversion. For a quick smoke test:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles --split train --max-episodes 1
```

### Converting local MCAPs & the episode format

If you already have local MCAPs, call the lower-level converter directly:

```bash
uv run scripts/export_mcap.py ./train_run_1 ./out
```

The input is expected to look like:

```text
train_run_1/
  <task_name>/
    episode_<uuid>/
      episode.mcap
```

You can also pass the number of worker processes:

```bash
uv run scripts/export_mcap.py ./train_run_1 ./out 8
```

Each output episode is written to `./out/episode_<uuid>/` in the same format
the trainer reads:

```text
episode_<uuid>/
  states_actions.bin               # (num_steps, 28) float64: 14 states + 14 actions
  combined_camera-images-rgb.mp4   # 30 fps vertical stack of 224x224 camera views
  episode_metadata.json            # task name, cameras, resolutions, timing, num_steps
```

The mp4 is encoded in a manner that allows for efficient dataloading. For details, see the ABC paper.

## Plücker camera conditioning

Enable optional DiT CaPE with `--model.use-cape`. The Plücker ray calculation
follows the [CamPose authors' snippet](https://github.com/ripl/CamPoseOpensource),
and the five-layer geometry CNN adapts their ACT `BackboneLateConcat`.
Its final convolution uses stride 1 instead of 2 and outputs `vit_embed_dim`
channels (768 by default), matching both ABC's ViT-B/16 grid and feature width:
224x224 rays produce a 768x14x14 feature map directly, with total stride 16.
No feature-grid resizing is used; mismatched grids raise an error.
RGB still goes through the original DINOv3/CLIP encoder. Geometry features
are concatenated with the corresponding patch features and projected from
`2 * vit_embed_dim` to `vit_embed_dim` (1536 to 768 by default) before the
existing attention pool.
The CLS token, camera ID embeddings, and DiT architecture are retained.
`FrozenBatchNorm2d` is copied from the ACT implementation, including its
configurable epsilon and `num_batches_tracked` checkpoint compatibility.
`get_plucker_raymap` is copied from the authors' official OpenCV README snippet.
The outer `plucker_raymap` adapter only handles batch dimensions and ACT's
`[m,d]` CHW layout. The RoboSuite embedder instead has
dataset-specific camera axes, pixel offsets and a hardcoded 256x256 grid.

`cape.py` contains rays and late fusion; `camera_geometry.py` contains offline
calibration/FK, and `camera_parameters.py` loads its per-frame output during
training. The fusion projection starts at
`[I, 0]` with zero bias, preserving the parent policy's initial function.
Training can initialize the new branch from an RGB-only parent; inference
and resume require all CaPE weights. Defaults leave existing runs unchanged.

`train_cape.sh` defaults to batch size 64 per GPU to leave memory for the
additional CaPE branch. `BATCH_SIZE` overrides it. Set `SKIP_CAMERA_PREPARE=1`
when relaunching after successful camera preprocessing, so the existing camera
files are reused.

`python -m scripts.probe_cape_batch --cache-root DATA --pretrained-ckpt-name CKPT
--output-dir LOGS` probes the per-GPU batch limit with four fresh DDP workers
per trial. It reuses the normal compiled training loop and existing camera
files, runs five complete updates, and skips validation, W&B and checkpoint
writes. The default search starts at 64 and searches up to 89 (90 already
failed on the current four-H100 setup). Per-rank memory peaks and trial logs
are saved with `summary.json`. A short-run limit is not a guarantee for a long
run; leave headroom below that limit.

Before training, run offline preprocessing on the **final processed state
directories**. Supply the **complete three-camera** `calibrations.json.gz` to
`scripts.prepare_camera_parameters` with `--calibration-path`; the smaller
`scripts/dishrack_joint_retarget_calibrations.json.gz` contains only the top
camera and is not the correct input. For original, unretargeted states,
`--model-dir` points to the original calibration robot XMLs; mesh assets are
not needed. FK copies only the arm joint/link hierarchy, its original joint/camera
defaults and the three camera mounts, then compiles that small tree with
MuJoCo. It discards task objects, render geometry, assets, lights and actuators.
Original robot variants retain their own link parameters.
For `sequence_ik`/`joint_conversion` episodes, FK uses the target YAM model;
sequence-IK metadata supplies the target bases when available. Real camera
geometry is expressed in the simulation world by anchoring the calibrated
top camera to the target robot configuration's top camera.

Only **fixed** camera-to-link/base transforms, intrinsics and distortion
parameters are read from the calibration bundle. Offline preprocessing computes each wrist pose as
`T_world_from_link(state) @ T_link_from_camera` from the exact row in the final
`states_actions.bin`, before normalization, noise, or masking. It writes three
files **in that same episode directory**, without creating a separate output
root or timestamped directory:

```text
episode_.../
  states_actions.bin
  camera_parameters.json   # v4: K, D, distortion model, pixel centers, image sizes, camera order
  camera_parameters.npy    # per row: source state (14 float64) + poses (3x4x4 float32)
  camera_directions.npy    # fixed camera-space directions (3 cameras, 3, 224, 224), float32
```

The per-frame payload is 304 bytes for three cameras, including a copy of the
source state. The NPY is memory-mapped; workers read just the sampled frame
and keep a bounded cache of 16 open episodes. Training checks the saved state
against that frame's 14 raw state values (112 bytes), so changed alignment/conversion cannot silently
reuse old poses. Regenerate with `--overwrite` after changing states,
calibration or bases. The fixed direction table is about 1.72 MiB for three
224x224 cameras, independent of episode length. Missing files raise an error. Old v1/v2/v3
sidecars must be regenerated with `--overwrite` before training.

The expensive pixel-to-ray inversion runs **offline**, using K, the recorded
`distortion_model`, and five coefficients `D=[k1,k2,p1,p2,k3]`. Brown-Conrady
(including ROS `plumb_bob`) and Inverse Brown-Conrady use the corresponding
librealsense fixed-point updates; unknown models or invalid/nonconvergent
calibrations fail explicitly. K and directions use ABC's existing half-pixel
convention (`pixel_center=0.5`), including the +0.5 conversion of real cx/cy
when loading calibration. Native SDK K can be used by setting `pixel_center=0`
explicitly. RGB is not undistorted or rewritten.

Directions are resized with bicubic/antialiasing and letterboxed to the
**exported, unaugmented RGB grid** offline; black padding has zero directions.
This uses the same interpolation family as FFmpeg's bicubic RGB export;
different libraries need not produce bit-identical samples. The grid comes from
`episode_metadata.json:camera_resolutions` (`[width,height]`), then its
`image_width/image_height`, with 224x224 as the legacy export-grid default.
This grid is separate from the original camera size. Real K/D are evaluated
at the original size recorded in each camera's calibration. MCAP export now
saves each actual probed `source_camera_resolutions` and selected camera topic;
when that source size is present, preprocessing checks it against calibration.
Within each preprocessing worker, identical K/D/model/grid combinations reuse
one direction table. Each episode stores its own table for portability.
Training workers also share tables across episodes with identical calibration,
with a bounded cache of 32 distinct tables, avoiding repeated disk reads when
random sampling switches between episodes.

At training time workers load that small table once and, for each sampled
frame, only rotate directions by its saved camera rotation and compute
`moment = camera_position cross world_direction`. Moving wrist cameras do
not require solving distortion again. The resulting six channels still
undergo the same random spatial augmentation as RGB.

CaPE does not read temporal
calibration files, select states by timestamps, interpolate, or apply time
offsets; data preprocessing owns state/image alignment. Camera directions
account for lens distortion before the exporter's resize/letterbox. Real
calibration cx/cy are shifted by +0.5 on loading to match the snippet's pixel
centers; sim K is unchanged. This preserves v2's unified pixel convention.
RGB and rays share spatial augmentation;
photometric augmentation and normalization apply to RGB only.

Sim preprocessing uses the fixed ABC arm/camera configuration from the
bundled `yam_dishrack_base.xml`, extracting only its arm/camera tree as above.
Episode scene XMLs and meshes are not read. The 3,253 original dishrack sim
episodes were checked to have the same arm hierarchy, nominal bases and three
camera mounts/FOVs. Processed `sequence_ik.camera_from_base`, when present,
supplies the processed bases; otherwise the fixed nominal bases are used.
Current state supplies the joint angles for FK.
Sim episodes are identified by `render_backend` or the `sim_`/`sim ` task-name
prefix; real episodes without an accepted calibration fail explicitly.
Each camera uses its original `[width,height]` from
`source_camera_resolutions`. Published sim episodes without that field use
their recorded `image_width/image_height` (the local examples are 224x168).
Missing source sizes require explicit metadata rather than assuming 640x480
or 224x224. Ray tables match the exported RGB grid; the model's final 224x224
resize/padding is applied jointly to RGB and rays during preprocessing.
These simulated cameras explicitly use `distortion_model="none"` and five
zero distortion coefficients; their direction tables use pinhole projection.
Real cameras each retain their own recorded model and D. Retargeting joints
does not change the RGB camera model: sequences that retain real RGB still
use the corresponding real-camera calibration.
Missing real calibration or required robot definitions raises an error during preprocessing.
Training only constructs world-coordinate rays from the saved direction table
and per-frame poses, then applies the shared RGB/ray spatial augmentation.
Per-frame full ray maps are not stored on disk.

```bash
# Point these arguments at the intended processed-state directories on the
# server (including the desired no-offset version). Outputs stay beside them.
uv run python -m scripts.prepare_camera_parameters \
  /path/to/processed_states/train_real \
  /path/to/processed_states/val_real \
  --calibration-path /path/to/calibrations.json.gz \
  --model-dir /path/to/original/calibration/models \
  --workers 8 --overwrite

uv run python train.py \
  --load-pretrained \
  --pretrained-ckpt-name "$PWD/cache/abc_dit_xl_200k_model.pt" \
  --model.use-cape \
  --cache-root /path/to/processed_states \
  --mixture-preset load_the_plates_into_the_dish_rack \
  --output-dir output/dit_cape
```

Evaluate CaPE-trained weights with `--model.use-cape --parallel-worlds 0`.
The sequential simulator and viewer already run FK for their actual state;
they provide current camera poses and intrinsics. Sim inference renders at
1024x768 by default and uses Pillow's bicubic resize to 224x168. Its supplied
K describes that resized 224x168 RGB, then RGB and rays receive 28 zero rows
above and below. The high-resolution render's K is not passed to the policy.
Other inference callers
supply `camera_intrinsics` and `camera_to_world` per camera in the observation.
For distorted RGB they additionally supply `camera_distortion[cam]` with
`{"distortion_model": "inverse_brown_conrady", "D": [k1,k2,p1,p2,k3]}`,
and optionally `pixel_center`. K and distortion must describe the supplied
RGB grid. If RGB already includes an export resize/letterbox while K remains
at its original calibration size, also supply that original `height` and
`width` in `camera_distortion[cam]`. Inference first matches the supplied RGB
grid with bicubic, then applies the model's shared bilinear resize/padding.
Inference caches directions by calibration/grid/device, and only
updates the current camera pose on subsequent observations. Omitting
`camera_distortion` retains the existing pinhole inference behavior.
Batched MJWarp eval is currently rejected for CaPE. Fast warmup requires a
real observation with calibration. Robot/websocket acquisition must also
provide its current calibrated camera poses before deploying a CaPE model.

```bash
OMP_NUM_THREADS=1 uv run python -m unittest discover -s tests -p 'test_cape.py' -v
```

## Multi-node training

For multi-node jobs without a shared filesystem, predownload a deterministic
node-local shard on each node before launching training:

```bash
ABC_CACHE=/local_nvme/abc_cache HF_TOKEN=... \
uv run scripts/prepare_hf_shards.py \
  --tasks organize_the_condiment_bottles \
  --num-nodes 8 --node-rank $NODE_RANK --workers 8

ABC_CACHE=/local_nvme/abc_cache \
uv run torchrun --nnodes 8 --node-rank $NODE_RANK --nproc-per-node 8 train.py
```

The predownload step writes this node's converted episodes into the usual
`train_real/` and `val_real/` directories. Training auto-detects the
`hf_status/shard.json` marker and uses local-rank sampling, so validation
metrics are accumulated across different validation shards on different GPUs.

The revision is pinned to a commit SHA at run time and training verifies at
startup that all nodes sharded the same snapshot. If the dataset may change
while nodes prepare, pass the same explicit `--revision <sha>` to every node.

## Subtask & operator conditioning

In addition to the task prompt, our policies can condition  **subtask** labels and on
the episode's **operator** id. The MCAP converter (`scripts/export_mcap.py`)
extracts both from the release MCAPs when present and writes two optional
extra files next to the episode:

```text
episode_<uuid>/
  subtasks.json      # {"<frame_idx>": "<subtask label>", ...}  — per-frame subtask
  operator.json      # {"operator_id": "<uuid>"}                — the teleoperator id
```

- **Subtasks** Enable at train time with `--prompt.use-subtask-as-prompt`,
  choosing `--prompt.subtask-mode {replace,append}` (`replace` swaps the task prompt for the subtask label;
  `append` formats both via `--prompt.subtask-append-format`).
- **Operators** We map UUIDs to short deterministic labels (`operator 0`, … or names). The labels are computed
  per-task such that operators with more hours (proxy for quality) have lower numbers.
  Enable with `--prompt.use-operator-id-as-prompt` and choose `--prompt.operator-prompting-mode {text_indexed,text_name}`; the label is appended as `"{prompt}. {operator}"`.

  The per-task map must be built prior to training and passed via
  `--prompt.operator-label-map-path`. Build it with:

  ```bash
  ABC_CACHE=cache/tshirt uv run scripts/build_operator_label_map.py \
      --out cache/tshirt/operator_label_map.json
  ```

  then pass `--prompt.operator-label-map-path cache/tshirt/operator_label_map.json`.

  On a node-sharded multi-node cache no single machine holds every episode, and
  a map built from one shard would mis-rank operators and miss those on other
  nodes. Instead, build one manifest per node from its local shard, gather the
  shard manifests on one machine, and fold them into a global ranking —
  hours and episode counts sum exactly across shards, so the result matches a
  full single-machine scan:

  ```bash
  uv run scripts/build_operator_label_map.py --out shard_$NODE.json   # on each node
  uv run scripts/build_operator_label_map.py \
      --combine shard_0.json shard_1.json ... --out operator_label_map.json
  ```

  then copy the combined manifest to every node at the same path and pass it
  via `--prompt.operator-label-map-path`.

Both are off by default. Some episodes do not have eg. subtask annotations and
for these training will drop back to task prompt only.

(We intend to release the global manifest in future but this is TODO.)

## Visualizing episodes and policies

`viz_policy.py` rolls out a checkpoint live in a viser window, and
`viz_episode.py` plays back dataset episodes the same way, no checkpoint
needed — browse a pool (`--root cache/train_sim`) or open one episode
directly; `--mode physics` re-simulates the recorded actions instead of posing
the arms. The default pose mode replays the whole recorded scene, objects
included, for episodes that ship `scene_qpos.npy` (the current sim_224
release), and falls back to posing the arms alone — objects held at their
start pose — for episodes without it:

```bash
uv run viz_policy.py --sim.checkpoint cache/bottles_75k.pt --port 8080
uv run viz_episode.py --root cache/train_sim --port 8080
```
