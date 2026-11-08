#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/L202500703/abc
unset MUJOCO_GL

GPU_ID=0
seeds=(
    22373076	1815032804	796088752	1800435190	857321548
)

for seed in "${seeds[@]}"; do
    POLICY_SEED="$seed"
    CHECKPOINT="/mnt/afs/L202500703/abc_storage/output/dit_xml_three_stage_real_joint_noise_snr40_20261107_113412/100000.pt"
    EVAL_OUTPUT_DIR="/mnt/afs/L202500703/abc_storage/output/eval_xml_three_stage_noise_snr40_random6d_slsqp_pink_init_strict_100k_highres_pseed${POLICY_SEED}_$(date +%Y%m%d_%H%M%S)"

    mkdir -p "$EVAL_OUTPUT_DIR"

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    nohup .venv/bin/python eval_policy.py \
    --checkpoint "$CHECKPOINT" \
    --policy dit \
    --task dishrack \
    --gpu-id "$GPU_ID" \
    --num-worlds 100 \
    --parallel-worlds 0 \
    --camera-backend mjwarp \
    --camera-height 168 \
    --camera-width 224 \
    --no-fast-inference \
    --deterministic-inference \
    --seed 20260511 \
    --policy-seed "$POLICY_SEED" \
    --num-chunks 236 \
    --diffusion-steps 10 \
    --execute-chunk-dim 15 \
    --rtc \
    --rtc-prefix-length 4 \
    --rtc-inference-lead-steps 4 \
    --compensate-arm-bases \
    --base-compensation-ik slsqp \
    --base-compensation-reference xml \
    --randomization '{"plate_count":3,"plate_variant":"plate_0","dish_rack_variant":"dish_rack_0","randomize_variants":false,"randomize_scales":false,"randomize_arm_bases":true,"arm_base_source":"training_6d","arm_collision_mode":"eef_only"}' \
    --log-every-chunk \
    --output-dir "$EVAL_OUTPUT_DIR" \
    > "$EVAL_OUTPUT_DIR/eval.log" 2>&1 < /dev/null &

    EVAL_PID=$!
    printf '%s\n' "$EVAL_PID" > "$EVAL_OUTPUT_DIR/eval.pid"
    printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$EVAL_PID" "$EVAL_OUTPUT_DIR/eval.log" "$EVAL_OUTPUT_DIR"

done
