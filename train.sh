








cd "$(dirname "$0")" || exit 1

# uv run train.py \
#   --load-pretrained \
#   --cache-root cache/dishrack_benchmark \
#   --pretrained-ckpt-name /home/ps/Documents/projects/abc/cache/abc_dit_xl_200k_model.pt \
#   --mixture-preset load_the_plates_into_the_dish_rack \
#   --batch-size 41 \
#   --num-workers 2 \
#   --no-compile \
#   --train-steps 10000 \
#   --log-every 5 \
#   --val-every 10000 \
#   --ckpt-every 10000


uv run torchrun --standalone --nproc-per-node 4 train.py \
  --load-pretrained \
  --eef-adapters output/eef_adapters_fixed_stats_350k/best.pt \
  --cache-root cache/dishrack_real \
  --pretrained-ckpt-name "$PWD/cache/abc_dit_xl_200k_model.pt" \
  --mixture-preset load_the_plates_into_the_dish_rack \
  --batch-size 90 \
  --num-workers 16 \
  --train-steps 10000 \
  --log-every 5 \
  --val-every 10000 \
  --ckpt-every 10000




# cd /home/ps/Documents/projects/abc

# uv run train.py \
#   --load-pretrained \
#   --cache-root cache/dishrack_real \
#   --pretrained-ckpt-name /home/ps/Documents/projects/abc/cache/abc_dit_xl_200k_model.pt \
#   --mixture-preset "load_the_plates_into_the_dish_rack" \
#   --batch-size 1 \
#   --num-workers 4 \
#   --no-compile \
#   --train-steps 100 \
#   --log-every 5 \
#   --val-every 1000 \
#   --ckpt-every 1000
