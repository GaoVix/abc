#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

for task in \
  throw_plastic_bottles_in_bin \
  pouring \
  inhand_transfer_item_to_other_side \
  set_up_chess_pieces_on_board \
  sort_lego_blocks \
  sort_nuts_and_bolts \
  put_relative \
  grab_specific_object_from_clutter \
  conveyor_pick \
  count_into_opaque_box \
  ball_tray_balancing \
  multi_drawer_search \
  spell_abc \
  spell_agi \
  spell_cat \
  spell_dog \
  spell_fish \
  spell_xdof \
  spell_yam
do
  chunks=236
  # Official LEGO evaluation uses 400 chunks.
  if [ "$task" = sort_lego_blocks ]; then
    chunks=400
  fi
  echo "Evaluating $task ($chunks chunks)"
  uv run eval_policy.py \
    --checkpoint "${ABC_CACHE:-cache}/abc_dit_xl_200k_model.pt" \
    --task "$task" \
    --num-worlds 10 \
    --num-chunks "$chunks" \
    --seed 20260511 \
    --policy-seed 0 \
    --save-video \
    --video-every-n-actions 1 \
    --video-fps 30 \
    --log-every-chunk \
    --output-dir "outputs/dit_200k/$task"
done
