#!/bin/bash
# Windowed/local DINOv3 drift re-eval on the 30s videos, fanned across 8 GPUs.
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host=$(hostname)"
NSHARDS=8
pids=()
for gpu in $(seq 0 $((NSHARDS-1))); do
  CUDA_VISIBLE_DEVICES=${gpu} "$DREAMX_PY" window_eval.py \
    --folder demo_closed_loop_30s/ --modes open pid pid2 \
    --stride 2 --lags 4 8 \
    --shard_id ${gpu} --num_shards ${NSHARDS} > /tmp/win_shard${gpu}.log 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shards"
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard ${i} exit=0" || { echo "shard ${i} FAIL"; fail=1; }; done
echo "ALL_WINDOW_DONE fail=${fail}"
