#!/bin/bash
# 4-axis eval on the ctxnoise run: windowed drift + camera tracking, 8-GPU sharded.
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NSHARDS=8; pids=()
for gpu in $(seq 0 $((NSHARDS-1))); do
  ( CUDA_VISIBLE_DEVICES=${gpu} "$DREAMX_PY" window_eval.py --folder demo_shift_30s/ \
      --modes open shift --stride 2 --lags 4 8 --shard_id ${gpu} --num_shards ${NSHARDS}
    CUDA_VISIBLE_DEVICES="" "$DREAMX_PY" camera_track.py --folder demo_shift_30s/ \
      --modes open shift --stride 8 --shard_id ${gpu} --num_shards ${NSHARDS} \
  ) > /tmp/ctxeval_${gpu}.log 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard ${i} exit=0" || { echo "shard ${i} FAIL"; fail=1; }; done
echo "ALL_CTXEVAL_DONE fail=${fail}"
