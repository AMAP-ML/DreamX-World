#!/bin/bash
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO"; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NS=8; pids=()
for gpu in $(seq 0 7); do
( CUDA_VISIBLE_DEVICES=$gpu $PY window_eval.py --folder demo_goldctx_30s/ --modes open gold_ctx --stride 2 --lags 4 8 --shard_id $gpu --num_shards $NS
  CUDA_VISIBLE_DEVICES=$gpu $PY window_eval.py --folder demo_ffwd_30s/ --modes open ffwd --stride 2 --lags 4 8 --shard_id $gpu --num_shards $NS
  CUDA_VISIBLE_DEVICES="" $PY camera_track.py --folder demo_goldctx_30s/ --modes open gold_ctx --stride 8 --shard_id $gpu --num_shards $NS
  CUDA_VISIBLE_DEVICES="" $PY camera_track.py --folder demo_ffwd_30s/ --modes open ffwd --stride 8 --shard_id $gpu --num_shards $NS
) > /tmp/fgeval_$gpu.log 2>&1 &
pids+=($!); done
for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard $i ok" || echo "shard $i FAIL"; done
echo "ALL_FG_EVAL_DONE"
