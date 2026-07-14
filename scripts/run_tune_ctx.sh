#!/bin/bash
# Tune adaptive context_noise: sweep ctx_max on 3 items, open vs ctxnoise, one config per GPU.
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host=$(hostname)"

CTX_MAXES=(0.25 0.40 0.55)
pids=()
gpu=0
for cm in "${CTX_MAXES[@]}"; do
  CUDA_VISIBLE_DEVICES=${gpu} "$DREAMX_PY" demo_closed_loop.py \
    --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
    --model_name ./Wan2.2-TI2V-5B --transformer_path ./configs/dreamx-ar/ \
    --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
    --data_path configs/dreamx/eval.json \
    --output_folder tune_ctx_cm${cm}/ \
    --num_output_frames 123 --seed 42 --chunk_relative \
    --modes open ctxnoise \
    --ctx_base 0.1 --ctx_k 2.0 --ctx_max ${cm} --ctx_deadband 0.05 \
    --drift_stride 3 --max_items 3 \
    --shard_id 0 --num_shards 1 > /tmp/tune_cm${cm}.log 2>&1 &
  pids+=($!); gpu=$((gpu+1))
done
echo "launched ${#pids[@]} configs"
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "config ${CTX_MAXES[$i]} exit=0" || { echo "config ${CTX_MAXES[$i]} FAIL"; fail=1; }; done
echo "ALL_TUNE_DONE fail=${fail}"
