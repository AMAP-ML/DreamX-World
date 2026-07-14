#!/bin/bash
# Open-loop vs closed-loop demo, fanned across 8 GPUs on one node (shard by item).
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/wt_code/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host=$(hostname)"; nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

NSHARDS=8
pids=()
for gpu in $(seq 0 $((NSHARDS-1))); do
  CUDA_VISIBLE_DEVICES=${gpu} "$DREAMX_PY" demo_closed_loop.py \
    --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
    --model_name ./Wan2.2-TI2V-5B \
    --transformer_path ./configs/dreamx-ar/ \
    --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
    --data_path configs/dreamx/eval.json \
    --output_folder demo_closed_loop_out/ \
    --num_output_frames 63 --seed 42 --chunk_relative \
    --modes open i pid \
    --kp 0.2 --ki 0.1 --leak 0.85 --gain_max 0.3 \
    --shard_id ${gpu} --num_shards ${NSHARDS} > /tmp/demo_shard${gpu}.log 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shards: ${pids[*]}"
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard ${i} exit=0" || { echo "shard ${i} FAIL"; fail=1; }; done
echo "ALL_DEMO_DONE fail=${fail}"
