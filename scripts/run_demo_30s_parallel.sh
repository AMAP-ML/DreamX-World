#!/bin/bash
# 30s open vs closed-loop (v1 pid, v2 pid2) demo, fanned across 8 GPUs on one node.
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
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
    --output_folder demo_closed_loop_30s/ \
    --num_output_frames 123 --seed 42 --chunk_relative \
    --modes open pid pid2 \
    --kp 0.2 --ki 0.1 --leak 0.85 --gain_max 0.3 \
    --kp2 0.05 --ki2 0.12 --leak2 0.9 --deadband2 0.10 \
    --drift_stride 2 \
    --shard_id ${gpu} --num_shards ${NSHARDS} > /tmp/demo30_shard${gpu}.log 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} shards: ${pids[*]}"
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard ${i} exit=0" || { echo "shard ${i} FAIL"; fail=1; }; done
echo "ALL_DEMO30_DONE fail=${fail}"
