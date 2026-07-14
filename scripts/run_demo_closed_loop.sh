#!/bin/bash
# Head-to-head open-loop vs closed-loop DreamX-World demo on ONE H800 GPU.
set -x
REPO=/home/ma-user/work/dataset/VidGen_data_obs/wt_code/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$DREAMX_PY" demo_closed_loop.py \
  --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
  --model_name ./Wan2.2-TI2V-5B \
  --transformer_path ./configs/dreamx-ar/ \
  --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
  --data_path configs/dreamx/eval.json \
  --output_folder demo_closed_loop_out/ \
  --num_output_frames 63 \
  --seed 42 \
  --chunk_relative \
  --modes open i pid \
  --kp 0.2 --ki 0.1 --leak 0.85 --gain_max 0.3 \
  --max_items 4
echo "DEMO_DONE exit=$?"
