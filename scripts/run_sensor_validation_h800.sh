#!/bin/bash
# Sensor validation sweep for DreamX-World closed-loop control.
# Runs on ONE H800 GPU (submit with num_nodes=1, gpus_per_node=1).
# Self-contained: uses the dreamx conda env by absolute path, no `conda activate`.
set -x
REPO=/home/ma-user/work/dataset/VidGen_data_obs/wt_code/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Slurm allocates the GPU and sets CUDA_VISIBLE_DEVICES; do not override.

echo "host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

for G in 0.3 0.6 1.0; do
  echo "================= INJECT GAIN ${G} ================="
  "$DREAMX_PY" validate_sensors.py \
    --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
    --model_name ./Wan2.2-TI2V-5B \
    --transformer_path ./configs/dreamx-ar/ \
    --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
    --data_path configs/dreamx/eval.json \
    --output_folder sensor_validation_h800_g${G}/ \
    --num_output_frames 63 \
    --gold dino3 \
    --cheap latent_moments latent_mean latent_pooled4 \
    --noise_seeds 3 \
    --chunk_relative \
    --inject_gain ${G} \
    --max_items 12
  echo "gain ${G} exit=$?"
done
echo "ALL_VALIDATION_DONE"
