#!/bin/bash
# Parallel sensor-validation gain sweep for DreamX-World closed-loop control.
# Fans one injection-gain arm out per GPU on a single node (submit gpus_per_node=8).
# Each arm is an independent validate_sensors.py run pinned to its own GPU.
set -u
REPO=/home/ma-user/work/dataset/VidGen_data_obs/wt_code/DreamX-World
DREAMX_PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd "$REPO" || exit 1

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

# Fine dose-response grid; one gain per GPU.
GAINS=(0.15 0.3 0.45 0.6 0.75 0.9 1.0)

pids=()
gpu=0
for G in "${GAINS[@]}"; do
  logf="/tmp/val_par_g${G}.log"
  echo ">>> launching gain ${G} on GPU ${gpu} (log ${logf})"
  CUDA_VISIBLE_DEVICES=${gpu} "$DREAMX_PY" validate_sensors.py \
    --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
    --model_name ./Wan2.2-TI2V-5B \
    --transformer_path ./configs/dreamx-ar/ \
    --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
    --data_path configs/dreamx/eval.json \
    --output_folder sensor_validation_par_g${G}/ \
    --num_output_frames 63 \
    --gold dino3 \
    --cheap latent_moments latent_mean latent_pooled4 \
    --noise_seeds 3 \
    --chunk_relative \
    --inject_gain ${G} \
    --max_items 12 > "${logf}" 2>&1 &
  pids+=($!)
  gpu=$((gpu+1))
done

echo "launched ${#pids[@]} arms: ${pids[*]}"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "arm gain=${GAINS[$i]} exit=0"
  else
    echo "arm gain=${GAINS[$i]} exit=FAIL"; fail=1
  fi
done
echo "ALL_PARALLEL_DONE fail=${fail}"
