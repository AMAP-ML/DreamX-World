#!/bin/bash
cd /home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NS=4; pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu $PY demo_closed_loop.py \
    --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
    --model_name ./Wan2.2-TI2V-5B --transformer_path ./configs/dreamx-ar/ \
    --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
    --data_path configs/dreamx/eval.json --output_folder demo_smoothval/ \
    --num_output_frames 63 --seed 42 --chunk_relative --modes open smooth \
    --smooth_beta 0.7 --smooth_gain 0.3 \
    --drift_stride 3 --max_items 8 --shard_id $gpu --num_shards $NS > /tmp/smoothval_$gpu.log 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard $i ok" || fail=1; done
echo "ALL_SMOOTHVAL_DONE fail=$fail" > /home/ma-user/work/dataset/VidGen_data_obs/DreamX-World/.smoothval_done
