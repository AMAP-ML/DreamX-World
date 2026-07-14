#!/bin/bash
cd /home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NS=8; pids=()
for gpu in $(seq 0 $((NS-1))); do
  CUDA_VISIBLE_DEVICES=$gpu $PY demo_closed_loop.py \
    --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
    --model_name ./Wan2.2-TI2V-5B --transformer_path ./configs/dreamx-ar/ \
    --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
    --data_path configs/dreamx/eval.json --output_folder demo_reroll_n8_s123/ \
    --num_output_frames 123 --seed 123 --chunk_relative --modes open reroll \
    --reroll_n 7 --reroll_thresh 0.2 \
    --drift_stride 2 --shard_id $gpu --num_shards $NS > /tmp/reroll_n8_s123_$gpu.log 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard $i ok" || fail=1; done
echo "ALL_n8_s123_DONE fail=$fail" > /home/ma-user/work/dataset/VidGen_data_obs/DreamX-World/.n8_s123_done
