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
    --data_path configs/dreamx/eval.json --output_folder demo_goldctx_s123_30s/ \
    --num_output_frames 123 --seed 123 --chunk_relative --modes open gold_ctx \
    --kp2 0.05 --ki2 0.12 --leak2 0.9 --deadband2 0.10 --gain_max 0.3 \
    --decode_every 3 --gold_thresh 0.2 --ctx_base 0.1 --ctx_k 2.0 --ctx_max 0.4 --ctx_deadband 0.1 \
    --drift_stride 2 --shard_id $gpu --num_shards $NS > /tmp/fs123_$gpu.log 2>&1 &
  pids+=($!)
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" && echo "shard $i ok" || fail=1; done
echo "ALL_FS123_DONE fail=$fail" > /home/ma-user/work/dataset/VidGen_data_obs/DreamX-World/.fs123_done
