#!/bin/bash
# eval_all.sh <folder> <mode>  — full 4-axis eval + verdict for one run folder, local GPUs.
D=$1; M=$2
PY=/home/ma-user/work/dataset/xiaoyi_video_env/miniconda3/envs/dreamx/bin/python
cd /home/ma-user/work/dataset/VidGen_data_obs/DreamX-World
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$s $PY window_eval.py --folder $D/ --modes open $M --stride 2 --lags 4 8 --shard_id $s --num_shards 4 > $D/.we_$s.log 2>&1 & done
wait
for s in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY camera_track.py --folder $D/ --modes open $M --stride 8 --shard_id $s --num_shards 8 > $D/.ct_$s.log 2>&1 & done
for s in 0 1 2 3; do CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY fair_eval.py --folder $D/ --modes open $M --resize_w 384 --shard_id $s --num_shards 4 > $D/.fe_$s.log 2>&1 & done
wait
CUDA_VISIBLE_DEVICES="" $PY combine_eval.py --folder $D --modes open $M
rm -f $D/.we_*.log $D/.ct_*.log $D/.fe_*.log
