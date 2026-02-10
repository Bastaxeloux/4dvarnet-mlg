#!/bin/bash
# Script de test H100 avec 1 seul GPU
# Usage: ./run_h100_single.sh [GPU_ID]

source ~/.bashrc
cd /home/malegu/4D-MLG/Croscim
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Starting single-GPU test on H100 (GPU $GPU_ID)..."
HYDRA_FULL_ERROR=1 nohup python main.py xp=SST/multires > run_gefion_single.log 2>&1 &
echo "Test started in background (GPU $GPU_ID)."
echo $! > process_gefion_single.pid

echo "Process ID: $(cat process_gefion_single.pid)"
echo "Log file: run_gefion_single.log"
echo ""
echo "Monitor with:"
echo "  tail -f run_gefion_single.log"
echo ""
echo "Kill with:"
echo "  kill \$(cat process_gefion_single.pid)"