#!/bin/bash
# Script de lancement pour cluster H100 (8 GPUs DDP)
# Usage: ./run_h100.sh

source ~/.bashrc
cd /home/malegu/4D-MLG/Croscim
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"

# Multi-GPU DDP: PyTorch Lightning gère automatiquement avec strategy=ddp
# Pas besoin de CUDA_VISIBLE_DEVICES, il utilise tous les devices spécifiés dans config

echo "Starting multi GPU DDP training cluster..."
HYDRA_FULL_ERROR=1 nohup python main.py xp=SST/multires_gefion > run_gefion.log 2>&1 &
echo "Training started in background."
echo $! > process_gefion.pid

echo "Process ID: $(cat process_gefion.pid)"
echo "Log file: run_gefion.log"
echo ""
echo "Monitor with:"
echo "  tail -f run_gefion.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill with:"
echo "  kill \$(cat process_gefion.pid)"