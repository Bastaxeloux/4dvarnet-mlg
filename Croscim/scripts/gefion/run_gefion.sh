#!/bin/bash
# Interactive launch for Gefion H100 DDP.
# Prefer `sbatch scripts/gefion/train_gefion.sh` for real runs.

source /dcai/users/guimae/4dvarnet-mlg/Croscim/scripts/gefion/env.sh
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"
mkdir -p logs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/outputs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/results
mkdir -p /dcai/projects/cu_0026/guimae/croscim/checkpoints

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
