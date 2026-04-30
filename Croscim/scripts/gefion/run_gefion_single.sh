#!/bin/bash
# Single-GPU run on Gefion (H100), background nohup mode
# Usage: ./run_gefion_single.sh [GPU_ID]
#
# À exécuter dans une session interactive (`srun --pty ...`) ou sur un node
# Gefion. Pour soumettre via SLURM, utiliser plutôt `submit_gefion_single.sh`.
#
# Note : non testé en conditions réelles, lire et vérifier modules + venv
# avant la première exécution.
set -euo pipefail

# Modules et environnement (même bloc que submit_gefion_single.sh)
source /dcai/users/guimae/4dvarnet-mlg/Croscim/scripts/gefion/env.sh
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
mkdir -p /dcai/projects/cu_0026/guimae/croscim/outputs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/results
mkdir -p /dcai/projects/cu_0026/guimae/croscim/checkpoints

# Variables d'environnement run-time
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"
export NUMEXPR_MAX_THREADS=28
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Starting single-GPU run on Gefion (GPU $GPU_ID)..."
nohup python main.py xp=SST/multires_single_gefion > run_gefion_single.log 2>&1 &
echo "Run started in background (GPU $GPU_ID)."
echo $! > process_gefion_single.pid

echo "Process ID: $(cat process_gefion_single.pid)"
echo "Log file: run_gefion_single.log"
echo ""
echo "Monitor with:"
echo "  tail -f run_gefion_single.log"
echo ""
echo "Kill with:"
echo "  kill \$(cat process_gefion_single.pid)"
