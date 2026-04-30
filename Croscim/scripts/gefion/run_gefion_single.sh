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
module load GCC/12.3.0 Boost/1.82.0 snappy/1.1.10 GSL/2.7 Eigen/3.4.0 CUDA/12.8.0 Python/3.11.3
source /dcai/projects/cu_0026/croscim_env/bin/activate
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
export PYTHONPATH=$PWD:$PYTHONPATH

# Variables d'environnement run-time
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"
export NUMEXPR_MAX_THREADS=28
export HYDRA_FULL_ERROR=1
export DASK_SCHEDULER=synchronous

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
