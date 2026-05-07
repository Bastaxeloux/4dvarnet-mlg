#!/bin/bash
#SBATCH --job-name=sst_test_ckpt
#SBATCH --account=cu_0026
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=12:00:00

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 /path/to/checkpoint.ckpt [hydra overrides...]"
    echo "Example: sbatch $0 /dcai/projects/cu_0026/guimae/croscim/checkpoints/last.ckpt"
    exit 1
fi

CHECKPOINT=$1
shift

CROSCIM_ROOT="${CROSCIM_ROOT:-/dcai/users/guimae/4dvarnet-mlg/Croscim}"

cd "$CROSCIM_ROOT"
mkdir -p logs

if [ ! -e "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-manual} | Node: ${SLURM_JOB_NODELIST:-unknown} | Start: $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Extra Hydra overrides: $*"

export TEST_LOG_FILE="logs/test_${SLURM_JOB_ID:-manual}.log"

srun --kill-on-bad-exit=1 bash scripts/gefion/run_test_checkpoint_gefion.sh "$CHECKPOINT" auto "$@"
