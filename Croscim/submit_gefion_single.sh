#!/bin/bash
#SBATCH --job-name=sst_exp
#SBATCH --account=cu_0026
#SBATCH --output=logs/exp_%A.out
#SBATCH --error=logs/exp_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=12:00:00

# Script pour lancer UNE SEULE run avec paramètres customisés
# Usage: sbatch submit_gefion_single.sh [EXPERIMENT_NAME] [OVERRIDES...]
#
# Exemples:
#   sbatch submit_gefion_single.sh exp_lr1e4 model.opt_lr=1e-4
#   sbatch submit_gefion_single.sh exp_grad05 model.loss_weights.grad=0.05
#   sbatch submit_gefion_single.sh exp_bignet solver.solver_x1.prior_cost.dim_hidden=128

EXPERIMENT_NAME=${1:-"default"}
shift  # Retire le premier argument, garde les overrides

echo "================================================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start: $(date)"
echo "Overrides: $@"
echo "================================================================"

# Modules et environnement
module load GCC/12.3.0 Boost/1.82.0 snappy/1.1.10 GSL/2.7 Eigen/3.4.0 CUDA/12.8.0 Python/3.11.3
source /dcai/projects/cu_0026/croscim_env/bin/activate
cd /dcai/projects/cu_0026/Croscim
export PYTHONPATH=$PWD:$PYTHONPATH
mkdir -p logs

# Variables d'environnement
export NUMEXPR_MAX_THREADS=48
export HYDRA_FULL_ERROR=1
export DASK_SCHEDULER=synchronous

# Info GPU
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Lancement avec overrides
# TensorBoard version = experiment name pour différencier les runs
python main.py xp=SST/multires_gefion \
    trainer.logger.version="$EXPERIMENT_NAME" \
    "$@" \
    2>&1 | tee logs/train_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log

echo ""
echo "================================================================"
echo "Job finished: $(date)"
echo "================================================================"
