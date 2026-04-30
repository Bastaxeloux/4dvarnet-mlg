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

# Script pour lancer UNE run avec paramètres customisés sur 1 GPU H100
# Usage: sbatch submit_gefion_single.sh EXP_NAME [OVERRIDES...]
#
# Exemples:
#   sbatch submit_gefion_single.sh exp_lr1e4 +model.opt_lr=1e-4
#   sbatch submit_gefion_single.sh exp_grad05 model.loss_weights.grad=0.05
#   sbatch submit_gefion_single.sh exp_bignet model.solver.solvers.solver_x1.prior_cost.dim_hidden=128
#   sbatch submit_gefion_single.sh exp_prior02 model.loss_weights.prior=0.2 +model.opt_lr=5e-4
#
# Paramètres testables:
#   +model.opt_lr=...                    Learning rate (1e-3, 5e-4, 1e-4, 2e-3)
#   model.loss_weights.grad=...          Gradient loss weight (0.001, 0.01, 0.05)
#   model.loss_weights.prior=...         Prior loss weight (0.05, 0.1, 0.2)
#   model.solver.solvers.solver_x1.n_step=...           Iterations (10, 15, 20, 30)
#   model.solver.solvers.solver_x1.prior_cost.dim_hidden=...  Network size (64, 96, 128, 192)
#   model.solver.solvers.solver_x1.grad_mod.dim_hidden=...    ConvLSTM size (96, 128, 192)
#   datamodule.dl_kw.batch_size=...      Batch size (4, 6, 8, 10)
#   trainer.accumulate_grad_batches=...  Accumulation (2, 4, 8)

EXPERIMENT_NAME=${1:-"default"}

echo "================================================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start: $(date)"
echo "Overrides: $@"
echo "================================================================"

# Modules et environnement
source /dcai/users/guimae/4dvarnet-mlg/Croscim/scripts/gefion/env.sh
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
mkdir -p logs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/outputs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/results
mkdir -p /dcai/projects/cu_0026/guimae/croscim/checkpoints

# Variables d'environnement
export NUMEXPR_MAX_THREADS=28
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""
python main.py xp=SST/multires_single_gefion trainer.logger.version="$EXPERIMENT_NAME" "$@" 2>&1 | tee logs/train_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log
echo ""
echo "Job finished: $(date)"
