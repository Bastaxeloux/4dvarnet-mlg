#!/bin/bash
#SBATCH --job-name=sst_multires
#SBATCH --account=cu_0026
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=8
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPUs: $SLURM_GPUS | Start: $(date)"

source /dcai/users/guimae/4dvarnet-mlg/Croscim/scripts/gefion/env.sh
cd /dcai/users/guimae/4dvarnet-mlg/Croscim

mkdir -p logs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/outputs
mkdir -p /dcai/projects/cu_0026/guimae/croscim/results
mkdir -p /dcai/projects/cu_0026/guimae/croscim/checkpoints

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import src.train; print('src.train:', src.train.__file__)"
python -c "import contrib.SST.models; print('contrib.SST.models:', contrib.SST.models.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NUMEXPR_MAX_THREADS=64
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONFAULTHANDLER=1

# Diagnostics DDP/NCCL : utile si le job deadlock ou bloque au premier batch.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "MASTER_ADDR=$MASTER_ADDR | WORLD_SIZE=$WORLD_SIZE | SLURM_NTASKS=$SLURM_NTASKS"

srun --kill-on-bad-exit=1 python main.py xp=SST/multires_gefion 2>&1 | tee logs/train_${SLURM_JOB_ID}.log


# sbatch train_gefion.sh
# squeue -u $USER
# tail -f logs/slurm_<job_id>.out
# ssh <node_name> && nvtop         # Voir GPU usage (ex: ssh dgx011)
# scancel <job_id>              # Annuler un job si nécessaire
# sacct -S 2026-01-01 --format=JobID,JobName,State,Elapsed,MaxRSS   # voir historique des jobs
