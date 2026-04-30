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

module load GCC/12.3.0 Boost/1.82.0 snappy/1.1.10 GSL/2.7 Eigen/3.4.0 CUDA/12.8.0 Python/3.11.3
source /dcai/projects/cu_0026/croscim_env/bin/activate
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
export PYTHONPATH=$PWD:$PYTHONPATH

mkdir -p logs

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total --format=csv

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NUMEXPR_MAX_THREADS=64
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

# Diagnostics DDP/NCCL : utile si le job deadlock ou bloque au premier batch.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Désactiver le threading Dask (évite deadlocks DDP)
export DASK_SCHEDULER=synchronous

echo "MASTER_ADDR=$MASTER_ADDR | WORLD_SIZE=$WORLD_SIZE | SLURM_NTASKS=$SLURM_NTASKS"

srun --kill-on-bad-exit=1 python main.py xp=SST/multires_gefion 2>&1 | tee logs/train_${SLURM_JOB_ID}.log


# sbatch train_gefion.sh
# squeue -u $USER
# tail -f logs/slurm_<job_id>.out
# ssh <node_name> && nvtop         # Voir GPU usage (ex: ssh dgx011)
# scancel <job_id>              # Annuler un job si nécessaire
# sacct -S 2026-01-01 --format=JobID,JobName,State,Elapsed,MaxRSS   # voir historique des jobs
