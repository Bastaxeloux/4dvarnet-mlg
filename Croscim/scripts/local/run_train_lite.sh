# Usage: ./run_train_lite.sh [GPU_ID]
# Exemple: ./run_train_lite.sh 3
source ~/.bashrc
# conda activate 4denv
cd /home/malegu/4D-MLG/Croscim
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"

GPU_ID=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "GPU $GPU_ID"
HYDRA_FULL_ERROR=1 nohup python main.py xp=SST/multires_lite > run_train_lite.log 2>&1 &
echo "Training started in background."

# then store the process id if needed
echo $! > process.pid
# and kill it with
# kill $(cat process.pid)

#lancer avec 
# conda activate 4denv && ./run_train_lite.sh 0

# voir la ram avec 
# watch -n 1 "echo '=== RAM ==='; free -h | head -2; echo '=== GPU ==='; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; echo '=== SWAP ==='; free -h | tail -1"