# Usage: ./run.sh [GPU_ID] [CKPT_PATH]
source ~/.bashrc
# conda activate 4denv
cd /home/malegu/4D-MLG/Croscim
export PYTHONWARNINGS="ignore::UserWarning:esm_tools"

GPU_ID=${1:-2}
CKPT_PATH=${2:-""}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "GPU $GPU_ID"

if [ -n "$CKPT_PATH" ]; then
    echo "Resuming from checkpoint: $CKPT_PATH"
    # Detect most recent TensorBoard run to continue logging in the same run
    LAST_VERSION=$(ls -td /dmidata/projects/4dvarnet/results_sst_multires/run/*/ 2>/dev/null | head -1 | xargs basename)
    if [ -n "$LAST_VERSION" ]; then
        echo "Continuing TensorBoard run: $LAST_VERSION"
        HYDRA_FULL_ERROR=1 nohup python main.py xp=SST/multires +ckpt="$CKPT_PATH" trainer.logger.version="$LAST_VERSION" > run.log 2>&1 &
    else
        echo "No previous TensorBoard run found, starting new run"
        HYDRA_FULL_ERROR=1 nohup python main.py xp=SST/multires +ckpt="$CKPT_PATH" > run.log 2>&1 &
    fi
else
    HYDRA_FULL_ERROR=1 nohup python main.py xp=SST/multires > run.log 2>&1 &
fi
echo "Training started in background."
echo $! > process.pid
# kill it with
# kill $(cat process.pid)

#lancer avec 
# conda activate 4denv && ./run.sh 0

# voir la ram avec 
# watch -n 1 "echo '=== RAM ==='; free -h | head -2; echo '=== GPU ==='; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; echo '=== SWAP ==='; free -h | tail -1"