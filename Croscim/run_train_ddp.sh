export CUDA_VISIBLE_DEVICES=${1:-0,1,2,3}
export HYDRA_FULL_ERROR=1
echo "=== Test DDP sur GPUs: $CUDA_VISIBLE_DEVICES ==="
python main.py xp=SST/multires_lite_ddp 2>&1 | tee run_train_ddp.log
