#!/bin/bash
# Test checkpoint sur Gefion HPC (mono-GPU)
# Usage: ./run_test_checkpoint_gefion.sh /path/to/checkpoint.ckpt [GPU_ID]

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/checkpoint.ckpt [GPU_ID]"
    echo "Example: $0 /dcai/projects/cu_0026/checkpoints/epoch=030-val/loss=0.01234.ckpt 0"
    exit 1
fi

CHECKPOINT=$1
GPU_ID=${2:-0}

export CUDA_VISIBLE_DEVICES=$GPU_ID
export HYDRA_FULL_ERROR=1

echo "=== Test sur Gefion GPU $GPU_ID ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

HYDRA_FULL_ERROR=1 python3 main.py \
    xp=SST/multires_gefion \
    "entrypoints=[{_target_: src.test.base_test, trainer: \${trainer}, lit_mod: \${model}, dm: \${datamodule}, ckpt_path: '$CHECKPOINT', save_dir: '/dcai/projects/cu_0026/outputs/test_results'}]" \
    trainer.logger.save_dir=/dcai/projects/cu_0026/outputs/test_results \
    trainer.devices=1 \
    trainer.strategy=null \
    trainer.precision=bf16-mixed \
    datamodule.domains.test.time._args_=['2024-11-01','2024-11-30'] \
    datamodule.xrds_kw.test_single_day=true \
    datamodule.xrds_kw.test_date_idx=15 \
    2>&1 | tee test_gefion.log

echo ""
echo "Test completed. Check outputs/test_results/ for NetCDF files."
