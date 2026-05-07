#!/bin/bash
# Test checkpoint sur Gefion HPC (mono-GPU)
# Usage: ./run_test_checkpoint_gefion.sh /path/to/checkpoint.ckpt [GPU_ID] [hydra overrides...]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 /path/to/checkpoint.ckpt [GPU_ID|auto] [hydra overrides...]"
    echo "Example: $0 /dcai/projects/cu_0026/guimae/croscim/checkpoints/epoch=030-val/loss=0.01234.ckpt auto"
    exit 1
fi

CHECKPOINT=$1
GPU_ID=${2:-auto}
EXTRA_OVERRIDES=("${@:3}")
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-/dcai/projects/cu_0026/guimae/croscim/outputs/test_results}"
LOG_FILE="${TEST_LOG_FILE:-logs/test_${SLURM_JOB_ID:-manual}.log}"

source /dcai/users/guimae/4dvarnet-mlg/Croscim/scripts/gefion/env.sh
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
mkdir -p logs "$TEST_OUTPUT_DIR"

if [ "$GPU_ID" != "auto" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

echo "=== Test sur Gefion GPU $GPU_ID ==="
echo "Checkpoint: $CHECKPOINT"
echo "TensorBoard test logs: $TEST_OUTPUT_DIR"
echo "Main log: $LOG_FILE"
echo ""

HYDRA_FULL_ERROR=1 python3 main.py \
    xp=SST/multires_gefion \
    "entrypoints=[{_target_: src.test.base_test, trainer: \${trainer}, lit_mod: \${model}, dm: \${datamodule}, ckpt_path: '$CHECKPOINT', save_dir: '$TEST_OUTPUT_DIR'}]" \
    trainer.logger.save_dir="$TEST_OUTPUT_DIR" \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.precision=bf16-mixed \
    datamodule.xrds_kw.test_single_day=true \
    datamodule.xrds_kw.test_date_idx=15 \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Test completed."
echo "TensorBoard logs: $TEST_OUTPUT_DIR"
echo "Model artifacts: /dcai/projects/cu_0026/guimae/croscim/outputs/<test_run_id>/test/"
