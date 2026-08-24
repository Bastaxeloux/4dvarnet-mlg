#!/bin/bash
#SBATCH --job-name=sst_resu_test
#SBATCH --account=cai@a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/jz_test_%j.out
#SBATCH --error=logs/jz_test_%j.err

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: sbatch $0 CHECKPOINT TEST_ID [test_date_idx] [Hydra overrides...]" >&2
    exit 1
fi

CHECKPOINT="$1"
TEST_ID="$2"
TEST_DATE_IDX="${3:-183}"
shift "$(( $# >= 3 ? 3 : 2 ))"

PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
DATA_ROOT="${CROSCIM_DATA_ROOT:-${SCRATCH:?SCRATCH is not defined}/croscim/data_sst}"
TEST_ROOT="${CROSCIM_TEST_ROOT:-$ARTIFACT_ROOT/evaluations/diagnostic/$TEST_ID}"
STATS_FILE="$ARTIFACT_ROOT/manifests/norm_stats_2017_2022.yaml"
TB_DIR="$TEST_ROOT/tensorboard"
OUTPUT_DIR="$TEST_ROOT/outputs"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh arch/a100
umask 0007
mkdir -p logs "$TEST_ROOT" "$TB_DIR" "$OUTPUT_DIR" "$WORK/tmp"

export TMPDIR="$WORK/tmp"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CROSCIM_MEM_DEBUG=0
export CROSCIM_SSL_DEBUG=0
export CROSCIM_RESIDUAL_DEBUG=0

if [ ! -s "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found or empty: $CHECKPOINT" >&2
    exit 2
fi
if [ ! -s "$STATS_FILE" ]; then
    echo "ERROR: normalization file not found: $STATS_FILE" >&2
    exit 2
fi
if [ ! -d "$DATA_ROOT/data_2024" ]; then
    echo "ERROR: 2024 test data not found: $DATA_ROOT/data_2024" >&2
    exit 2
fi

python scripts/jeanzay/check_environment.py

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if ! grep -q "A100" <<<"$gpu_name"; then
    echo "ERROR: expected one A100 GPU, found: $gpu_name" >&2
    exit 2
fi

sha256sum "$CHECKPOINT" > "$TEST_ROOT/checkpoint.sha256"
sha256sum "$STATS_FILE" > "$TEST_ROOT/norm_stats.sha256"
python -c "import torch; p='$CHECKPOINT'; c=torch.load(p, map_location='cpu', weights_only=False); print('epoch:', c.get('epoch')); print('global_step:', c.get('global_step')); print('pytorch_lightning_version:', c.get('pytorch-lightning_version'))" \
    > "$TEST_ROOT/checkpoint_metadata.txt"

cat > "$TEST_ROOT/run_manifest.txt" <<EOF
test_id=$TEST_ID
purpose=diagnostic_only_not_for_publication_metrics
job_id=${SLURM_JOB_ID:-manual}
checkpoint=$CHECKPOINT
test_date_idx=$TEST_DATE_IDX
data_root=$DATA_ROOT
stats_file=$STATS_FILE
project_dir=$PROJECT_DIR
gpu=$gpu_name
start=$(date --iso-8601=seconds)
hydra_overrides=$*
EOF

entrypoint_override="entrypoints=[{_target_: src.test.base_test, trainer: \${trainer}, lit_mod: \${model}, dm: \${datamodule}, ckpt_path: '$CHECKPOINT', save_dir: '$TB_DIR'}]"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Test ID: $TEST_ID"
echo "Checkpoint: $CHECKPOINT"
cat "$TEST_ROOT/checkpoint_metadata.txt"
echo "Target date index: $TEST_DATE_IDX (183 is 2024-07-02)"
echo "Artifacts: $TEST_ROOT"
echo "Start: $(date --iso-8601=seconds)"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

srun --kill-on-bad-exit=1 python -u main.py \
    xp=SST/multires_jeanzay_resunet \
    "$entrypoint_override" \
    "paths.src=$PROJECT_DIR" \
    "paths.artifacts=$ARTIFACT_ROOT" \
    "model.outputs_dir=$OUTPUT_DIR" \
    "datamodule.sst_daily_paths=$DATA_ROOT" \
    "datamodule.xrds_kw.test_single_day=true" \
    "datamodule.xrds_kw.test_date_idx=$TEST_DATE_IDX" \
    "datamodule.dl_kw.batch_size=2" \
    "datamodule.dl_kw.num_workers=6" \
    "trainer.devices=1" \
    "trainer.strategy=auto" \
    "trainer.precision=bf16-mixed" \
    "trainer.callbacks=[]" \
    "trainer.num_sanity_val_steps=0" \
    "hydra.run.dir=$TEST_ROOT/hydra" \
    "$@" 2>&1 | tee "$TEST_ROOT/test.log"

printf 'end=%s\n' "$(date --iso-8601=seconds)" >> "$TEST_ROOT/run_manifest.txt"
touch "$TEST_ROOT/test_complete"

echo "Test complete: $TEST_ROOT"
