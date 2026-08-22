#!/bin/bash
#SBATCH --job-name=sst_resunet_pub
#SBATCH --account=cai@a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/jz_train_%j.out
#SBATCH --error=logs/jz_train_%j.err

set -euo pipefail

PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
DATA_ROOT="${CROSCIM_DATA_ROOT:-${SCRATCH:?SCRATCH is not defined}/croscim/data_sst}"
STATS_FILE="$ARTIFACT_ROOT/manifests/norm_stats_2017_2022.yaml"

if [ -z "${CROSCIM_RUN_ID:-}" ]; then
    echo "ERROR: CROSCIM_RUN_ID must be set and reused for every continuation." >&2
    exit 1
fi

RUN_ID="$CROSCIM_RUN_ID"
RUN_DIR="$ARTIFACT_ROOT/runs/$RUN_ID"
CHECKPOINT_DIR="$ARTIFACT_ROOT/checkpoints/$RUN_ID"
LAST_CHECKPOINT="$CHECKPOINT_DIR/last.ckpt"
TENSORBOARD_DIR="$ARTIFACT_ROOT/tensorboard/resunet_publication_2017_2024/$RUN_ID"

cd "$PROJECT_DIR"
git_commit="${CROSCIM_GIT_COMMIT:-unavailable}"
git_status="Git status unavailable on compute node"
if command -v git >/dev/null 2>&1; then
    git_commit="$(git rev-parse HEAD)"
    git_status="$(git status --short)"
fi
source scripts/jeanzay/env.sh
umask 0007
mkdir -p logs "$RUN_DIR" "$CHECKPOINT_DIR" \
  "$ARTIFACT_ROOT/outputs" "$ARTIFACT_ROOT/tensorboard"

python scripts/jeanzay/check_environment.py

if [ ! -f "$STATS_FILE" ]; then
    echo "ERROR: missing train-only normalization file: $STATS_FILE" >&2
    exit 1
fi

count_complete_stores() {
    local year_dir="$1"
    local resolution="$2"
    find "$year_dir" -mindepth 2 -maxdepth 2 -type f \
        -path "*_${resolution}.zarr/.zmetadata" | wc -l | tr -d ' '
}

for year in 2017 2018 2019 2020 2021 2022 2023 2024; do
    year_dir="$DATA_ROOT/data_$year"
    if [ ! -d "$year_dir" ]; then
        echo "ERROR: missing preprocessed year: $year_dir" >&2
        exit 1
    fi

    expected_days=365
    if (( year % 400 == 0 || (year % 4 == 0 && year % 100 != 0) )); then
        expected_days=366
    fi
    for resolution in x1 x3 x10; do
        count="$(count_complete_stores "$year_dir" "$resolution")"
        if [ "$count" -ne "$expected_days" ]; then
            echo "ERROR: $year $resolution has $count complete stores; expected $expected_days" >&2
            exit 1
        fi
    done
    echo "Verified $year: $expected_days complete x1/x3/x10 stores"
done

printf '%s\n' "$git_commit" > "$RUN_DIR/git_commit.txt"
printf '%s\n' "$git_status" > "$RUN_DIR/git_status.txt"
python -m pip freeze > "$RUN_DIR/pip_freeze.txt"
sha256sum "$STATS_FILE" > "$RUN_DIR/norm_stats.sha256"

python -c "from contrib.SST.statistics_io import load_stats_section; p='$STATS_FILE'; print('norm keys:', sorted(load_stats_section(p, 'norm_stats'))); print('covariate keys:', sorted(load_stats_section(p, 'norm_stats_covs')))"
python -c "import torch; print(f'PyTorch {torch.__version__}; CUDA={torch.cuda.is_available()}; visible GPUs={torch.cuda.device_count()}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [ "${#gpu_names[@]}" -ne 8 ] || printf '%s\n' "${gpu_names[@]}" | grep -qv 'A100'; then
    echo "ERROR: the A100 launcher requires exactly eight A100 GPUs." >&2
    exit 1
fi

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 10000))
export WORLD_SIZE="$SLURM_NTASKS"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export CROSCIM_MEM_DEBUG="${CROSCIM_MEM_DEBUG:-0}"
export CROSCIM_SSL_DEBUG="${CROSCIM_SSL_DEBUG:-0}"
export CROSCIM_RESIDUAL_DEBUG="${CROSCIM_RESIDUAL_DEBUG:-0}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

hydra_overrides=(
    "paths.src='$PROJECT_DIR'"
    "paths.artifacts='$ARTIFACT_ROOT'"
    "datamodule.sst_daily_paths='$DATA_ROOT'"
    "trainer.logger.version='$RUN_ID'"
    "hydra.run.dir='$RUN_DIR/hydra'"
)

resume_checkpoint="${CROSCIM_RESUME_CKPT:-}"
if [ -z "$resume_checkpoint" ] && [ -s "$LAST_CHECKPOINT" ]; then
    resume_checkpoint="$LAST_CHECKPOINT"
fi
if [ -n "$resume_checkpoint" ]; then
    if [ ! -s "$resume_checkpoint" ]; then
        echo "ERROR: resume checkpoint not found or empty: $resume_checkpoint" >&2
        exit 1
    fi
    hydra_overrides+=("entrypoints[0].ckpt='$resume_checkpoint'")
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "Node: $SLURM_NODELIST"
echo "Project: $PROJECT_DIR"
echo "Virtual environment: ${VIRTUAL_ENV:-unknown}"
echo "Data: $DATA_ROOT"
echo "Artifacts: $ARTIFACT_ROOT"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "TensorBoard: $TENSORBOARD_DIR"
echo "Last checkpoint: $LAST_CHECKPOINT"
echo "Resume checkpoint: ${resume_checkpoint:-none}"
echo "Start: $(date --iso-8601=seconds)"
echo "Additional Hydra overrides: $*"

srun --kill-on-bad-exit=1 python -u main.py \
  xp=SST/multires_jeanzay_resunet \
  "${hydra_overrides[@]}" \
  "$@" 2>&1 | tee -a "$RUN_DIR/train.log"

echo "End: $(date --iso-8601=seconds)"
