#!/bin/bash
#SBATCH --job-name=sst_eval
#SBATCH --account=cai@a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:58:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/jz_eval_%j.out
#SBATCH --error=logs/jz_eval_%j.err

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: sbatch $0 CHECKPOINT MANIFEST EVALUATION_ID MODE [FROZEN_PROTOCOL]" >&2
    exit 1
fi

CHECKPOINT="$1"
MANIFEST="$2"
EVALUATION_ID="$3"
MODE="$4"
FROZEN_PROTOCOL="${5:-}"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
DATA_ROOT="${CROSCIM_DATA_ROOT:-${SCRATCH:?SCRATCH is not defined}/croscim/data_sst_v2}"
PROTOCOL_ROOT="${CROSCIM_PROTOCOL_ROOT:-$ARTIFACT_ROOT/evaluation_protocol_v2}"
OUTPUT_ROOT="$ARTIFACT_ROOT/evaluations/$EVALUATION_ID"
COASTAL_MASK="$PROTOCOL_ROOT/static/coastal_mask_50km.npz"
OI_REPORT="$PROTOCOL_ROOT/dmi_oi_verification.json"
DMI_OI_UNITS="${CROSCIM_DMI_OI_UNITS:-kelvin}"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh arch/a100
umask 0007
mkdir -p logs "$OUTPUT_ROOT/job_logs" "$OUTPUT_ROOT/done/$MODE" "$WORK/tmp"
export TMPDIR="$WORK/tmp"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

for required in "$CHECKPOINT" "$MANIFEST" "$COASTAL_MASK" "$OI_REPORT"; do
    if [ ! -s "$required" ]; then
        echo "ERROR: required evaluation artifact is missing: $required" >&2
        exit 2
    fi
done
if [ -n "$FROZEN_PROTOCOL" ] && [ ! -s "$FROZEN_PROTOCOL" ]; then
    echo "ERROR: frozen protocol is missing: $FROZEN_PROTOCOL" >&2
    exit 2
fi
if [ -e "$OUTPUT_ROOT/fatal_error" ]; then
    echo "ERROR: a preceding allocation recorded a fatal evaluation error" >&2
    cat "$OUTPUT_ROOT/fatal_error" >&2
    exit 3
fi

expected_dates="$(python -c "import json; print(len(json.load(open('$MANIFEST'))['records']))")"
completed_dates="$(find "$OUTPUT_ROOT/done/$MODE" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
if [ "$completed_dates" -eq "$expected_dates" ]; then
    echo "Evaluation already complete: $completed_dates/$expected_dates dates"
    touch "$OUTPUT_ROOT/evaluation_complete_$MODE"
    exit 0
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Evaluation: $EVALUATION_ID"
echo "Mode: $MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Manifest: $MANIFEST"
echo "Progress before allocation: $completed_dates/$expected_dates"
echo "Start: $(date --iso-8601=seconds)"

walltime_signal=0
srun_pid=""
handle_walltime() {
    walltime_signal=1
    touch "$OUTPUT_ROOT/job_logs/walltime_${SLURM_JOB_ID:-manual}"
    if [ -n "$srun_pid" ]; then
        kill -TERM "$srun_pid" 2>/dev/null || true
    fi
}
trap handle_walltime USR1

set +e
evaluator_args=(
    --project-root "$PROJECT_DIR"
    --checkpoint "$CHECKPOINT"
    --manifest "$MANIFEST"
    --data-root "$DATA_ROOT"
    --output-root "$OUTPUT_ROOT"
    --coastal-mask "$COASTAL_MASK"
    --oi-verification "$OI_REPORT"
    --mode "$MODE"
    --dmi-oi-units "$DMI_OI_UNITS"
    --batch-size "${CROSCIM_EVAL_BATCH_SIZE:-2}"
    --num-workers "${CROSCIM_EVAL_NUM_WORKERS:-2}"
)
if [ -n "$FROZEN_PROTOCOL" ]; then
    evaluator_args+=(--frozen-protocol "$FROZEN_PROTOCOL")
fi
if [ -n "${CROSCIM_EVAL_LIMIT_DATES:-}" ]; then
    evaluator_args+=(--limit-dates "$CROSCIM_EVAL_LIMIT_DATES")
fi
srun --kill-on-bad-exit=1 --gpu-bind=closest \
    --output="$OUTPUT_ROOT/job_logs/job_${SLURM_JOB_ID}_rank_%t.out" \
    --error="$OUTPUT_ROOT/job_logs/job_${SLURM_JOB_ID}_rank_%t.err" \
    python -u -m contrib.SST.evaluation.evaluator \
        "${evaluator_args[@]}" &
srun_pid=$!
wait "$srun_pid"
status=$?
set -e

if [ "$status" -ne 0 ] && [ "$walltime_signal" -eq 0 ]; then
    printf 'job=%s status=%s time=%s\n' \
        "${SLURM_JOB_ID:-manual}" "$status" "$(date --iso-8601=seconds)" \
        > "$OUTPUT_ROOT/fatal_error"
    exit "$status"
fi

completed_dates="$(find "$OUTPUT_ROOT/done/$MODE" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
echo "Progress after allocation: $completed_dates/$expected_dates"
if [ "$completed_dates" -eq "$expected_dates" ]; then
    touch "$OUTPUT_ROOT/evaluation_complete_$MODE"
    echo "Evaluation complete"
fi
echo "End: $(date --iso-8601=seconds)"
