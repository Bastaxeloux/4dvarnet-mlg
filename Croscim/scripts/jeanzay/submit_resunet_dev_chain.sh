#!/bin/bash
# Submit sequential two-hour A100 development jobs for one publication run.
#
# Usage:
#   scripts/jeanzay/submit_resunet_dev_chain.sh [N_JOBS] [HYDRA_OVERRIDE ...]
#
# Set CROSCIM_RUN_ID to append to an existing run. Set CROSCIM_CHAIN_AFTER to
# the final job ID of an existing chain when appending jobs before it finishes.

set -euo pipefail

N_JOBS="${1:-12}"
if [ "$#" -gt 0 ]; then
    shift
fi

if ! [[ "$N_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: N_JOBS must be a positive integer." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SLURM_SCRIPT="$PROJECT_DIR/scripts/jeanzay/train_resunet_publication.slurm"
RUN_ID="${CROSCIM_RUN_ID:-resunet_a100_dev_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${CROSCIM_ARTIFACT_ROOT:-${WORK:?WORK is not defined}/croscim/publication}/runs/$RUN_ID"

if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: CROSCIM_RUN_ID contains unsupported characters: $RUN_ID" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/dev_chain_$(date +%Y%m%d_%H%M%S).txt"
PREVIOUS="${CROSCIM_CHAIN_AFTER:-}"

{
    echo "run_id=$RUN_ID"
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "jobs=$N_JOBS"
    echo "initial_dependency=${PREVIOUS:-none}"
    printf 'overrides='
    printf ' %q' "$@"
    echo
} | tee "$MANIFEST"

for index in $(seq 1 "$N_JOBS"); do
    submit_args=(
        --parsable
        --qos=qos_gpu_a100-dev
        --time=02:00:00
        --export="ALL,CROSCIM_RUN_ID=$RUN_ID,CROSCIM_CHAIN_INDEX=$index"
    )
    if [ -n "$PREVIOUS" ]; then
        submit_args+=(--dependency="afterok:$PREVIOUS")
    fi

    job_id="$(sbatch "${submit_args[@]}" "$SLURM_SCRIPT" "$@")"
    echo "job_$index=$job_id dependency=${PREVIOUS:-none}" | tee -a "$MANIFEST"
    PREVIOUS="$job_id"
done

printf '%s\n' "$PREVIOUS" > "$RUN_DIR/dev_chain_last_job.txt"
echo "run_id=$RUN_ID"
echo "last_job=$PREVIOUS"
echo "manifest=$MANIFEST"
