#!/bin/bash

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 N_JOBS CHECKPOINT MANIFEST EVALUATION_ID MODE" >&2
    exit 1
fi

N_JOBS="$1"
CHECKPOINT="$2"
MANIFEST="$3"
EVALUATION_ID="$4"
MODE="$5"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
CHAIN_DIR="$ARTIFACT_ROOT/evaluations/$EVALUATION_ID/chains"
DEPENDENCY="${CROSCIM_CHAIN_AFTER:-}"

if ! [[ "$N_JOBS" =~ ^[1-9][0-9]*$ ]] || [ "$N_JOBS" -gt 10 ]; then
    echo "ERROR: N_JOBS must be between 1 and 10 for the development QOS" >&2
    exit 2
fi
mkdir -p "$CHAIN_DIR" "$PROJECT_DIR/logs"
manifest="$CHAIN_DIR/chain_$(date +%Y%m%d_%H%M%S).txt"
printf 'evaluation_id=%s\nmode=%s\ncheckpoint=%s\nmanifest=%s\n' \
    "$EVALUATION_ID" "$MODE" "$CHECKPOINT" "$MANIFEST" > "$manifest"

for index in $(seq 1 "$N_JOBS"); do
    dependency_args=()
    if [ -n "$DEPENDENCY" ]; then
        dependency_args=(--dependency="afterany:$DEPENDENCY")
    fi
    job_id="$(sbatch --parsable \
        "${dependency_args[@]}" \
        scripts/jeanzay/evaluate_resunet_publication.slurm \
        "$CHECKPOINT" "$MANIFEST" "$EVALUATION_ID" "$MODE")"
    printf 'job_%s=%s dependency=%s\n' "$index" "$job_id" "${DEPENDENCY:-none}" | tee -a "$manifest"
    DEPENDENCY="$job_id"
done
printf '%s\n' "$DEPENDENCY" > "$CHAIN_DIR/last_job.txt"
echo "last_job=$DEPENDENCY"
echo "chain_manifest=$manifest"
