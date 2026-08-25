#!/bin/bash
# Submit sequential two-hour A100 development jobs for one publication run.
#
# Usage:
#   scripts/jeanzay/submit_resunet_dev_chain.sh [N_JOBS] [HYDRA_OVERRIDE ...]
#
# Set CROSCIM_RUN_ID to append to an existing run. Set CROSCIM_CHAIN_AFTER to
# the final job ID of an existing chain when appending jobs before it finishes.

set -euo pipefail

N_JOBS="${1:-10}"
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
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-${WORK:?WORK is not defined}/croscim/publication}"
RUN_DIR="$ARTIFACT_ROOT/runs/$RUN_ID"
LAST_CHECKPOINT="$ARTIFACT_ROOT/checkpoints/$RUN_ID/last.ckpt"
LAST_JOB_FILE="$RUN_DIR/dev_chain_last_job.txt"
NEXT_EPOCH_FILE="$RUN_DIR/dev_chain_next_epoch.txt"
EPOCHS_PER_RESOLUTION=8
MAX_EPOCHS=192

if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: CROSCIM_RUN_ID contains unsupported characters: $RUN_ID" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
MANIFEST="$RUN_DIR/dev_chain_$(date +%Y%m%d_%H%M%S).txt"
PREVIOUS="${CROSCIM_CHAIN_AFTER:-}"

if [ -n "$PREVIOUS" ] && [ -s "$NEXT_EPOCH_FILE" ]; then
    if [ -s "$LAST_JOB_FILE" ] && [ "$(cat "$LAST_JOB_FILE")" != "$PREVIOUS" ]; then
        echo "ERROR: CROSCIM_CHAIN_AFTER=$PREVIOUS does not match the planned final job $(cat "$LAST_JOB_FILE")." >&2
        exit 1
    fi
    NEXT_EPOCH="$(cat "$NEXT_EPOCH_FILE")"
elif [ -s "$LAST_CHECKPOINT" ]; then
    NEXT_EPOCH="$(python - "$LAST_CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(checkpoint["epoch"]) + 1)
PY
)"
else
    NEXT_EPOCH=0
fi

if ! [[ "$NEXT_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid next epoch: $NEXT_EPOCH" >&2
    exit 1
fi

{
    echo "run_id=$RUN_ID"
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "jobs=$N_JOBS"
    echo "initial_dependency=${PREVIOUS:-none}"
    echo "projected_start_epoch=$NEXT_EPOCH"
    echo "epoch_schedule=x10:4,x3:4,x1:2"
    printf 'overrides='
    printf ' %q' "$@"
    echo
} | tee "$MANIFEST"

for index in $(seq 1 "$N_JOBS"); do
    if [ "$NEXT_EPOCH" -ge "$MAX_EPOCHS" ]; then
        echo "Training is projected to finish at epoch $MAX_EPOCHS; no more jobs submitted." | tee -a "$MANIFEST"
        break
    fi

    position_in_cycle=$((NEXT_EPOCH % (EPOCHS_PER_RESOLUTION * 3)))
    resolution_index=$((position_in_cycle / EPOCHS_PER_RESOLUTION))
    position_in_resolution=$((position_in_cycle % EPOCHS_PER_RESOLUTION))
    remaining_in_resolution=$((EPOCHS_PER_RESOLUTION - position_in_resolution))

    case "$resolution_index" in
        0) resolution=x10; resolution_limit=4 ;;
        1) resolution=x3;  resolution_limit=4 ;;
        2) resolution=x1;  resolution_limit=2 ;;
    esac

    epochs_this_job="$resolution_limit"
    if [ "$remaining_in_resolution" -lt "$epochs_this_job" ]; then
        epochs_this_job="$remaining_in_resolution"
    fi
    if [ $((MAX_EPOCHS - NEXT_EPOCH)) -lt "$epochs_this_job" ]; then
        epochs_this_job=$((MAX_EPOCHS - NEXT_EPOCH))
    fi
    projected_next_epoch=$((NEXT_EPOCH + epochs_this_job))

    submit_args=(
        --parsable
        --qos=qos_gpu_a100-dev
        --time=02:00:00
        --export="ALL,CROSCIM_RUN_ID=$RUN_ID,CROSCIM_CHAIN_INDEX=$index,CROSCIM_EPOCHS_PER_JOB=$epochs_this_job"
    )
    if [ -n "$PREVIOUS" ]; then
        submit_args+=(--dependency="afterok:$PREVIOUS")
    fi

    job_id="$(sbatch "${submit_args[@]}" "$SLURM_SCRIPT" "$@")"
    echo "job_$index=$job_id dependency=${PREVIOUS:-none} resolution=$resolution start_epoch=$NEXT_EPOCH complete_epochs=$epochs_this_job projected_next_epoch=$projected_next_epoch" | tee -a "$MANIFEST"
    PREVIOUS="$job_id"
    NEXT_EPOCH="$projected_next_epoch"
    printf '%s\n' "$PREVIOUS" > "$LAST_JOB_FILE"
    printf '%s\n' "$NEXT_EPOCH" > "$NEXT_EPOCH_FILE"
done

echo "run_id=$RUN_ID"
echo "last_job=$PREVIOUS"
echo "projected_next_epoch=$NEXT_EPOCH"
echo "manifest=$MANIFEST"
