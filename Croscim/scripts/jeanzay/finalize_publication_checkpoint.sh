#!/bin/bash

set -euo pipefail

if [ "$#" -lt 3 ] || [ $((($# - 1) % 2)) -ne 0 ]; then
    echo "Usage: $0 RUN_ID CHECKPOINT PILOT_EVALUATION_ROOT [CHECKPOINT PILOT_EVALUATION_ROOT ...]" >&2
    exit 1
fi

RUN_ID="$1"
shift
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
RUN_DIR="$ARTIFACT_ROOT/runs/$RUN_ID"
SNAPSHOT_DIR="$ARTIFACT_ROOT/checkpoints/publication_frozen/$RUN_ID"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh
umask 0007
mkdir -p "$SNAPSHOT_DIR"

if [ ! -s "$RUN_DIR/norm_stats.sha256" ]; then
    echo "ERROR: training normalization hash is missing: $RUN_DIR/norm_stats.sha256" >&2
    exit 2
fi
sha256sum -c "$RUN_DIR/norm_stats.sha256"

candidate_args=()
while [ "$#" -gt 0 ]; do
    candidate_args+=(--candidate "$1" "$2")
    shift 2
done

python scripts/publication/select_checkpoint.py \
    "${candidate_args[@]}" \
    --output-dir "$SNAPSHOT_DIR"

cp "$RUN_DIR/norm_stats.sha256" "$SNAPSHOT_DIR/training_norm_stats.sha256"

echo "checkpoint=$SNAPSHOT_DIR/publication_best.ckpt"
echo "manifest=$SNAPSHOT_DIR/publication_best.json"
echo "selection_report=$SNAPSHOT_DIR/selection_report.json"
