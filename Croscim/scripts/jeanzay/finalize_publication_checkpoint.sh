#!/bin/bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RUN_ID CHECKPOINT" >&2
    exit 1
fi

RUN_ID="$1"
CHECKPOINT="$2"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
RUN_DIR="$ARTIFACT_ROOT/runs/$RUN_ID"
SNAPSHOT_DIR="$ARTIFACT_ROOT/checkpoints/publication_frozen/$RUN_ID"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh ""
umask 0007
mkdir -p "$SNAPSHOT_DIR"

if [ ! -s "$RUN_DIR/norm_stats.sha256" ]; then
    echo "ERROR: training normalization hash is missing: $RUN_DIR/norm_stats.sha256" >&2
    exit 2
fi
sha256sum -c "$RUN_DIR/norm_stats.sha256"

python -m contrib.SST.evaluation.checkpoint \
    --checkpoint "$CHECKPOINT" \
    --snapshot "$SNAPSHOT_DIR/publication_best.ckpt" \
    --manifest "$SNAPSHOT_DIR/publication_best.json"

cp "$RUN_DIR/norm_stats.sha256" "$SNAPSHOT_DIR/training_norm_stats.sha256"

echo "checkpoint=$SNAPSHOT_DIR/publication_best.ckpt"
echo "manifest=$SNAPSHOT_DIR/publication_best.json"
