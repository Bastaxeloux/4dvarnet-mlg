#!/bin/bash

set -euo pipefail

RUN_ID="${CROSCIM_RUN_ID:-resunet_resbatch_publication_20260822}"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 CHECKPOINT [TEST_DATE_IDX] [HYDRA_OVERRIDES...]" >&2
    echo "Pass a complete-cycle checkpoint explicitly; last.ckpt is not safe for cascade diagnostics." >&2
    exit 2
fi
SOURCE_CHECKPOINT="$1"
TEST_DATE_IDX="${2:-183}"

if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi

if [ ! -s "$SOURCE_CHECKPOINT" ]; then
    echo "ERROR: source checkpoint not found or empty: $SOURCE_CHECKPOINT" >&2
    exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
TEST_ID="${CROSCIM_TEST_ID:-${RUN_ID}_idx${TEST_DATE_IDX}_$timestamp}"
TEST_ROOT="$ARTIFACT_ROOT/evaluations/diagnostic/$TEST_ID"
SNAPSHOT_DIR="$ARTIFACT_ROOT/checkpoints/diagnostic_snapshots"
SNAPSHOT="$SNAPSHOT_DIR/${TEST_ID}.ckpt"

umask 0007
mkdir -p "$TEST_ROOT" "$SNAPSHOT_DIR" "$PROJECT_DIR/logs"

tmp_snapshot="$SNAPSHOT.tmp.$$"
cp "$SOURCE_CHECKPOINT" "$tmp_snapshot"
mv "$tmp_snapshot" "$SNAPSHOT"

sha256sum "$SOURCE_CHECKPOINT" > "$TEST_ROOT/source_checkpoint.sha256"
sha256sum "$SNAPSHOT" > "$TEST_ROOT/snapshot_checkpoint.sha256"

source_hash="$(cut -d' ' -f1 "$TEST_ROOT/source_checkpoint.sha256")"
snapshot_hash="$(cut -d' ' -f1 "$TEST_ROOT/snapshot_checkpoint.sha256")"
if [ "$source_hash" != "$snapshot_hash" ]; then
    echo "ERROR: checkpoint changed while it was being copied; retry submission." >&2
    rm -f "$SNAPSHOT"
    exit 3
fi

if command -v git >/dev/null 2>&1; then
    git -C "$PROJECT_DIR" rev-parse HEAD > "$TEST_ROOT/git_commit.txt"
    git -C "$PROJECT_DIR" status --short > "$TEST_ROOT/git_status.txt"
fi

cd "$PROJECT_DIR"
job_id="$(sbatch --parsable \
    scripts/jeanzay/test_resunet_diagnostic.slurm \
    "$SNAPSHOT" "$TEST_ID" "$TEST_DATE_IDX" "$@")"

printf '%s\n' "$job_id" > "$TEST_ROOT/job_id.txt"

echo "job_id=$job_id"
echo "test_id=$TEST_ID"
echo "source_checkpoint=$SOURCE_CHECKPOINT"
echo "snapshot=$SNAPSHOT"
echo "sha256=$snapshot_hash"
echo "artifacts=$TEST_ROOT"
echo "stdout=$PROJECT_DIR/logs/jz_test_${job_id}.out"
echo "stderr=$PROJECT_DIR/logs/jz_test_${job_id}.err"
