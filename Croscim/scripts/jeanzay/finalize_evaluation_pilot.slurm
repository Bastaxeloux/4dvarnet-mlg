#!/bin/bash
#SBATCH --job-name=sst_eval_pilot
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/jz_eval_pilot_%j.out
#SBATCH --error=logs/jz_eval_pilot_%j.err

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 TRAIN_RUN_ID PILOT_EVALUATION_ID" >&2
    exit 1
fi

RUN_ID="$1"
PILOT_ID="$2"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
PROTOCOL_ROOT="${CROSCIM_PROTOCOL_ROOT:-$ARTIFACT_ROOT/evaluation_protocol_v2}"
PILOT_MANIFEST="$PROTOCOL_ROOT/manifests/pilot.json"
PILOT_ROOT="$ARTIFACT_ROOT/evaluations/$PILOT_ID"
CHECKPOINT_MANIFEST="$ARTIFACT_ROOT/checkpoints/publication_frozen/$RUN_ID/publication_best.json"
COASTAL_MASK="$PROTOCOL_ROOT/static/coastal_mask_50km.npz"
FROZEN_PROTOCOL="$PROTOCOL_ROOT/frozen_protocol.json"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh
mkdir -p logs "$WORK/tmp"
export TMPDIR="$WORK/tmp"

for required in "$PILOT_MANIFEST" "$CHECKPOINT_MANIFEST" "$COASTAL_MASK"; do
    if [ ! -s "$required" ]; then
        echo "ERROR: required pilot artifact is missing: $required" >&2
        exit 2
    fi
done
if [ -e "$FROZEN_PROTOCOL" ]; then
    echo "ERROR: frozen protocol already exists: $FROZEN_PROTOCOL" >&2
    exit 2
fi

python -u scripts/publication/validate_exports.py \
    --manifest "$PILOT_MANIFEST" \
    --evaluation-root "$PILOT_ROOT" \
    --mode controlled \
    --coastal-mask "$COASTAL_MASK" \
    --output "$PILOT_ROOT/pilot_validation.json"

python -u scripts/publication/spatial_diagnostics.py \
    --manifest "$PILOT_MANIFEST" \
    --evaluation-root "$PILOT_ROOT" \
    --mode controlled \
    --output "$PILOT_ROOT/spatial_diagnostic.json"

python -u scripts/publication/freeze_protocol.py \
    --preparation "$PROTOCOL_ROOT/protocol_preparation.json" \
    --pilot-manifest "$PILOT_MANIFEST" \
    --pilot-evaluation-root "$PILOT_ROOT" \
    --pilot-validation "$PILOT_ROOT/pilot_validation.json" \
    --spatial-diagnostic "$PILOT_ROOT/spatial_diagnostic.json" \
    --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
    --project-root "$PROJECT_DIR" \
    --output "$FROZEN_PROTOCOL"

echo "Pilot accepted and protocol frozen: $FROZEN_PROTOCOL"
