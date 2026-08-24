#!/bin/bash
#SBATCH --job-name=sst_eval_final
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/jz_eval_final_%j.out
#SBATCH --error=logs/jz_eval_final_%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 TEST_EVALUATION_ID" >&2
    exit 1
fi

TEST_ID="$1"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
PROTOCOL_ROOT="${CROSCIM_PROTOCOL_ROOT:-$ARTIFACT_ROOT/evaluation_protocol_v2}"
TEST_MANIFEST="$PROTOCOL_ROOT/manifests/test.json"
FROZEN_PROTOCOL="$PROTOCOL_ROOT/frozen_protocol.json"
COASTAL_MASK="$PROTOCOL_ROOT/static/coastal_mask_50km.npz"
TEST_ROOT="$ARTIFACT_ROOT/evaluations/$TEST_ID"
RESULTS="$TEST_ROOT/results"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh ""
mkdir -p logs "$RESULTS" "$WORK/tmp"
export TMPDIR="$WORK/tmp"
export MPLBACKEND=Agg

for required in "$TEST_MANIFEST" "$FROZEN_PROTOCOL" "$COASTAL_MASK"; do
    if [ ! -s "$required" ]; then
        echo "ERROR: required final-evaluation artifact is missing: $required" >&2
        exit 2
    fi
done

python -u scripts/publication/validate_exports.py \
    --manifest "$TEST_MANIFEST" \
    --evaluation-root "$TEST_ROOT" \
    --mode controlled \
    --coastal-mask "$COASTAL_MASK" \
    --frozen-protocol "$FROZEN_PROTOCOL" \
    --output "$TEST_ROOT/final_validation.json"

python -u scripts/publication/aggregate_metrics.py \
    --manifest "$TEST_MANIFEST" \
    --evaluation-root "$TEST_ROOT" \
    --mode controlled \
    --output-dir "$RESULTS"

python -u scripts/publication/select_qualitative_cases.py \
    --manifest "$TEST_MANIFEST" \
    --evaluation-root "$TEST_ROOT" \
    --coastal-mask "$COASTAL_MASK" \
    --mode controlled \
    --output "$RESULTS/qualitative_cases.json"

python -u scripts/publication/make_figures.py \
    --evaluation-root "$TEST_ROOT" \
    --mode controlled \
    --cases "$RESULTS/qualitative_cases.json" \
    --summary "$RESULTS/metrics_summary.csv" \
    --daily-metrics "$RESULTS/metrics_daily_sufficient.csv" \
    --output-dir "$RESULTS/figures"

sha256sum "$RESULTS"/figures/* > "$RESULTS/figures.sha256"
touch "$RESULTS/postprocessing_complete"
echo "Final evaluation artifacts: $RESULTS"
