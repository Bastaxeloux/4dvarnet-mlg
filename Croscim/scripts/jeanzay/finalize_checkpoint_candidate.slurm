#!/bin/bash
#SBATCH --job-name=sst_eval_candidate
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/jz_eval_candidate_%j.out
#SBATCH --error=logs/jz_eval_candidate_%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 PILOT_EVALUATION_ID" >&2
    exit 1
fi

PILOT_ID="$1"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
PROTOCOL_ROOT="${CROSCIM_PROTOCOL_ROOT:-$ARTIFACT_ROOT/evaluation_protocol_v2}"
PILOT_MANIFEST="$PROTOCOL_ROOT/manifests/pilot.json"
COASTAL_MASK="$PROTOCOL_ROOT/static/coastal_mask_50km.npz"
PILOT_ROOT="$ARTIFACT_ROOT/evaluations/$PILOT_ID"
RESULTS="$PILOT_ROOT/results"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh ""
mkdir -p logs "$RESULTS" "$WORK/tmp"
export TMPDIR="$WORK/tmp"
export MPLBACKEND=Agg

python -u scripts/publication/validate_exports.py \
    --manifest "$PILOT_MANIFEST" \
    --evaluation-root "$PILOT_ROOT" \
    --mode controlled \
    --coastal-mask "$COASTAL_MASK" \
    --output "$PILOT_ROOT/pilot_validation.json"

python -u scripts/publication/aggregate_metrics.py \
    --manifest "$PILOT_MANIFEST" \
    --evaluation-root "$PILOT_ROOT" \
    --mode controlled \
    --block-days 1 \
    --output-dir "$RESULTS"

python -u scripts/publication/select_qualitative_cases.py \
    --manifest "$PILOT_MANIFEST" \
    --evaluation-root "$PILOT_ROOT" \
    --coastal-mask "$COASTAL_MASK" \
    --mode controlled \
    --output "$RESULTS/patch_gallery.json" \
    --catalog "$RESULTS/patch_catalog.csv" \
    --n-per-slot 15

python -u scripts/publication/make_figures.py \
    --evaluation-root "$PILOT_ROOT" \
    --mode controlled \
    --kind gallery \
    --cases "$RESULTS/patch_gallery.json" \
    --output-dir "$RESULTS/patch_gallery"

python -u scripts/publication/make_figures.py \
    --evaluation-root "$PILOT_ROOT" \
    --mode controlled \
    --kind diagnostics \
    --summary "$RESULTS/metrics_summary.csv" \
    --daily-metrics "$RESULTS/metrics_daily_sufficient.csv" \
    --bootstrap-intervals "$RESULTS/bootstrap_intervals.csv" \
    --output-dir "$RESULTS/diagnostics"

touch "$RESULTS/candidate_postprocessing_complete"
echo "Candidate ready: $PILOT_ROOT"
