#!/bin/bash
#SBATCH --job-name=sst_eval_results
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/jz_eval_results_%j.out
#SBATCH --error=logs/jz_eval_results_%j.err

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch $0 EVALUATION_ID MANIFEST BLOCK_DAYS" >&2
    exit 1
fi

EVALUATION_ID="$1"
MANIFEST="$2"
BLOCK_DAYS="$3"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
PROTOCOL_ROOT="${CROSCIM_PROTOCOL_ROOT:-$ARTIFACT_ROOT/evaluation_protocol_v2}"
EVALUATION_ROOT="$ARTIFACT_ROOT/evaluations/$EVALUATION_ID"
RESULTS="$EVALUATION_ROOT/results"
COASTAL_MASK="$PROTOCOL_ROOT/static/coastal_mask_50km.npz"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh ""
mkdir -p logs "$RESULTS" "$WORK/tmp"
export TMPDIR="$WORK/tmp"
export MPLBACKEND=Agg

python -u scripts/publication/aggregate_metrics.py \
    --manifest "$MANIFEST" \
    --evaluation-root "$EVALUATION_ROOT" \
    --mode controlled \
    --block-days "$BLOCK_DAYS" \
    --output-dir "$RESULTS"

python -u scripts/publication/select_qualitative_cases.py \
    --manifest "$MANIFEST" \
    --evaluation-root "$EVALUATION_ROOT" \
    --coastal-mask "$COASTAL_MASK" \
    --mode controlled \
    --output "$RESULTS/patch_gallery.json" \
    --catalog "$RESULTS/patch_catalog.csv" \
    --n-per-slot 15

python -u scripts/publication/make_figures.py \
    --evaluation-root "$EVALUATION_ROOT" \
    --mode controlled \
    --kind gallery \
    --cases "$RESULTS/patch_gallery.json" \
    --output-dir "$RESULTS/patch_gallery"

python -u scripts/publication/make_figures.py \
    --evaluation-root "$EVALUATION_ROOT" \
    --mode controlled \
    --kind diagnostics \
    --summary "$RESULTS/metrics_summary.csv" \
    --daily-metrics "$RESULTS/metrics_daily_sufficient.csv" \
    --bootstrap-intervals "$RESULTS/bootstrap_intervals.csv" \
    --output-dir "$RESULTS/diagnostics"

python -u scripts/publication/make_figures.py \
    --evaluation-root "$EVALUATION_ROOT" \
    --mode controlled \
    --kind report \
    --output-dir "$RESULTS/summary"

touch "$RESULTS/postprocessing_complete"
echo "Results: $RESULTS"
