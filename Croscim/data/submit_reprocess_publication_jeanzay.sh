#!/bin/bash
# Submit one independent CPU preprocessing job per publication year.

set -euo pipefail

: "${WORK:?WORK is not defined}"
: "${SCRATCH:?SCRATCH is not defined}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${CROSCIM_REPROCESS_DATA_ROOT:-$SCRATCH/croscim/data_sst_v2}"
EXTRACT_ROOT="${CROSCIM_REPROCESS_EXTRACT_ROOT:-$SCRATCH/croscim/extract_v2}"
LOG_ROOT="${CROSCIM_REPROCESS_LOG_ROOT:-$WORK/croscim/logs/reprocess_v2}"
MANIFEST_ROOT="${CROSCIM_REPROCESS_MANIFEST_ROOT:-$WORK/croscim/manifests/reprocess_v2}"
NB_CORES="${CROSCIM_REPROCESS_WORKERS:-32}"
if [ "$#" -eq 0 ]; then
    YEARS=(2017 2018 2019 2020 2021 2022 2023 2024)
else
    YEARS=("$@")
fi

mkdir -p "$DATA_ROOT" "$EXTRACT_ROOT" "$LOG_ROOT" "$MANIFEST_ROOT"
cd "$PROJECT_DIR"

echo "Output: $DATA_ROOT"
echo "Workers per year: $NB_CORES"
JOB_IDS=()
for year in "${YEARS[@]}"; do
    case "$year" in
        2017|2018|2019|2020|2021|2022|2023|2024) ;;
        *) echo "ERROR: unsupported year: $year" >&2; exit 1 ;;
    esac

    job_id="$(sbatch --parsable \
        --job-name="prep_v2_${year}" \
        --output="$LOG_ROOT/preprocess_${year}_%j.out" \
        --error="$LOG_ROOT/preprocess_${year}_%j.err" \
        --export="ALL,DATA_ROOT=$DATA_ROOT,EXTRACT_ROOT=$EXTRACT_ROOT,LOG_ROOT=$LOG_ROOT,MANIFEST_ROOT=$MANIFEST_ROOT,NB_CORES=$NB_CORES" \
        data/process_year_jeanzay.slurm "$year")"
    JOB_IDS+=("$job_id")
    echo "$year -> job $job_id"
done

if [ "${#YEARS[@]}" -eq 8 ]; then
    dependency="$(IFS=:; echo "${JOB_IDS[*]}")"
    stats_job="$(sbatch --parsable \
        --dependency="afterok:$dependency" \
        scripts/jeanzay/compute_statistics_publication.slurm)"
    echo "statistics -> job $stats_job (after all annual jobs)"
fi
