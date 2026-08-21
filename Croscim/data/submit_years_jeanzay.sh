#!/bin/bash

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: bash data/submit_years_jeanzay.sh YEAR [YEAR ...]"
    echo "Example: bash data/submit_years_jeanzay.sh 2022"
    exit 1
fi

: "${WORK:?WORK is not defined}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$WORK/croscim/logs"
mkdir -p "$LOG_ROOT"

cd "$PROJECT_DIR"

for year in "$@"; do
    case "$year" in
        2017|2018|2019|2020|2021|2022|2023|2024) ;;
        *)
            echo "ERROR: unsupported publication year: $year (expected 2017-2024)" >&2
            exit 1
            ;;
    esac

    job_id="$(sbatch \
        --parsable \
        --job-name="prep_${year}" \
        --output="$LOG_ROOT/preprocess_${year}_%j.out" \
        --error="$LOG_ROOT/preprocess_${year}_%j.err" \
        data/process_year_jeanzay.slurm "$year")"
    echo "$year -> job $job_id"
done

