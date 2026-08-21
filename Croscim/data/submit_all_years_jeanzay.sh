#!/bin/bash

set -euo pipefail

: "${WORK:?WORK is not defined}"

YEARS=(2017 2018 2019 2020 2021 2022 2023 2024)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROOT="$WORK/croscim/logs"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_DIR"

stage_job_id="$(sbatch \
    --parsable \
    --output="$LOG_ROOT/stage_sqfs_%j.out" \
    --error="$LOG_ROOT/stage_sqfs_%j.err" \
    data/stage_sqfs_jeanzay.slurm "${YEARS[@]}")"

process_job_id="$(sbatch \
    --parsable \
    --dependency="afterok:$stage_job_id" \
    --output="$LOG_ROOT/preprocess_all_%j.out" \
    --error="$LOG_ROOT/preprocess_all_%j.err" \
    data/process_all_years_jeanzay.slurm "${YEARS[@]}")"

echo "SQFS staging -> job $stage_job_id"
echo "Sequential preprocessing 2017-2024 -> job $process_job_id"
echo "The preprocessing job will start only if staging succeeds."
