#!/bin/bash
#SBATCH --job-name=prep_2017_2024
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=preprocess_all_%j.out
#SBATCH --error=preprocess_all_%j.err

set -euo pipefail

: "${WORK:?WORK is not defined}"
: "${SCRATCH:?SCRATCH is not defined}"

if [ "$#" -eq 0 ]; then
    YEARS=(2017 2018 2019 2020 2021 2022 2023 2024)
else
    YEARS=("$@")
fi

for year in "${YEARS[@]}"; do
    case "$year" in
        2017|2018|2019|2020|2021|2022|2023|2024) ;;
        *)
            echo "ERROR: unsupported publication year: $year (expected 2017-2024)"
            exit 1
            ;;
    esac
done

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
YEAR_SCRIPT="$PROJECT_DIR/data/process_year_jeanzay.slurm"

if [ ! -f "$YEAR_SCRIPT" ]; then
    echo "ERROR: annual preprocessing script not found: $YEAR_SCRIPT"
    exit 1
fi

cd "$PROJECT_DIR"

echo "Sequential Jean Zay preprocessing"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Years: ${YEARS[*]}"
echo "Start: $(date --iso-8601=seconds)"

for year in "${YEARS[@]}"; do
    echo
    echo "========== START $year =========="
    bash "$YEAR_SCRIPT" "$year"
    echo "========== COMPLETE $year =========="
done

echo "All requested years completed."
echo "End: $(date --iso-8601=seconds)"
