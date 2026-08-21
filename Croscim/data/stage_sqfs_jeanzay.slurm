#!/bin/bash
#SBATCH --job-name=stage_sqfs
#SBATCH --account=cai@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=stage_sqfs_%j.out
#SBATCH --error=stage_sqfs_%j.err

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: sbatch data/stage_sqfs_jeanzay.slurm YEAR [YEAR ...]"
    exit 1
fi

: "${WORK:?WORK is not defined}"
: "${SCRATCH:?SCRATCH is not defined}"
: "${STORE:?STORE is not defined}"

SOURCE_DIR="${SOURCE_DIR:-$STORE/croscim/sqfs}"
DEST_DIR="${DEST_DIR:-$SCRATCH/croscim/sqfs}"
LOG_ROOT="${LOG_ROOT:-$WORK/croscim/logs}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_LOG="$LOG_ROOT/stage_sqfs_${JOB_ID}.log"

mkdir -p "$DEST_DIR" "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

command -v rsync >/dev/null 2>&1 || {
    echo "ERROR: rsync is not available on this node."
    exit 1
}
command -v unsquashfs >/dev/null 2>&1 || {
    echo "ERROR: unsquashfs is not available on this node."
    exit 1
}

echo "Job ID: $JOB_ID"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo "Years: $*"
echo "Start: $(date --iso-8601=seconds)"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: source archive directory not found: $SOURCE_DIR"
    exit 1
fi

for year in "$@"; do
    case "$year" in
        2017|2018|2019|2020|2021|2022|2023|2024) ;;
        *)
            echo "ERROR: unsupported publication year: $year (expected 2017-2024)"
            exit 1
            ;;
    esac

    filename="L4_all_${year}_GBL_0.05_REAN_4_production_test.sqfs"
    source_file="$SOURCE_DIR/$filename"
    dest_file="$DEST_DIR/$filename"

    if [ ! -f "$source_file" ]; then
        echo "ERROR: source archive not found: $source_file"
        exit 1
    fi

    echo
    echo "Staging $year: $filename"
    rsync -ah --partial --append-verify --info=progress2 \
        "$source_file" "$DEST_DIR/"
    unsquashfs -s "$dest_file" >/dev/null
    stat --printf='Validated: %n (%s bytes)\n' "$dest_file"
done

echo "Staging complete."
echo "End: $(date --iso-8601=seconds)"
