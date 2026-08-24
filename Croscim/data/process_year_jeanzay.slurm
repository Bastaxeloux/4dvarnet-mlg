#!/bin/bash
#SBATCH --job-name=croscim_prep
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=preprocess_%x_%j.out
#SBATCH --error=preprocess_%x_%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch data/process_year_jeanzay.slurm YEAR"
    exit 1
fi

YEAR="$1"
case "$YEAR" in
    2017|2018|2019|2020|2021|2022|2023|2024) ;;
    *)
        echo "ERROR: unsupported publication year: $YEAR (expected 2017-2024)"
        exit 1
        ;;
esac

: "${WORK:?WORK is not defined}"
: "${SCRATCH:?SCRATCH is not defined}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$SCRATCH/croscim/sqfs}"
DATA_ROOT="${DATA_ROOT:-$SCRATCH/croscim/data_sst}"
EXTRACT_ROOT="${EXTRACT_ROOT:-$SCRATCH/croscim/extract}"
LOG_ROOT="${LOG_ROOT:-$WORK/croscim/logs}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$WORK/croscim/manifests}"
NB_CORES="${NB_CORES:-32}"
if [ "$NB_CORES" -gt "${SLURM_CPUS_PER_TASK:-40}" ]; then
    echo "ERROR: NB_CORES=$NB_CORES exceeds the allocated CPU count (${SLURM_CPUS_PER_TASK:-40})."
    exit 1
fi

EXTRACT_DIR="$EXTRACT_ROOT/squash_${YEAR}_extract"
OUTPUT_DIR="$DATA_ROOT/data_${YEAR}"
JOB_ID="${SLURM_JOB_ID:-manual}"
TMP_ROOT="${TMP_ROOT:-$SCRATCH/croscim/tmp/$JOB_ID}"
MISSING_DAYS_FILE="$TMP_ROOT/missing_days_${YEAR}.txt"
RUN_LOG="$LOG_ROOT/preprocess_${YEAR}_${JOB_ID}.log"
MANIFEST="$MANIFEST_ROOT/preprocess_${YEAR}_${JOB_ID}.txt"

mkdir -p "$DATA_ROOT" "$EXTRACT_ROOT" "$LOG_ROOT" "$MANIFEST_ROOT" "$TMP_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Job ID: $JOB_ID"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Year: $YEAR"
echo "Project: $PROJECT_DIR"
echo "Archives: $ARCHIVE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Extraction: $EXTRACT_DIR"
echo "Workers: $NB_CORES"
echo "Start: $(date --iso-8601=seconds)"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh

command -v unsquashfs >/dev/null 2>&1 || {
    echo "ERROR: unsquashfs is not available on this node."
    exit 1
}

python - <<'PY'
import sys

import dask
import netCDF4
import numpy as np
import xarray as xr
import zarr

major = int(zarr.__version__.split(".", 1)[0])
if major != 2:
    raise RuntimeError(f"Zarr 2 is required, found {zarr.__version__}")

print(f"python: {sys.executable}")
print(f"numpy: {np.__version__}")
print(f"xarray: {xr.__version__}")
print(f"zarr: {zarr.__version__}")
print(f"dask: {dask.__version__}")
print(f"netCDF4: {netCDF4.__version__}")
PY

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "ERROR: archive directory not found: $ARCHIVE_DIR"
    exit 1
fi

shopt -s nullglob
ARCHIVES=("$ARCHIVE_DIR/L4_all_${YEAR}_GBL_0.05_REAN_"*_production*.sqfs)
shopt -u nullglob

if [ "${#ARCHIVES[@]}" -eq 0 ]; then
    echo "ERROR: no SQFS archive found for $YEAR in $ARCHIVE_DIR"
    exit 1
fi

if [ "${#ARCHIVES[@]}" -gt 1 ]; then
    echo "ERROR: multiple SQFS archives found for $YEAR; select one explicitly:"
    printf '  %s\n' "${ARCHIVES[@]}"
    exit 1
fi

SQFS="${ARCHIVES[0]}"
unsquashfs -s "$SQFS" >/dev/null

if (( YEAR % 400 == 0 || (YEAR % 4 == 0 && YEAR % 100 != 0) )); then
    EXPECTED_DAYS=366
else
    EXPECTED_DAYS=365
fi

count_resolution() {
    local resolution="$1"
    find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -type f \
        -path "*_${resolution}.zarr/.zmetadata" 2>/dev/null | wc -l | tr -d ' '
}

mkdir -p "$OUTPUT_DIR"
X1_COUNT="$(count_resolution x1)"
X3_COUNT="$(count_resolution x3)"
X10_COUNT="$(count_resolution x10)"

{
    echo "job_id=$JOB_ID"
    echo "year=$YEAR"
    echo "node=${SLURM_NODELIST:-local}"
    echo "start=$(date --iso-8601=seconds)"
    echo "project_dir=$PROJECT_DIR"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "archive=$SQFS"
    stat --printf='archive_size=%s\narchive_mtime=%y\n' "$SQFS"
    echo "output_dir=$OUTPUT_DIR"
    echo "workers=$NB_CORES"
} > "$MANIFEST"

if [ "$X1_COUNT" -eq "$EXPECTED_DAYS" ] && [ "$X3_COUNT" -eq "$EXPECTED_DAYS" ] && [ "$X10_COUNT" -eq "$EXPECTED_DAYS" ]; then
    echo "Already complete: expected=$EXPECTED_DAYS x1=$X1_COUNT x3=$X3_COUNT x10=$X10_COUNT"
    python data/validate_preprocessed_year.py "$YEAR" \
        --data-root "$DATA_ROOT"
    echo "status=already_complete" >> "$MANIFEST"
    echo "end=$(date --iso-8601=seconds)" >> "$MANIFEST"
    exit 0
fi

echo "Selected archive: $SQFS"

if [ -d "$EXTRACT_DIR" ]; then
    DAY_DIRS="$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$DAY_DIRS" -eq "$EXPECTED_DAYS" ]; then
        echo "Extraction already complete ($DAY_DIRS days); skipping unsquashfs."
    else
        echo "Incomplete extraction ($DAY_DIRS/$EXPECTED_DAYS); rebuilding it."
        rm -rf "$EXTRACT_DIR"
        unsquashfs -p "$NB_CORES" -d "$EXTRACT_DIR" "$SQFS"
    fi
else
    unsquashfs -p "$NB_CORES" -d "$EXTRACT_DIR" "$SQFS"
fi

echo "1/4 Verify extracted files"
python data/verif_fichiers.py "$YEAR" \
    --source-dir "$EXTRACT_DIR" \
    --missing-days-file "$MISSING_DAYS_FILE"

echo "2/4 Add missing ASCII placeholders"
python data/ajout_ascii_manquant.py "$YEAR" \
    --source-dir "$EXTRACT_DIR" \
    --missing-days-file "$MISSING_DAYS_FILE"

echo "3/4 Convert x1 to Zarr"
python data/converter.py "$YEAR" \
    --parallel "$NB_CORES" \
    --source-dir "$EXTRACT_DIR" \
    --zarr-output-dir "$OUTPUT_DIR"

echo "4/4 Compute x3 and x10"
python data/compute_res_yearly.py "$YEAR" \
    --parallel "$NB_CORES" \
    --save-format zarr \
    --output-dir "$OUTPUT_DIR"

echo "Validate preprocessed fields"
python data/validate_preprocessed_year.py "$YEAR" \
    --data-root "$DATA_ROOT"

X1_COUNT="$(count_resolution x1)"
X3_COUNT="$(count_resolution x3)"
X10_COUNT="$(count_resolution x10)"

echo "Final summary: expected=$EXPECTED_DAYS x1=$X1_COUNT x3=$X3_COUNT x10=$X10_COUNT"
{
    echo "x1_count=$X1_COUNT"
    echo "x3_count=$X3_COUNT"
    echo "x10_count=$X10_COUNT"
} >> "$MANIFEST"

if [ "$X1_COUNT" -ne "$EXPECTED_DAYS" ] || [ "$X3_COUNT" -ne "$EXPECTED_DAYS" ] || [ "$X10_COUNT" -ne "$EXPECTED_DAYS" ]; then
    echo "ERROR: incomplete year; extraction retained for inspection: $EXTRACT_DIR"
    echo "status=incomplete" >> "$MANIFEST"
    echo "end=$(date --iso-8601=seconds)" >> "$MANIFEST"
    exit 1
fi

if [ "${KEEP_EXTRACT:-0}" != "1" ]; then
    echo "Removing successful extraction: $EXTRACT_DIR"
    du -sh "$EXTRACT_DIR" || true
    rm -rf "$EXTRACT_DIR"
fi

rm -rf "$TMP_ROOT"
echo "status=complete" >> "$MANIFEST"
echo "end=$(date --iso-8601=seconds)" >> "$MANIFEST"
echo "Completed: $YEAR"
echo "End: $(date --iso-8601=seconds)"
