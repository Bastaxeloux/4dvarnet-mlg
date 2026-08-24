#!/bin/bash
#SBATCH --job-name=sst_stats_pub
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=logs/jz_stats_%j.out
#SBATCH --error=logs/jz_stats_%j.err

set -euo pipefail

PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
DATA_ROOT="${CROSCIM_DATA_ROOT:-${SCRATCH:?SCRATCH is not defined}/croscim/data_sst_v2}"
MANIFEST_DIR="$WORK/croscim/publication/manifests"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh ""
mkdir -p logs "$MANIFEST_DIR"

python -c "import numpy,xarray,zarr,netCDF4,dask,yaml,tqdm; print('Statistics environment OK; zarr', zarr.__version__)"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Data root: $DATA_ROOT"
echo "Output directory: $MANIFEST_DIR"
echo "Start: $(date --iso-8601=seconds)"

python contrib/SST/compute_statistics.py "$DATA_ROOT" \
  --years 2017 2018 2019 2020 2021 2022 \
  --n-samples 1500 \
  --seed 20260821 \
  --require-complete-years \
  --output-yaml "$MANIFEST_DIR/norm_stats_2017_2022_v2.yaml" \
  --output-txt "$MANIFEST_DIR/norm_stats_2017_2022_v2.txt" \
  --manifest "$MANIFEST_DIR/norm_stats_sample_2017_2022_v2.yaml"

sha256sum \
  "$MANIFEST_DIR/norm_stats_2017_2022_v2.yaml" \
  "$MANIFEST_DIR/norm_stats_sample_2017_2022_v2.yaml" \
  > "$MANIFEST_DIR/norm_stats_2017_2022_v2.sha256"

echo "End: $(date --iso-8601=seconds)"
