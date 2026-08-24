#!/bin/bash
#SBATCH --job-name=sst_eval_prep
#SBATCH --account=cai@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/jz_eval_prep_%j.out
#SBATCH --error=logs/jz_eval_prep_%j.err

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: sbatch $0 RAW_DMI_OI_NETCDF YYYY-MM-DD" >&2
    exit 1
fi

RAW_OI_NETCDF="$1"
RAW_DATE="$2"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-${WORK:?WORK is not defined}/croscim/repo/Croscim}"
ARTIFACT_ROOT="${CROSCIM_ARTIFACT_ROOT:-$WORK/croscim/publication}"
DATA_ROOT="${CROSCIM_DATA_ROOT:-${SCRATCH:?SCRATCH is not defined}/croscim/data_sst_v2}"
PROTOCOL_ROOT="$ARTIFACT_ROOT/evaluation_protocol_v2"
NORMALIZATION="$ARTIFACT_ROOT/manifests/norm_stats_2017_2022_v2.yaml"

cd "$PROJECT_DIR"
source scripts/jeanzay/env.sh ""
mkdir -p logs "$PROTOCOL_ROOT" "$WORK/tmp"
export TMPDIR="$WORK/tmp"

python scripts/publication/prepare_protocol.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$PROTOCOL_ROOT" \
    --normalization "$NORMALIZATION" \
    --dmi-oi-units "${CROSCIM_DMI_OI_UNITS:-kelvin}" \
    --raw-netcdf "$RAW_OI_NETCDF" \
    --raw-date "$RAW_DATE"
