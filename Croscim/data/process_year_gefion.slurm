#!/bin/bash
#SBATCH --job-name=preprocess_sst
#SBATCH --account=cu_0026
#SBATCH --output=preprocess_%x_%j.out
#SBATCH --error=preprocess_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch data/process_year_gefion.slurm YEAR"
    exit 1
fi

YEAR="$1"
ARCHIVE_DIR="${ARCHIVE_DIR:-/dcai/projects/cu_0026/data_sst/sqfs}"
DATA_ROOT="${DATA_ROOT:-/dcai/projects/cu_0026/data_sst}"
USER_ROOT="${USER_ROOT:-/dcai/projects/cu_0026/guimae/croscim}"
EXTRACT_ROOT="${EXTRACT_ROOT:-${USER_ROOT}/tmp}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NB_CORES="${NB_CORES:-${SLURM_CPUS_PER_TASK:-96}}"

EXTRACT_DIR="${EXTRACT_ROOT}/squash_${YEAR}_extract"
OUTPUT_DIR="${DATA_ROOT}/data_${YEAR}"
MISSING_DAYS_FILE="/tmp/missing_days_${YEAR}_${SLURM_JOB_ID:-manual}.txt"

mkdir -p logs
mkdir -p "$EXTRACT_ROOT"
RUN_LOG="logs/preprocess_${YEAR}_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Year: $YEAR"
echo "Archive dir: $ARCHIVE_DIR"
echo "Data root: $DATA_ROOT"
echo "User root: $USER_ROOT"
echo "Extract root: $EXTRACT_ROOT"
echo "Project dir: $PROJECT_DIR"
echo "Workers: $NB_CORES"
echo "Start: $(date)"

if ! command -v unsquashfs >/dev/null 2>&1; then
    echo "unsquashfs introuvable avant chargement des modules."
fi

module load GCC/12.3.0 Boost/1.82.0 snappy/1.1.10 GSL/2.7 Eigen/3.4.0 Python/3.11.3
source /dcai/projects/cu_0026/croscim_env/bin/activate

cd "$PROJECT_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

command -v unsquashfs >/dev/null 2>&1 || {
    echo "ERREUR: unsquashfs introuvable dans l'environnement Gefion."
    exit 1
}

if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "ERREUR: dossier archives introuvable: $ARCHIVE_DIR"
    exit 1
fi

shopt -s nullglob
ARCHIVES=("${ARCHIVE_DIR}/L4_all_${YEAR}_GBL_0.05_REAN_"*_production*.sqfs)
shopt -u nullglob

if [ "${#ARCHIVES[@]}" -eq 0 ]; then
    echo "ERREUR: aucune archive SQFS trouvée pour $YEAR dans $ARCHIVE_DIR"
    exit 1
fi

if [ "${#ARCHIVES[@]}" -gt 1 ]; then
    echo "ERREUR: plusieurs archives candidates pour $YEAR. Choix manuel requis:"
    printf '  %s\n' "${ARCHIVES[@]}"
    exit 1
fi

SQFS="${ARCHIVES[0]}"

if (( YEAR % 400 == 0 || (YEAR % 4 == 0 && YEAR % 100 != 0) )); then
    EXPECTED_DAYS=366
else
    EXPECTED_DAYS=365
fi

mkdir -p "$OUTPUT_DIR"

X1_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_x1.zarr" 2>/dev/null | wc -l | tr -d ' ')
X3_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_x3.zarr" 2>/dev/null | wc -l | tr -d ' ')
X10_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_x10.zarr" 2>/dev/null | wc -l | tr -d ' ')

if [ "$X1_COUNT" -ge "$EXPECTED_DAYS" ] && [ "$X3_COUNT" -ge "$EXPECTED_DAYS" ] && [ "$X10_COUNT" -ge "$EXPECTED_DAYS" ]; then
    echo "Données déjà complètes pour $YEAR: x1=$X1_COUNT x3=$X3_COUNT x10=$X10_COUNT"
    exit 0
fi

echo "Archive sélectionnée: $SQFS"
echo "Sortie: $OUTPUT_DIR"
echo "Extraction: $EXTRACT_DIR"

if [ -d "$EXTRACT_DIR" ]; then
    DAY_DIRS=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
    if [ "$DAY_DIRS" -eq "$EXPECTED_DAYS" ]; then
        echo "Extraction déjà complète ($DAY_DIRS jours), skip extraction."
    else
        echo "Extraction existante incomplète ($DAY_DIRS/$EXPECTED_DAYS), recréation."
        rm -rf "$EXTRACT_DIR"
        unsquashfs -p "$NB_CORES" -d "$EXTRACT_DIR" "$SQFS"
    fi
else
    unsquashfs -p "$NB_CORES" -d "$EXTRACT_DIR" "$SQFS"
fi

echo "1/4 Vérification des fichiers"
python3 data/verif_fichiers.py "$YEAR" \
    --source-dir "$EXTRACT_DIR" \
    --missing-days-file "$MISSING_DAYS_FILE"

echo "2/4 Correction des ASCII manquants"
python3 data/ajout_ascii_manquant.py "$YEAR" \
    --source-dir "$EXTRACT_DIR" \
    --missing-days-file "$MISSING_DAYS_FILE"

echo "3/4 Conversion x1 vers Zarr"
python3 data/converter.py "$YEAR" \
    --parallel "$NB_CORES" \
    --source-dir "$EXTRACT_DIR" \
    --zarr-output-dir "$OUTPUT_DIR"

echo "4/4 Calcul x3 et x10"
python3 data/compute_res_yearly.py "$YEAR" \
    --parallel "$NB_CORES" \
    --save-format zarr \
    --output-dir "$OUTPUT_DIR"

X1_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_x1.zarr" 2>/dev/null | wc -l | tr -d ' ')
X3_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_x3.zarr" 2>/dev/null | wc -l | tr -d ' ')
X10_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*_x10.zarr" 2>/dev/null | wc -l | tr -d ' ')

echo "Résumé final $YEAR: attendu=$EXPECTED_DAYS x1=$X1_COUNT x3=$X3_COUNT x10=$X10_COUNT"

if [ "$X1_COUNT" -lt "$EXPECTED_DAYS" ] || [ "$X3_COUNT" -lt "$EXPECTED_DAYS" ] || [ "$X10_COUNT" -lt "$EXPECTED_DAYS" ]; then
    echo "ERREUR: année incomplète, extraction conservée pour inspection: $EXTRACT_DIR"
    exit 1
fi

if [ "${KEEP_EXTRACT:-0}" != "1" ]; then
    echo "Nettoyage extraction: $EXTRACT_DIR"
    du -sh "$EXTRACT_DIR" || true
    rm -rf "$EXTRACT_DIR"
fi

echo "Terminé: $YEAR"
echo "End: $(date)"
