#!/bin/bash
#SBATCH --job-name=sst_stats_2017_2024
#SBATCH --account=cu_0026
#SBATCH --partition=defq
#SBATCH --output=logs/stats_%j.out
#SBATCH --error=logs/stats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_DIR="/dcai/users/guimae/4dvarnet-mlg/Croscim"
DATA_ROOT="/dcai/projects/cu_0026/data_sst"

cd "$PROJECT_DIR"
mkdir -p logs
source scripts/gefion/env.sh

for year in 2014 2015 2016; do
    if [ -d "$DATA_ROOT/data_$year" ]; then
        echo "ERREUR: $DATA_ROOT/data_$year existe encore."
        exit 1
    fi
done

for year in {2017..2024}; do
    if [ ! -d "$DATA_ROOT/data_$year" ]; then
        echo "ERREUR: dossier manquant: $DATA_ROOT/data_$year"
        exit 1
    fi

    expected=365
    if (( year % 400 == 0 || (year % 4 == 0 && year % 100 != 0) )); then
        expected=366
    fi

    for res in 1 3 10; do
        count=$(find "$DATA_ROOT/data_$year" -maxdepth 1 -type d \
            -name "*_x${res}.zarr" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$count" -lt "$expected" ]; then
            echo "ERREUR: data_$year x$res incomplet: $count/$expected"
            exit 1
        fi
    done
    echo "data_$year complet: $expected jours en x1/x3/x10"
done

echo "Calcul des statistiques sur 1000 fichiers x1 échantillonnés parmi 2017-2024"
python contrib/SST/compute_statistics.py "$DATA_ROOT"

echo "Résultats:"
cat contrib/SST/norm_stats.yaml
