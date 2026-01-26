#!/bin/bash
set -e

# Liste ordonnée des années à traiter (dans CET ordre)
YEARS_LIST=(2019 2018 2017 2016 2015 2014 2013 2011 2010 2025)

declare -A YEARS
YEARS[2024]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2024_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2023]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2023_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2022]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2022_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2021]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2021_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2020]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2020_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2019]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2019_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2018]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2018_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2017]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2017_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2016]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2016_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2015]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2015_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2014]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2014_GBL_0.05_REAN_4_production_test.sqfs"
YEARS[2013]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2013_GBL_0.05_REAN_4_production_test.sqfs"

YEARS[2011]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2011_GBL_0.05_REAN_3_production_test.sqfs"
YEARS[2010]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2010_GBL_0.05_REAN_3_production_test.sqfs"

YEARS[2025]="/net/isilon/ifs/arch/home/sstdev/Projects/C3S/GBL_0.05_REAN/Tar_files/final_production_v1/L4_all_2025_GBL_0.05_REAN_4_production_test.sqfs"


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NB_CORES=8
REF_FILE="/dmidata/projects/4dvarnet/reference_ascii.asc"

echo "TRAITEMENT DE ${#YEARS_LIST[@]} ANNEES (${NB_CORES} coeurs)"
echo "Ordre: ${YEARS_LIST[@]}"
echo ""

for YEAR in "${YEARS_LIST[@]}"; do
    SQSH="${YEARS[$YEAR]}"
    EXTRACT_DIR="/dmidata/projects/4dvarnet/squash_${YEAR}_extract"
    OUTPUT_DIR="/dmidata/projects/4dvarnet/Gefion/data_${YEAR}"

    echo ""
    echo "--------------- ANNEE $YEAR ---------------"
    echo "Archive: $SQSH"
    echo ""

    # Vérifier si déjà traité (check zarr x1, x3, x10 dans le même dossier)
    if [ -d "$OUTPUT_DIR" ]; then
        X1_COUNT=$(find "$OUTPUT_DIR" -type d -name "*_x1.zarr" 2>/dev/null | wc -l)
        X3_COUNT=$(find "$OUTPUT_DIR" -type d -name "*_x3.zarr" 2>/dev/null | wc -l)
        X10_COUNT=$(find "$OUTPUT_DIR" -type d -name "*_x10.zarr" 2>/dev/null | wc -l)
        if [ "$X1_COUNT" -ge 365 ] && [ "$X3_COUNT" -ge 365 ] && [ "$X10_COUNT" -ge 365 ]; then
            echo "Données Zarr déjà présentes (x1: $X1_COUNT, x3: $X3_COUNT, x10: $X10_COUNT) - SKIP"
            continue
        fi
    fi

    if [ ! -f "$SQSH" ]; then
        echo "ERREUR: Archive introuvable: $SQSH"
        exit 1
    fi

    echo "1/5 Extraction du squash..."
    if [ -d "$EXTRACT_DIR" ]; then
        DIR_COUNT=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        if [ "$DIR_COUNT" -eq 365 ] || [ "$DIR_COUNT" -eq 366 ]; then
            echo "Extraction deja complete ($DIR_COUNT dossiers), skip"
        else
            echo "Extraction incomplete ($DIR_COUNT dossiers), re-extraction..."
            rm -rf "$EXTRACT_DIR"
            unsquashfs -p $NB_CORES -d "$EXTRACT_DIR" "$SQSH"
        fi
    else
        unsquashfs -p $NB_CORES -d "$EXTRACT_DIR" "$SQSH"
    fi
    echo "OK"

    echo ""

    echo "2/5 Verification des fichiers..."
    python3 "$SCRIPT_DIR/verif_fichiers.py" "$YEAR"
    echo ""

    echo "3/5 Correction fichiers manquants..."
    python3 "$SCRIPT_DIR/ajout_ascii_manquant.py" "$YEAR"
    echo ""

    echo "4/5 Conversion => Zarr..."
    mkdir -p "$OUTPUT_DIR"
    python3 "$SCRIPT_DIR/converter.py" $YEAR --parallel $NB_CORES --zarr-output-dir "$OUTPUT_DIR"
    echo ""

    echo "5/5 Calcul resolutions x3 et x10 (Zarr)..."
    python3 "$SCRIPT_DIR/compute_res_yearly.py" $YEAR --parallel $NB_CORES --save-format zarr --output-dir "$OUTPUT_DIR"
    echo ""

    echo "Nettoyage extraction..."
    echo "Suppression extraction ($EXTRACT_DIR)..."
    du -sh "$EXTRACT_DIR"
    find "$EXTRACT_DIR" -delete
    echo "OK - Espace libere"
    echo ""

    OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    X1_COUNT=$(find "$OUTPUT_DIR" -type d -name "*_x1.zarr" 2>/dev/null | wc -l)
    X3_COUNT=$(find "$OUTPUT_DIR" -type d -name "*_x3.zarr" 2>/dev/null | wc -l)
    X10_COUNT=$(find "$OUTPUT_DIR" -type d -name "*_x10.zarr" 2>/dev/null | wc -l)

    echo "--------------- ANNEE $YEAR TERMINEE ---------------"
    echo "Dossier: $OUTPUT_DIR/ ($OUTPUT_SIZE)"
    echo "Fichiers Zarr: x1=$X1_COUNT, x3=$X3_COUNT, x10=$X10_COUNT"
    echo ""
done

echo ""
echo "FINITOOOOOOOO!"


# Pour lancer ce script:
# chmod +x process_all_years.sh
# nohup ./process_all_years.sh > log_$(date +%Y%m%d).log 2>&1 &
# echo $! > process.pid
# tail -f log_*.log

# Pour arreter en cours :
# kill $(cat process.pid)