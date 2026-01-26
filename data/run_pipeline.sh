#!/bin/bash
# Script helper pour le pipeline SST

set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PIPELINE_SCRIPT="$SCRIPT_DIR/pipeline_yearly.py"

echo "=== PIPELINE SST HELPER ==="
echo ""

# Vérification que le script principal existe
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "ERREUR: $PIPELINE_SCRIPT non trouvé"
    exit 1
fi

# Fonction d'aide
show_help() {
    echo "Usage: $0 [ANNÉE] [OPTIONS]"
    echo ""
    echo "Exemples d'utilisation:"
    echo "  $0 2024                    # Traite toute l'année 2024"
    echo "  $0 1985 --days 1-100       # Traite les 100 premiers jours de 1985"
    echo "  $0 2020 --sqsh data.sqsh   # Extrait data.sqsh et traite 2020"
    echo "  $0 --check 2024            # Vérifie seulement les fichiers existants"
    echo ""
    echo "Options:"
    echo "  --days RANGE       Plage de jours (ex: 1-365, 50-150)"
    echo "  --sqsh FICHIER     Archive .sqsh à extraire"
    echo "  --check            Mode vérification seulement"
    echo "  --no-cleanup       Pas de nettoyage interactif"
    echo "  --compression N    Niveau compression (1-9, défaut: 6)"
    echo "  --help, -h         Affiche cette aide"
}

# Analyse des arguments
YEAR=""
DAYS_RANGE="1-365"
SQSH_FILE=""
CHECK_ONLY=false
NO_CLEANUP=""
COMPRESSION="6"

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --days)
            DAYS_RANGE="$2"
            shift 2
            ;;
        --sqsh)
            SQSH_FILE="$2"
            shift 2
            ;;
        --no-cleanup)
            NO_CLEANUP="--no-cleanup"
            shift
            ;;
        --compression)
            COMPRESSION="$2"
            shift 2
            ;;
        [0-9]*)
            if [ -z "$YEAR" ]; then
                YEAR="$1"
            else
                echo "ERREUR: Année déjà spécifiée ($YEAR)"
                exit 1
            fi
            shift
            ;;
        *)
            echo "ERREUR: Option inconnue: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validation de l'année
if [ -z "$YEAR" ]; then
    echo "ERREUR: Année requise"
    show_help
    exit 1
fi

if [ "$YEAR" -lt 1982 ] || [ "$YEAR" -gt 2025 ]; then
    echo "ERREUR: Année doit être entre 1982 et 2025"
    exit 1
fi

# Validation du fichier .sqsh si fourni
if [ ! -z "$SQSH_FILE" ] && [ ! -f "$SQSH_FILE" ]; then
    echo "ERREUR: Fichier .sqsh introuvable: $SQSH_FILE"
    exit 1
fi

# Vérification des dépendances
echo "Vérification des dépendances..."

# unsquashfs (si fichier .sqsh fourni)
if [ ! -z "$SQSH_FILE" ] && ! command -v unsquashfs &> /dev/null; then
    echo "ERREUR: unsquashfs non trouvé. Installez squashfs-tools:"
    echo "  sudo apt install squashfs-tools  # Ubuntu/Debian"
    echo "  sudo yum install squashfs-tools  # CentOS/RHEL"
    exit 1
fi

# Python et modules requis
if ! python3 -c "import numpy, xarray, netCDF4, tqdm, zarr" 2>/dev/null; then
    echo "ERREUR: Modules Python manquants. Installez:"
    echo "  pip install numpy xarray netcdf4 tqdm zarr"
    exit 1
fi

echo "Dépendances OK"
echo ""

# Mode vérification seulement
if [ "$CHECK_ONLY" = true ]; then
    echo "=== MODE VÉRIFICATION ==="
    echo "Année: $YEAR"
    echo "Plage de jours: $DAYS_RANGE"
    
    # Appel du script Python en mode dry-run (on pourrait ajouter cette option)
    python3 "$PIPELINE_SCRIPT" "$YEAR" --days-range "$DAYS_RANGE" --dry-run 2>/dev/null || {
        echo "Simulation du pipeline pour vérification..."
        python3 -c "
from pathlib import Path
import sys
sys.path.append('$SCRIPT_DIR')
from pipeline_yearly import YearlySSTPipeline

pipeline = YearlySSTPipeline($YEAR)
day_range = tuple(map(int, '$DAYS_RANGE'.split('-'))) if '-' in '$DAYS_RANGE' else (int('$DAYS_RANGE'), int('$DAYS_RANGE'))
status = pipeline.check_existing_files(day_range)
"
    }
    exit 0
fi

# Construction de la commande
CMD_ARGS=("$YEAR" "--days-range" "$DAYS_RANGE" "--compression" "$COMPRESSION")

if [ ! -z "$SQSH_FILE" ]; then
    CMD_ARGS+=("--sqsh-archive" "$SQSH_FILE")
fi

if [ ! -z "$NO_CLEANUP" ]; then
    CMD_ARGS+=("$NO_CLEANUP")
fi

# Affichage du résumé
echo "=== PARAMÈTRES ==="
echo "Année: $YEAR"
echo "Plage de jours: $DAYS_RANGE"
echo "Archive .sqsh: ${SQSH_FILE:-"Aucune (utilise dossier existant)"}"
echo "Compression: $COMPRESSION"
echo "Nettoyage automatique: ${NO_CLEANUP:+Non}"
echo ""

# Confirmation
read -p "Continuer ? (o/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[OoYy]$ ]]; then
    echo "Annulé."
    exit 0
fi

echo ""
echo "=== DÉBUT TRAITEMENT ==="
echo "Commande: python3 $PIPELINE_SCRIPT ${CMD_ARGS[*]}"
echo ""

# Exécution
exec python3 "$PIPELINE_SCRIPT" "${CMD_ARGS[@]}"
