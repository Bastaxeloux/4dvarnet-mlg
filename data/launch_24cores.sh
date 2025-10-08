DEFAULT_YEAR=$(date +%Y)
DEFAULT_START_DAY=1
DEFAULT_END_DAY=365
DEFAULT_PROCESSES=6
DEFAULT_ROOT_DIR="/dmidata/users/malegu/data/squashfs-root"

show_help() {
    echo "Usage: $0 [YEAR] [START_DAY] [END_DAY] [PROCESSES] [SQSH_ARCHIVE]"
    echo "$0 2024 1 365 6 data.sqsh"
    echo ""
    exit 0
}

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

YEAR=${1:-$DEFAULT_YEAR}
START_DAY=${2:-$DEFAULT_START_DAY}
END_DAY=${3:-$DEFAULT_END_DAY}
PROCESSES=${4:-$DEFAULT_PROCESSES}
SQSH_ARCHIVE=$5

[[ "$YEAR" =~ ^[0-9]{4}$ ]] && [ "$YEAR" -ge 1982 ] && [ "$YEAR" -le 2025 ] || { echo "Année invalide: $YEAR"; exit 1; }
[ "$START_DAY" -ge 1 ] && [ "$START_DAY" -le 366 ] || { echo "START_DAY invalide: $START_DAY"; exit 1; }
[ "$END_DAY" -ge 1 ] && [ "$END_DAY" -le 366 ] || { echo "END_DAY invalide: $END_DAY"; exit 1; }
[ "$START_DAY" -le "$END_DAY" ] || { echo "START_DAY > END_DAY"; exit 1; }
[ "$PROCESSES" -ge 1 ] && [ "$PROCESSES" -le 48 ] || { echo "PROCESSES invalide: $PROCESSES"; exit 1; }
[ -z "$SQSH_ARCHIVE" ] || [ -f "$SQSH_ARCHIVE" ] || { echo "Archive introuvable: $SQSH_ARCHIVE"; exit 1; }

TOTAL_DAYS=$((END_DAY - START_DAY + 1))
DAYS_PER_PROCESS=$((TOTAL_DAYS / PROCESSES))

echo "Année $YEAR | Jours $START_DAY-$END_DAY | $PROCESSES processus"

if [ ! -z "$SQSH_ARCHIVE" ]; then
    echo "Extraction: $SQSH_ARCHIVE"
    python3 daily_converter.py --extract "$SQSH_ARCHIVE" "$YEAR" "$START_DAY-$END_DAY"
    exit 0
fi

ROOT_DIR="$DEFAULT_ROOT_DIR"
if [ "$YEAR" -ne $(date +%Y) ]; then
    YEAR_ROOT_DIR="/dmidata/users/malegu/data/year_$YEAR"
    [ -d "$YEAR_ROOT_DIR" ] && ROOT_DIR="$YEAR_ROOT_DIR"
fi

[ -d "$ROOT_DIR" ] || { echo "Dossier introuvable: $ROOT_DIR"; exit 1; }

pkill -f "daily_converter.*[0-9]" 2>/dev/null || true
sleep 1
mkdir -p logs
python3 daily_converter.py --sequential "$YEAR" "$START_DAY" "$END_DAY"
