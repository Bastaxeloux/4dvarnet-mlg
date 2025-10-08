#!/bin/bash

# Lancement parallèle sur 24 cœurs
# 365 jours / 24 processus = ~15 jours par processus

PROCESSES=6
TOTAL_DAYS=365
DAYS_PER_PROCESS=$((TOTAL_DAYS / PROCESSES))

echo "Lancement de $PROCESSES processus parallèles"
echo "~$DAYS_PER_PROCESS jours par processus"

# Arrêt des processus existants
pkill -f "daily_converter" 2>/dev/null || true
sleep 2

# Lancement des 24 processus
for i in $(seq 1 $PROCESSES); do
    start_day=$(((i-1) * DAYS_PER_PROCESS + 1))
    
    if [ $i -eq $PROCESSES ]; then
        end_day=$TOTAL_DAYS
    else
        end_day=$((i * DAYS_PER_PROCESS))
    fi
    
    echo "Processus $i: jours $start_day à $end_day"
    
    nohup python3 daily_converter.py $start_day $end_day $i > logs/process_$i.log 2>&1 &
done

echo ""
echo "Tous les processus lancés"
echo "Surveillance: tail -f logs/process_*.log"
echo "Nombre actif: ps aux | grep daily_converter | grep -v grep | wc -l"
echo ""
echo "Progression globale (Ctrl+C pour arrêter):"
sleep 3

# Barre de progression tqdm-style
while true; do
    c=$(ls /dmidata/users/malegu/data/daily_output/*.nc 2>/dev/null | wc -l)
    filled=$((c * 50 / 365))
    bar=$(printf "%${filled}s" | tr ' ' '-')
    empty=$(printf "%$((50-filled))s" | tr ' ' ' ')
    printf "\r[%s%s] %d/365 (%d%%) " "$bar" "$empty" "$c" $((c*100/365))
    
    if [ $c -ge 365 ]; then
        echo -e "\nTerminé! Tous les fichiers créés."
        break
    fi
    
    sleep 5
done
