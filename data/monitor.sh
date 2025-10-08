#!/bin/bash

# Monitoring simple des 24 processus

echo "=== ÉTAT ==="
active=$(ps aux | grep "daily_converter.*[0-9]" | grep -v grep | wc -l)
echo "Processus actifs: $active/24"

echo ""
echo "=== FICHIERS CRÉÉS ==="
nc_count=$(ls /dmidata/users/malegu/data/daily_output/*.nc 2>/dev/null | wc -l)
echo "Fichiers NetCDF: $nc_count/365"

echo ""
echo "=== LOGS ==="
for i in {1..24}; do
    # Cherche d'abord dans logs/, puis dans le répertoire current
    if [ -f "logs/process_$i.log" ]; then
        last_line=$(tail -1 logs/process_$i.log 2>/dev/null)
    elif [ -f "process_$i.log" ]; then
        last_line=$(tail -1 process_$i.log 2>/dev/null)
    else
        last_line=""
    fi
    
    if [ ! -z "$last_line" ]; then
        echo "Processus $i: $last_line"
    fi
done | head -10

echo ""
echo "=== COMMANDES UTILES ==="
echo "Surveiller tous: tail -f logs/process_*.log (ou process_*.log pour l'actuel)"
echo "Arrêter tous: pkill -f daily_converter"
echo "Redémarrer: ./launch_24cores.sh"
