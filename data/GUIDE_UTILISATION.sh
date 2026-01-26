#!/bin/bash
# Script d'aide pour les nouvelles fonctionnalités du pipeline SST

echo "=== Create NETCDF day-by-day for every year ==="
echo ""

echo "1. You can verify if a file has already been extracted"
echo "   python3 daily_converter.py --check YEAR [begin_day-end_day]"

echo "2. You can extract the .SQSH"
echo "   python3 daily_converter.py --extract ARCHIVE.sqsh [YEAR] [begin_day-end_day]"
echo ""

echo "3. EVERYTHING"
echo "If you don't specify anything, it will extract, check and create the netcdf and zarr if necessary"
echo "   ./launch_24cores.sh [YEAR] [first_day] [end_day] [NB_PROCESSUS] [ARCHIVE_SQSH]"
echo ""

echo "4. MONITORING with :"
echo "   ./monitor.sh                                      # Vérification ponctuelle"
echo "   tail -f logs/process_*.log                        # Suivi en temps réel"
echo ""

echo "5. WORKFLOW :"
echo "   python3 daily_converter.py --check 1985"
echo "   ./launch_24cores.sh 1985 1 365 6 /path/to/2009.sqsh"
echo "   python3 daily_converter.py --check 1985"

echo "The output files are by default in :"
echo "  /dmidata/users/malegu/data/daily_output/"
