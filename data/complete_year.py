#!/usr/bin/env python3
"""
Script simple pour compléter une année de données SST manquantes
Usage: python3 complete_year.py 2024
"""

import sys
from pathlib import Path
from tqdm import tqdm
import time

# Import des fonctions du daily_converter
sys.path.append('.')
from daily_converter import (
    process_one_day, 
    check_existing_files,
    Path
)

def complete_year(year):
    """Complete tous les fichiers manquants pour une année"""
    
    # Vérification initiale
    print(f"Vérification année {year}...")
    output_base_dir = Path('/dmidata/users/malegu/data/daily_output')
    existing_files = check_existing_files(year, output_base_dir)
    
    # Trouve les jours manquants
    source_dir = Path('/dmidata/users/malegu/data/squashfs-root')
    available_days = {d.name: d for d in source_dir.iterdir() if d.is_dir()}
    
    missing_days = []
    for day_name, status in existing_files.items():
        if day_name in available_days:
            if not status['nc_exists'] or not status['zarr_exists']:
                missing_days.append((day_name, available_days[day_name]))
    
    total = len(existing_files)
    existing_nc = sum(1 for s in existing_files.values() if s['nc_exists'])
    existing_zarr = sum(1 for s in existing_files.values() if s['zarr_exists'])
    
    print(f"État actuel: NetCDF {existing_nc}/{total}, Zarr {existing_zarr}/{total}")
    print(f"À traiter: {len(missing_days)} jours")
    
    if not missing_days:
        print("Rien à faire !")
        return
    
    # Traitement avec tqdm
    created_nc = 0
    created_zarr = 0
    errors = 0
    
    for day_name, day_dir in tqdm(missing_days, desc=f"Année {year}", unit="jour"):
        try:
            saved_formats = process_one_day(
                day_dir, 
                fmt='both', 
                compression_level=6, 
                force_overwrite=False
            )
            
            if 'NetCDF' in saved_formats:
                created_nc += 1
            if 'Zarr' in saved_formats:
                created_zarr += 1
                
            if saved_formats:
                tqdm.write(f"✓ {day_name}: {', '.join(saved_formats)}")
                
        except Exception as e:
            errors += 1
            tqdm.write(f"✗ {day_name}: {e}")
    
    print(f"\nTerminé:")
    print(f"NetCDF créés: {created_nc}")
    print(f"Zarr créés: {created_zarr}")
    print(f"Erreurs: {errors}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 complete_year.py YEAR")
        sys.exit(1)
    
    year = int(sys.argv[1])
    complete_year(year)
