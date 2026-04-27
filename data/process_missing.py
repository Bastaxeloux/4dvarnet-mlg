#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# Import des fonctions du daily_converter
from daily_converter import process_one_day

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 process_missing.py YEAR START_DAY [END_DAY]")
        sys.exit(1)
    
    year = int(sys.argv[1])
    start_day = int(sys.argv[2])
    end_day = int(sys.argv[3]) if len(sys.argv) > 3 else start_day
    
    # Dossiers
    squashfs_root = Path('/dmidata/users/malegu/data/squashfs-root')
    output_dir = Path(f'/dmidata/users/malegu/data/daily_output/{year}')
    output_dir.mkdir(exist_ok=True)
    
    # Créer le mapping des dossiers journaliers
    day_dirs = {d.name: d for d in squashfs_root.iterdir() if d.is_dir()}
    
    # Trouver les jours manquants dans la plage
    missing_days = []
    for day in range(start_day, end_day + 1):
        start_date = datetime(year, 1, 1)
        target_date = start_date + timedelta(days=day - 1)
        day_name = f"{target_date.strftime('%Y%m%d')}12"
        
        # Vérifier si les fichiers existent
        nc_file = output_dir / f"{day_name}_combined_sst.nc"
        zarr_file = output_dir / f"{day_name}_combined_sst.zarr"
        
        if not nc_file.exists() or not zarr_file.exists():
            if day_name in day_dirs:
                missing_days.append((day, day_name, day_dirs[day_name]))
    
    if not missing_days:
        print(f"Tous les fichiers existent pour les jours {start_day}-{end_day}")
        return
    
    print(f"Traitement de {len(missing_days)} jours manquants (jours {start_day}-{end_day})")
    
    # Traitement avec barre tqdm
    errors = 0
    for day_num, day_name, day_dir in tqdm(missing_days, desc="Conversion", unit="jour"):
        try:
            saved_formats = process_one_day(day_dir, fmt='both', compression_level=6)
            if saved_formats:
                tqdm.write(f"J{day_num}: {', '.join(saved_formats)}")
        except Exception as e:
            errors += 1
            tqdm.write(f"ERREUR J{day_num}: {e}")
    
    print(f"Terminé: {len(missing_days)-errors}/{len(missing_days)} OK, {errors} erreurs")

if __name__ == '__main__':
    main()
