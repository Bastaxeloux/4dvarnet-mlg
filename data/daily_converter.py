import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc
import subprocess
import shutil
from datetime import datetime, timedelta

def read_reference_netcdf(netcdf_path):
    """
    Lit la grille et les variables d'intérêt du NetCDF de référence.
    Retourne lon, lat, time, analysed_st, analysis_error, sea_ice_fraction
    """
    ds = xr.open_dataset(netcdf_path)
    lon = ds.lon.values
    lat = ds.lat.values
    time = ds.time.values if 'time' in ds else None
    analysed_st = ds['analysed_st'].values if 'analysed_st' in ds else None
    analysis_error = ds['analysis_error'].values if 'analysis_error' in ds else None
    sea_ice_fraction = ds['sea_ice_fraction'].values if 'sea_ice_fraction' in ds else None
    ds.close()
    return lon, lat, time, analysed_st, analysis_error, sea_ice_fraction

def read_ascii_file(asc_path):
    """
    Lit un fichier .asc et retourne les données sous forme d'array numpy
    """
    with open(asc_path, 'r') as f:
        lines = f.readlines()
    data_lines = lines[3:]
    data = [list(map(float, line.strip().split())) for line in data_lines]
    data = np.array(data)
    # Remplace les valeurs manquantes par np.nan
    data[data == 99.0] = np.nan
    data[data == 999.0] = np.nan
    return data

def read_all_satellite_files(day_dir, day_name):
    """
    Lit les 8 fichiers satellites (av et std pour 4 capteurs)
    Retourne un dict {var_name: data}
    """
    patterns = {
        'aasti_av': f'{day_name}_aasti_ist_l2p_av.asc',
        'aasti_std': f'{day_name}_aasti_ist_l2p_std_av.asc',
        'avhrr_av': f'{day_name}_avhrr_c3s_l3u_av.asc',
        'avhrr_std': f'{day_name}_avhrr_c3s_l3u_std_av.asc',
        'pmw_av': f'{day_name}_pmw_cci_l2p_av.asc',
        'pmw_std': f'{day_name}_pmw_cci_l2p_std_av.asc',
        'slstr_av': f'{day_name}_slstr_c3s_l3u_av.asc',
        'slstr_std': f'{day_name}_slstr_c3s_l3u_std_av.asc',
    }
    data = {}
    for var, fname in patterns.items():
        asc_path = day_dir / fname
        data[var] = read_ascii_file(asc_path) if asc_path.exists() else None
    return data

def read_extra_ascii_files(day_dir, day_name):
    """
    Lit surfmask et oi pour le jour
    """
    surfmask = read_ascii_file(day_dir / f'surfmask_{day_name}.asc')
    oi = read_ascii_file(day_dir / f'oi_{day_name}.asc')
    return surfmask, oi

def squeeze_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[0] > 1:
        # Prend le premier pas de temps
        return arr[0]
    return arr

def create_daily_dataset(lon, lat, time, sat_data, surfmask, oi, analysed_st, analysis_error, sea_ice_fraction):
    """
    Crée un xarray.Dataset avec toutes les variables pour un jour
    """
    data_vars = {}
    # Satellites
    for var, arr in sat_data.items():
        if arr is not None:
            data_vars[var] = (['lat', 'lon'], arr.astype(np.float32))
    # Extra ASCII
    data_vars['surfmask'] = (['lat', 'lon'], surfmask.astype(np.float32))
    data_vars['oi'] = (['lat', 'lon'], oi.astype(np.float32))
    # NetCDF ref (squeeze si besoin)
    if analysed_st is not None:
        data_vars['analysed_st'] = (['lat', 'lon'], squeeze_2d(analysed_st).astype(np.float32))
    if analysis_error is not None:
        data_vars['analysis_error'] = (['lat', 'lon'], squeeze_2d(analysis_error).astype(np.float32))
    if sea_ice_fraction is not None:
        data_vars['sea_ice_fraction'] = (['lat', 'lon'], squeeze_2d(sea_ice_fraction).astype(np.float32))
    ds = xr.Dataset(
        data_vars,
        coords={
            'lon': (['lon'], lon.astype(np.float32)),
            'lat': (['lat'], lat.astype(np.float32)),
            'time': time if time is not None else [0]
        }
    )
    return ds

def save_daily_dataset(ds, output_path, fmt='netcdf', compression_level=6, force_overwrite=False):
    """
    Sauvegarde le dataset au format NetCDF, Zarr, ou les deux avec compression
    """
    saved_formats = []
    
    # NetCDF
    if fmt in ['netcdf', 'both']:
        nc_path = output_path.with_suffix('.nc')
        if not nc_path.exists() or force_overwrite:
            encoding = {var: {"zlib": True, "complevel": compression_level, "dtype": "float32"} 
                       for var in ds.data_vars}
            try:
                ds.to_netcdf(nc_path, format='NETCDF4', engine='netcdf4', encoding=encoding)
            except Exception:
                ds.to_netcdf(nc_path, engine='h5netcdf', encoding=encoding)
            saved_formats.append('NetCDF')
    
    # Zarr
    if fmt in ['zarr', 'both']:
        zarr_path = output_path.with_suffix('.zarr')
        if not zarr_path.exists() or force_overwrite:
            import zarr
            # Nettoyer dossier zarr existant
            if zarr_path.exists():
                shutil.rmtree(zarr_path)
            encoding = {var: {"compressor": zarr.Blosc(cname='zstd', clevel=compression_level, shuffle=2),
                             "dtype": "float32"} for var in ds.data_vars}
            ds.to_zarr(str(zarr_path), mode='w', consolidated=True, encoding=encoding)
            saved_formats.append('Zarr')
    
    ds.close()
    return saved_formats

def extract_sqsh_archive(sqsh_path, extract_dir):
    """Extrait une archive .sqsh"""
    sqsh_path = Path(sqsh_path)
    extract_dir = Path(extract_dir)
    
    if not sqsh_path.exists():
        raise FileNotFoundError(f"Archive introuvable: {sqsh_path}")
    
    print(f"Extraction: {sqsh_path.name}")
    
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    
    cmd = ["unsquashfs", "-d", str(extract_dir), str(sqsh_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    squashfs_root = extract_dir / "squashfs-root" if (extract_dir / "squashfs-root").exists() else extract_dir
    day_dirs = [d for d in squashfs_root.iterdir() if d.is_dir()]
    print(f"Extraction OK: {len(day_dirs)} dossiers")
    
    return squashfs_root

def get_year_from_path_or_ask(root_dir):
    """Détermine l'année à partir des dossiers ou demande à l'utilisateur"""
    root_dir = Path(root_dir)
    day_dirs = [d.name for d in root_dir.iterdir() if d.is_dir() and len(d.name) >= 8]
    
    if day_dirs:
        first_dir = sorted(day_dirs)[0]
        if first_dir[:4].isdigit():
            year = int(first_dir[:4])
            print(f"Année détectée: {year}")
            return year
    
    while True:
        try:
            year = int(input("Année (1982-2025): "))
            if 1982 <= year <= 2025:
                return year
            print("Année invalide")
        except ValueError:
            print("Nombre requis")

def check_existing_files(year, output_base_dir, day_range=None):
    """Vérifie quels fichiers NetCDF/Zarr existent déjà pour une année"""
    # Dossier spécifique à l'année
    output_dir = Path(output_base_dir) / str(year)
    
    # Détermine le nombre de jours dans l'année
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    max_days = 366 if is_leap else 365
    
    if day_range is None:
        day_range = (1, max_days)
    
    start_day, end_day = day_range
    existing_files = {}
    
    for day in range(start_day, end_day + 1):
        # Génère le nom du dossier pour ce jour
        start_date = datetime(year, 1, 1)
        target_date = start_date + timedelta(days=day - 1)
        day_name = f"{target_date.strftime('%Y%m%d')}12"
        
        nc_path = output_dir / f"{day_name}.nc"
        zarr_path = output_dir / f"{day_name}.zarr"
        
        existing_files[day_name] = {
            'nc_exists': nc_path.exists(),
            'zarr_exists': zarr_path.exists(),
            'nc_path': nc_path,
            'zarr_path': zarr_path
        }
    
    return existing_files

def process_one_day(day_dir, fmt='both', output_dir=None, compression_level=6, force_overwrite=False):
    """Traite un jour : lit les fichiers, crée et sauvegarde le dataset"""
    day_name = day_dir.name
    
    if output_dir is None:
        # Détermine l'année du jour pour créer le bon dossier
        year = day_name[:4] if len(day_name) >= 4 and day_name[:4].isdigit() else "unknown"
        output_dir = Path(f'/dmidata/users/malegu/data/daily_output/{year}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / day_name
    
    # Vérifie si besoin de traiter
    nc_exists = output_path.with_suffix('.nc').exists()
    zarr_exists = output_path.with_suffix('.zarr').exists()
    
    need_nc = fmt in ['netcdf', 'both'] and (force_overwrite or not nc_exists)
    need_zarr = fmt in ['zarr', 'both'] and (force_overwrite or not zarr_exists)
    
    if not need_nc and not need_zarr:
        return []
    
    # Lecture des données
    netcdf_files = list(day_dir.glob('*-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc'))
    if not netcdf_files:
        raise FileNotFoundError(f"NetCDF manquant: {day_dir}")
    
    ref_nc = netcdf_files[0]
    lon, lat, time, analysed_st, analysis_error, sea_ice_fraction = read_reference_netcdf(ref_nc)
    sat_data = read_all_satellite_files(day_dir, day_name)
    surfmask, oi = read_extra_ascii_files(day_dir, day_name)
    ds = create_daily_dataset(lon, lat, time, sat_data, surfmask, oi, analysed_st, analysis_error, sea_ice_fraction)
    
    return save_daily_dataset(ds, output_path, fmt=fmt, compression_level=compression_level, force_overwrite=force_overwrite)

def process_all_days(root_dir, fmt='both', compression_level=6):
    """Traite tous les jours d'un dossier"""
    from tqdm import tqdm
    import time
    
    root = Path(root_dir)
    day_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    
    print(f"Traitement: {len(day_dirs)} jours, format {fmt}")
    
    errors = 0
    created = 0
    start_time = time.time()
    
    for day_dir in tqdm(day_dirs, desc="Traitement", unit="jour"):
        try:
            saved_formats = process_one_day(day_dir, fmt=fmt, compression_level=compression_level)
            if saved_formats and len(saved_formats) > 0:
                created += 1
                tqdm.write(f"✓ {day_dir.name}: {', '.join(saved_formats)}")
        except Exception as e:
            errors += 1
            tqdm.write(f"✗ {day_dir.name}: {e}")
    
    elapsed = time.time() - start_time
    print(f"Terminé: {created} créés, {errors} erreurs en {elapsed/60:.1f}min")

def main():
    import sys
    
    if len(sys.argv) >= 2 and sys.argv[1] == '--extract':
        if len(sys.argv) < 3:
            print("Usage: --extract archive.sqsh [year] [day_range]")
            return
        sqsh_path = sys.argv[2]
        year = int(sys.argv[3]) if len(sys.argv) > 3 else None
        day_range_str = sys.argv[4] if len(sys.argv) > 4 else "1-365"
        process_year_from_sqsh(sqsh_path, year, day_range_str)
        
    elif len(sys.argv) >= 2 and sys.argv[1] == '--check':
        if len(sys.argv) < 3:
            print("Usage: --check year [day_range]")
            return
        year = int(sys.argv[2])
        day_range_str = sys.argv[3] if len(sys.argv) > 3 else "1-365"
        
        if "-" in day_range_str:
            start_day, end_day = map(int, day_range_str.split("-"))
        else:
            start_day = end_day = int(day_range_str)
        
        output_base_dir = Path('/dmidata/users/malegu/data/daily_output')
        existing = check_existing_files(year, output_base_dir, (start_day, end_day))
        
        total = len(existing)
        nc_count = sum(1 for v in existing.values() if v['nc_exists'])
        zarr_count = sum(1 for v in existing.values() if v['zarr_exists'])
        
        print(f"Année {year} | Jours {start_day}-{end_day} ({total} jours)")
        print(f"NetCDF: {nc_count}/{total} | Zarr: {zarr_count}/{total}")
        
        missing = [k for k, v in existing.items() if not v['nc_exists'] or not v['zarr_exists']]
        if missing:
            print(f"Manquants: {len(missing)}")
            if len(missing) <= 10:
                print(f"{', '.join(missing[:10])}")
        else:
            print("Complet !")
            
    elif len(sys.argv) >= 2 and sys.argv[1] == '--days':
        # Mode traitement de jours spécifiques: --days "1 15 32 45" process_id
        if len(sys.argv) < 4:
            print("Usage: --days 'day1 day2 day3...' process_id")
            return
        
        days_str = sys.argv[2]
        process_id = int(sys.argv[3])
        days_list = list(map(int, days_str.split()))
        
        root = Path('/dmidata/users/malegu/data/squashfs-root')
        day_dirs = {d.name: d for d in root.iterdir() if d.is_dir()}
        
        errors = 0
        created = 0
        
        for day in days_list:
            year = 2024
            start_date = datetime(year, 1, 1)
            target_date = start_date + timedelta(days=day - 1)
            day_name = f"{target_date.strftime('%Y%m%d')}12"
            
            if day_name in day_dirs:
                day_dir = day_dirs[day_name]
                try:
                    saved_formats = process_one_day(day_dir, fmt='both', compression_level=6)
                    if saved_formats:
                        created += 1
                        print(f"P{process_id}: {day_name}")
                except Exception as e:
                    errors += 1
                    print(f"P{process_id} ERROR: {day_name}")
        
        print(f"P{process_id}: {created} OK, {errors} ERR")
            
    elif len(sys.argv) >= 2 and sys.argv[1] == '--sequential':
        # Mode séquentiel simple avec tqdm
        if len(sys.argv) < 5:
            print("Usage: --sequential year start_day end_day")
            return
        
        year = int(sys.argv[2])
        start_day = int(sys.argv[3])
        end_day = int(sys.argv[4])
        
        from tqdm import tqdm
        
        squashfs_root = Path('/dmidata/users/malegu/data/squashfs-root')
        output_dir = Path(f'/dmidata/users/malegu/data/daily_output/{year}')
        output_dir.mkdir(exist_ok=True)
        day_dirs = {d.name: d for d in squashfs_root.iterdir() if d.is_dir()}
        
        # Boucle simple avec tqdm
        print(f"Vérification jours {start_day} à {end_day}")
        
        processed = 0
        errors = 0
        
        for day in tqdm(range(start_day, end_day + 1), desc="Traitement", unit="jour"):
            start_date = datetime(year, 1, 1)
            target_date = start_date + timedelta(days=day - 1)
            day_name = f"{target_date.strftime('%Y%m%d')}12"
            
            # Vérifier si les fichiers existent
            nc_file = output_dir / f"{day_name}.nc"
            zarr_file = output_dir / f"{day_name}.zarr"
            
            # Si les deux existent, on passe
            if nc_file.exists() and zarr_file.exists():
                continue
                
            # Vérifier que le dossier source existe
            if day_name not in day_dirs:
                continue
            
            try:
                # Traiter seulement ce qui manque - temporairement NetCDF seulement
                fmt = 'netcdf'
                if nc_file.exists():
                    continue  # Skip si NetCDF existe déjà
                
                saved_formats = process_one_day(day_dirs[day_name], fmt=fmt, compression_level=6)
                if saved_formats:
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                # Debug plus détaillé pour comprendre l'erreur Zarr
                if '.zgroup' in str(e):
                    zarr_path = output_dir / f"{day_name}.zarr"
                    tqdm.write(f"ERREUR ZARR J{day}: existe={zarr_path.exists()}")
                else:
                    tqdm.write(f"ERREUR J{day}: {str(e)[:60]}")
        
        print(f"Terminé: {processed} créés, {errors} erreurs")
            
    elif len(sys.argv) == 4:
        # Mode parallèle
        start_day, end_day, process_id = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        
        root = Path('/dmidata/users/malegu/data/squashfs-root')
        day_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        
        print(f"Processus {process_id}: jours {start_day}-{end_day}")
        
        errors = 0
        created = 0
        
        for i in range(start_day, end_day + 1):
            if i <= len(day_dirs):
                day_dir = day_dirs[i-1]
                try:
                    saved_formats = process_one_day(day_dir, fmt='both', compression_level=6)
                    if saved_formats and len(saved_formats) > 0:
                        created += 1
                        print(f"P{process_id}: {day_dir.name} -> {', '.join(saved_formats)}")
                except Exception as e:
                    errors += 1
                    print(f"P{process_id} ERREUR {day_dir.name}: {e}")
        
        print(f"P{process_id} terminé: {created} créés, {errors} erreurs")
    else:
        process_all_days('/dmidata/users/malegu/data/squashfs-root', fmt='both', compression_level=6)

if __name__ == '__main__':
    main()

def process_year_from_sqsh(sqsh_path, year=None, day_range_str="1-365", cleanup_after=True):
    """Pipeline: extraction .sqsh + traitement + nettoyage"""
    from tqdm import tqdm
    
    sqsh_path = Path(sqsh_path)
    base_dir = Path('/dmidata/users/malegu/data')
    
    print(f"Archive: {sqsh_path.name} | Plage: {day_range_str}")
    
    # Extraction
    extract_dir = base_dir / f"temp_extract_{sqsh_path.stem}"
    data_dir = extract_sqsh_archive(sqsh_path, extract_dir)
    
    if year is None:
        year = get_year_from_path_or_ask(data_dir)
    
    # Parse plage
    if "-" in day_range_str:
        start_day, end_day = map(int, day_range_str.split("-"))
    else:
        start_day = end_day = int(day_range_str)
    
    # Trouve les jours manquants
    output_base_dir = base_dir / "daily_output"
    existing_files = check_existing_files(year, output_base_dir, (start_day, end_day))
    output_dir = output_base_dir / str(year)
    day_dirs = {d.name: d for d in data_dir.iterdir() if d.is_dir()}
    days_to_process = []
    
    for day in range(start_day, end_day + 1):
        start_date = datetime(year, 1, 1)
        day_name = f"{(start_date + timedelta(days=day - 1)).strftime('%Y%m%d')}12"
        
        if day_name in day_dirs:
            file_info = existing_files.get(day_name, {'nc_exists': False, 'zarr_exists': False})
            if not file_info['nc_exists'] or not file_info['zarr_exists']:
                days_to_process.append(day_dirs[day_name])
    
    if not days_to_process:
        print("Rien à faire.")
        return
    
    # Traitement
    print(f"Traitement: {len(days_to_process)} jours")
    stats = {'nc_created': 0, 'zarr_created': 0, 'errors': 0}
    
    for day_dir in tqdm(days_to_process, desc="Conversion"):
        try:
            saved_formats = process_one_day(day_dir, fmt='both', output_dir=output_dir, compression_level=6)
            stats['nc_created'] += saved_formats.count('NetCDF')
            stats['zarr_created'] += saved_formats.count('Zarr')
        except Exception as e:
            stats['errors'] += 1
            tqdm.write(f"ERREUR {day_dir.name}: {e}")
    
    print(f"Créés: NC={stats['nc_created']}, Zarr={stats['zarr_created']}, Erreurs={stats['errors']}")
    
    # Nettoyage
    if cleanup_after:
        response = input("Supprimer fichiers temp ? (o/N): ").strip().lower()
        if response in ['o', 'oui', 'y', 'yes']:
            shutil.rmtree(extract_dir)
            print("Nettoyé.")
        else:
            print(f"Conservés: {extract_dir}")
    
    print("Terminé.")
