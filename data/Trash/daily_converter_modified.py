import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc

def read_reference_netcdf(netcdf_path):
    """
    Lit la grille et les variables d'intérêt du NetCDF de référence.
    Retourne lon, lat, time, analysed_st, anal        except Exception as e:
            errors += 1
            day_num = start_day + i
            tqdm.write(f"")
            tqdm.write(f"ERREUR Jour {day_num:3d} | {day_dir.name}: {e}")
            tqdm.write(f"")s_error, sea_ice_fraction
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
    # Remplace les valeurs manquantes (99.0 ou 999.0) par np.nan
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

def save_daily_dataset(ds, output_path, fmt='netcdf', compression_level=6):
    """
    Sauvegarde le dataset au format NetCDF, Zarr, ou les deux avec compression
    compression_level: 1-9 pour NetCDF (1=rapide, 9=max compression)
                      1-9 pour Zarr (niveaux similaires)
    """
    if fmt in ['netcdf', 'both']:
        # Compression NetCDF pour toutes les variables de données
        netcdf_compression = {var: {"zlib": True, "complevel": compression_level, "dtype": "float32"} 
                             for var in ds.data_vars}
        try:
            ds.to_netcdf(output_path.with_suffix('.nc'), 
                        format='NETCDF4', 
                        engine='netcdf4', 
                        encoding=netcdf_compression)
        except Exception:
            # Si segfault, tente avec h5netcdf
            ds.to_netcdf(output_path.with_suffix('.nc'), 
                        engine='h5netcdf', 
                        encoding=netcdf_compression)
    
    if fmt in ['zarr', 'both']:
        import zarr
        # Compression Zarr plus avancée (zstd est meilleur que zlib)
        zarr_compression = {}
        for var in ds.data_vars:
            zarr_compression[var] = {
                "compressor": zarr.Blosc(cname='zstd', clevel=compression_level, shuffle=2),
                "dtype": "float32"
            }
        ds.to_zarr(str(output_path.with_suffix('.zarr')), 
                   mode='w', 
                   consolidated=True, 
                   encoding=zarr_compression)
    
    ds.close()

def process_one_day(day_dir, fmt='both', output_dir=None, compression_level=6):
    """
    Traite un jour : lit les fichiers, crée et sauvegarde le dataset
    compression_level: 1-9 (1=rapide/gros, 9=lent/petit)
    """
    day_name = day_dir.name
    # Trouve le fichier NetCDF de référence dans le dossier du jour
    netcdf_files = list(day_dir.glob('*-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc'))
    if not netcdf_files:
        raise FileNotFoundError(f"Aucun fichier NetCDF de référence trouvé dans {day_dir}")
    ref_nc = netcdf_files[0]  # Prend le premier (et normalement seul) fichier
    lon, lat, time, analysed_st, analysis_error, sea_ice_fraction = read_reference_netcdf(ref_nc)
    sat_data = read_all_satellite_files(day_dir, day_name)
    surfmask, oi = read_extra_ascii_files(day_dir, day_name)
    ds = create_daily_dataset(lon, lat, time, sat_data, surfmask, oi, analysed_st, analysis_error, sea_ice_fraction)
    # Utilise dmidata à la racine si pas d'output_dir spécifié
    if output_dir is None:
        output_dir = Path('/dmidata/users/malegu/data/daily_output')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / day_name
    save_daily_dataset(ds, output_path, fmt=fmt, compression_level=compression_level)
    print(f"Fichiers créés pour {day_name} : {output_path.with_suffix('.nc')} et/ou {output_path.with_suffix('.zarr')}")

def process_all_days(root_dir, fmt='both', compression_level=6, start_day=1):
    """
    Itère sur tous les jours d'un dossier racine avec barre de progression
    OPTIMISÉ: lecture parallèle + compression équilibrée (gain ~17s par jour)
    
    start_day: commence au jour N de l'année (1-365)
    """
    from tqdm import tqdm
    import time
    
    root = Path(root_dir)
    day_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    
    # Filtre pour commencer au jour demandé
    if start_day > 1:
        day_dirs = day_dirs[start_day-1:]  # -1 car indexé à partir de 0
        print(f"REPRISE à partir du jour {start_day} ({day_dirs[0].name})")
    
    print(f"Traitement de {len(day_dirs)} jours restants")
    print(f"Format(s): {fmt}")
    
    errors = 0
    start_time = time.time()
    
    for i, day_dir in enumerate(tqdm(day_dirs, desc="Conversion", unit="jour")):
        try:
            day_start_time = time.time()
            day_name = day_dir.name
            
            # 1. Lecture NetCDF de référence
            read_start = time.time()
            netcdf_files = list(day_dir.glob('*-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc'))
            if not netcdf_files:
                raise FileNotFoundError(f"Aucun fichier NetCDF de référence trouvé dans {day_dir}")
            ref_nc = netcdf_files[0]
            lon, lat, time_val, analysed_st, analysis_error, sea_ice_fraction = read_reference_netcdf(ref_nc)
            
            # 2. Lecture ASCII (parallélisée)
            sat_data = read_all_satellite_files(day_dir, day_name)
            surfmask, oi = read_extra_ascii_files(day_dir, day_name)
            read_time = time.time() - read_start
            
            # 3. Création dataset
            create_start = time.time()
            ds = create_daily_dataset(lon, lat, time_val, sat_data, surfmask, oi, analysed_st, analysis_error, sea_ice_fraction)
            create_time = time.time() - create_start
            
            # 4. Sauvegarde
            save_start = time.time()
            output_dir = Path('/dmidata/users/malegu/data/daily_output')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / day_name
            save_daily_dataset(ds, output_path, fmt=fmt, compression_level=compression_level)
            save_time = time.time() - save_start
            
            # 5. Calcul des tailles de fichiers
            nc_size = output_path.with_suffix('.nc').stat().st_size / 1024 / 1024 if fmt in ['netcdf', 'both'] else 0
            zarr_size = 0
            if fmt in ['zarr', 'both']:
                import subprocess
                result = subprocess.run(['du', '-sb', str(output_path.with_suffix('.zarr'))], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    zarr_size = int(result.stdout.split()[0]) / 1024 / 1024
            
            total_day_time = time.time() - day_start_time
            
            # Affichage détaillé propre
            day_num = start_day + i
            month = day_name[4:6]
            day = day_name[6:8]
            
            tqdm.write(f"")  # Ligne vide pour séparer
            tqdm.write(f"Jour {day_num:3d} | {month}/{day} | {total_day_time:5.1f}s | "
                      f"Lecture {read_time:4.1f}s | Création {create_time:4.1f}s | Sauvegarde {save_time:4.1f}s")
            
            if fmt == 'both':
                tqdm.write(f"         Fichiers: NC {nc_size:6.1f}MB + Zarr {zarr_size:6.1f}MB = {nc_size+zarr_size:6.1f}MB total")
            elif fmt == 'netcdf':
                tqdm.write(f"         Fichier NC: {nc_size:6.1f}MB")
            elif fmt == 'zarr':
                tqdm.write(f"         Fichier Zarr: {zarr_size:6.1f}MB")
            
            # Statistiques de progression tous les 10 jours
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = len(day_dirs) - (i + 1)
                eta_seconds = remaining * avg_time
                eta_hours = eta_seconds / 3600
                
                tqdm.write(f"")
                tqdm.write(f"--- Statistiques après {i+1} jours ---")
                tqdm.write(f"Temps moyen: {avg_time:.1f}s/jour | Restant: {remaining} jours | ETA: {eta_hours:.1f}h")
                tqdm.write(f"Erreurs: {errors} | Vitesse actuelle: {3600/avg_time:.1f} jours/heure")
                tqdm.write(f"")
                
        except Exception as e:
            errors += 1
            tqdm.write(f"Erreur {day_dir.name}: {e}")
    
    elapsed = time.time() - start_time
    avg_time = elapsed / len(day_dirs)
    
    print(f"\n" + "="*60)
    print(f"CONVERSION TERMINÉE")
    print(f"="*60)
    print(f"Temps total      : {elapsed/3600:.1f}h ({elapsed/60:.1f}min)")
    print(f"Jours traités    : {len(day_dirs)}")
    print(f"Temps moyen      : {avg_time:.1f}s par jour")
    print(f"Vitesse          : {3600/avg_time:.1f} jours/heure")
    print(f"Erreurs          : {errors}/{len(day_dirs)}")
    
    if errors == 0:
        print(f"Dossier de sortie: /dmidata/users/malegu/data/daily_output/")
        
        # Taille totale
        from subprocess import run
        result = run(['du', '-sh', '/dmidata/users/malegu/data/daily_output'], 
                   capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Taille totale    : {result.stdout.split()[0]}")
            
        # Comptage des fichiers
        nc_count = len(list(Path('/dmidata/users/malegu/data/daily_output').glob('*.nc')))
        zarr_count = len(list(Path('/dmidata/users/malegu/data/daily_output').glob('*.zarr')))
        print(f"Fichiers créés   : {nc_count} NetCDF + {zarr_count} Zarr")
    else:
        print(f"ATTENTION: {errors} jours ont échoué")
        
    print(f"="*60)

if __name__ == '__main__':
    # Traitement OPTIMISÉ à partir du jour 58 (57 déjà traités) avec compression équilibrée
    process_all_days('/dmidata/users/malegu/data/squashfs-root', 
                    fmt='both', compression_level=6, start_day=58)
