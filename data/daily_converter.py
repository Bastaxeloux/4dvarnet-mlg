import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc

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

def process_all_days(root_dir, fmt='both', compression_level=6):
    """
    Itère sur tous les jours d'un dossier racine avec barre de progression
    """
    from tqdm import tqdm
    import time
    
    root = Path(root_dir)
    day_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    
    print(f"Traitement de {len(day_dirs)} jours avec compression niveau {compression_level}")
    print(f"Format(s): {fmt}")
    
    errors = 0
    start_time = time.time()
    
    for day_dir in tqdm(day_dirs, desc="Conversion journalière", unit="jour"):
        try:
            # Pas d'output pour chaque jour, tqdm se charge de l'affichage
            day_name = day_dir.name
            # Trouve le fichier NetCDF de référence dans le dossier du jour
            netcdf_files = list(day_dir.glob('*-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-*.nc'))
            if not netcdf_files:
                raise FileNotFoundError(f"Aucun fichier NetCDF de référence trouvé dans {day_dir}")
            ref_nc = netcdf_files[0]  # Prend le premier (et normalement seul) fichier
            lon, lat, time_val, analysed_st, analysis_error, sea_ice_fraction = read_reference_netcdf(ref_nc)
            sat_data = read_all_satellite_files(day_dir, day_name)
            surfmask, oi = read_extra_ascii_files(day_dir, day_name)
            ds = create_daily_dataset(lon, lat, time_val, sat_data, surfmask, oi, analysed_st, analysis_error, sea_ice_fraction)
            
            output_dir = Path('/dmidata/users/malegu/data/daily_output')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / day_name
            save_daily_dataset(ds, output_path, fmt=fmt, compression_level=compression_level)
            
            # Mise à jour occasionnelle de la description
            if int(day_name[6:8]) % 10 == 1:  # Tous les 10 jours
                tqdm.write(f"Mois {day_name[4:6]}, jour {day_name[6:8]} - {errors} erreurs jusqu'ici")
                
        except Exception as e:
            errors += 1
            tqdm.write(f"Erreur {day_dir.name}: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n=== TERMINÉ ===")
    print(f"Temps total: {elapsed/3600:.1f}h ({elapsed/60:.1f}min)")
    print(f"Erreurs: {errors}/{len(day_dirs)}")
    if errors == 0:
        print(f"Tous les fichiers créés dans: /dmidata/users/malegu/data/daily_output/")
        
        # Estimation de la taille totale
        if len(day_dirs) > 0:
            sample_size = Path('/dmidata/users/malegu/data/daily_output').glob('*')
            sample_files = list(sample_size)
            if sample_files:
                from subprocess import run, PIPE
                result = run(['du', '-sh', '/dmidata/users/malegu/data/daily_output'], 
                           capture_output=True, text=True)
                print(f"Taille totale: {result.stdout.split()[0]}")
    else:
        print(f"Certains jours ont échoué. Vérifiez les erreurs ci-dessus.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        # Mode parallèle: python3 daily_converter.py start_day end_day process_id
        start_day = int(sys.argv[1])
        end_day = int(sys.argv[2])
        process_id = int(sys.argv[3])
        
        root = Path('/dmidata/users/malegu/data/squashfs-root')
        day_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        selected_dirs = day_dirs[start_day-1:end_day]
        
        print(f"Processus {process_id}: jours {start_day}-{end_day} ({len(selected_dirs)} jours)")
        
        errors = 0
        for i, day_dir in enumerate(selected_dirs, 1):
            try:
                process_one_day(day_dir, fmt='both', compression_level=6)
                if i % 5 == 0:
                    print(f"Processus {process_id}: {i}/{len(selected_dirs)} jours traités")
            except Exception as e:
                errors += 1
                print(f"Erreur processus {process_id} - {day_dir.name}: {e}")
        
        print(f"Processus {process_id} terminé: {len(selected_dirs)-errors}/{len(selected_dirs)} succès")
    else:
        # Mode normal: traitement complet
        process_all_days('/dmidata/users/malegu/data/squashfs-root', 
                        fmt='both', compression_level=6)
