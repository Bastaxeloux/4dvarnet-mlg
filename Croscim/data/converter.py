import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc
from tqdm import tqdm

def read_netcdf(path):
    ds = xr.open_dataset(path)
    required_vars = ["lat", "lon", "time", "analysed_st", "analysis_error", "sea_ice_fraction"]
    missing = [var for var in required_vars if var not in ds]
    if missing:
        ds.close() # on ferme avant l'erreur sinon fuite de mémoire
        raise ValueError(f"Missing variables: {missing}")
    data = {var: ds[var].values for var in required_vars}
    ds.close()
    return data['lat'], data['lon'], data['time'], data['analysed_st'], data['analysis_error'], data['sea_ice_fraction']

def read_ascii(path):
    """
    Ici on va lire les fichiers ASCII nécessaires pour constituer notre netcdf final
    """
    with open(path,"r") as f:
        lines = f.readlines()
    infos = lines[:3]
    data = [list(map(float, line.strip().split())) for line in lines[3:]]
    # quand on lit un fichier texte, tout est lu comme des STRINGS, il est donc nécessaire de convertir en FLOAT avec map et de repasser a une liste
    data = np.array(data)
    data[data == 999.0] = np.nan
    data[data == 99.0] = np.nan
    return infos, data

def read_sat_ascii(directory, day):
    """
    A partir du dossier d'une journée, on va lire les fichiers ASCII des 4 satellites nécessaires
    Utilise des patterns pour supporter les changements de nomenclature (c3s/cci)
    """
    # Patterns avec wildcards pour supporter c3s et cci
    ascii_patterns = {
        "aasti_av": f"{day}_aasti_*av.asc",
        "aasti_std": f"{day}_aasti_*std_av.asc",
        "avhrr_av": f"{day}_avhrr_*av.asc",
        "avhrr_std": f"{day}_avhrr_*std_av.asc",
        "pmw_av": f"{day}_pmw_*av.asc",
        "pmw_std": f"{day}_pmw_*std_av.asc",
        "slstr_av": f"{day}_slstr_*av.asc",
        "slstr_std": f"{day}_slstr_*std_av.asc",
    }
    data = {}
    for sat, pattern in ascii_patterns.items():
        # Si le pattern contient un wildcard, utiliser glob
        if '*' in pattern:
            matches = list(directory.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No file matching pattern: {directory / pattern}")
            path = matches[0]  # Prendre le premier match
        else:
            path = directory / pattern
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
        _ , datas = read_ascii(path)
        data[sat] = datas
    return data


def read_oi_surfmask_ascii(directory, day):
    """
    Lit le fichier ASCII du masque de surface OI pour une journée donnée.
    """
    surfmask_name = directory / f"surfmask_{day}.asc"
    oi_path = directory / f"oi_{day}.asc"
    if not surfmask_name.exists() or not oi_path.exists():
        raise FileNotFoundError(f"Missing OI or surfmask file for {day}")
    _, surfmask = read_ascii(surfmask_name)
    _, oi_data = read_ascii(oi_path)
    return surfmask, oi_data

def squeeze_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] >= 1:
        return arr[0]
    return arr

def create_full_dataset(lon,lat,time,sat_data,surfmask,oi_data,analysed_st,analysis_error,sea_ice_fraction, verbose=False):
    """
    On va ici combiner toutes les données dans un seul xarray Dataset
    On retourne un objet de type xarray.Dataset
    """
    if verbose:
        print("Debug - Dimensions des données d'entrée:")
        print(f"lon shape: {lon.shape}")
        print(f"lat shape: {lat.shape}")
        print(f"time shape: {time.shape}")
        print(f"analysed_st shape: {analysed_st.shape}")
        print(f"surfmask shape: {surfmask.shape}")
        print(f"oi_data shape: {oi_data.shape}")
    
    data = {}
    missing = []
    if surfmask is None: missing.append("surfmask")
    if oi_data is None: missing.append("oi_data")
    if analysed_st is None: missing.append("analysed_st")
    if analysis_error is None: missing.append("analysis_error")
    if sea_ice_fraction is None: missing.append("sea_ice_fraction")
    if missing:
        raise ValueError(f"Missing data for: {missing}") # peut etre inutile mais je préfère verifier a chaque etape que toutes les data sont présentes
    for var, data_array in sat_data.items():
        if data_array is not None:
            if verbose:
                print(f"Debug - {var} avant squeeze_2d: {data_array.shape}")
            arr = squeeze_2d(data_array).astype(np.float32)
            if verbose:
                print(f"Debug - {var} après squeeze_2d: {arr.shape}")
            data[var] = (['lat', 'lon'], arr)
            if verbose:
                print(f"Debug - {var} après création du tuple")
        else:
            raise ValueError(f"Data for {var} is None")
    data['surfmask'] = (['lat', 'lon'], squeeze_2d(surfmask))
    data['oi_data'] = (['lat', 'lon'], squeeze_2d(oi_data))
    data['analysed_st'] = (['lat', 'lon'], squeeze_2d(analysed_st).astype(np.float32))
    data['analysis_error'] = (['lat', 'lon'], squeeze_2d(analysis_error).astype(np.float32))
    data['sea_ice_fraction'] = (['lat', 'lon'], squeeze_2d(sea_ice_fraction).astype(np.float32))
    ds = xr.Dataset(data, coords={'lon': (['lon'], lon.astype(np.float32)),
                                  'lat': (['lat'], lat.astype(np.float32)),
                                  'time': (['time'], time.astype(np.float64))})
    return ds


def save_datasets(ds, nc_output_path=None, zarr_output_path=None, save_format="both", compression_level=6, force_overwrite=False):
    """
    Ici on sauvegarde le dataset.
    On peut choisir le format : netcdf, zarr ou both
    Le niveau de compression peut être ajusté entre 1 et 9.
    nc_output_path: chemin pour NetCDF (sans extension)
    zarr_output_path: chemin pour Zarr (sans extension)
    """
    formats = []
    if save_format in ("netcdf", "both") and nc_output_path is not None:
        nc_path = Path(nc_output_path).with_suffix('.nc')
        comp = dict(zlib=True, complevel=compression_level)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(nc_path, format='NETCDF4', encoding=encoding)
        formats.append('NetCDF')
    if save_format in ("zarr", "both") and zarr_output_path is not None:
        zarr_path = Path(zarr_output_path).with_suffix('.zarr')
        encoding = {var: {'chunks': (768, 768)} for var in ds.data_vars}
        ds.to_zarr(zarr_path, mode='w', encoding=encoding)
        formats.append('Zarr')
    return formats

def process_one_day(directory_path, nc_output_dir=None, zarr_output_dir=None, fmt="both", compression_level=6, force_overwrite=False):
    """
    Traite une journée complète de données.
    directory_path : Path vers le dossier de la journée
    nc_output_dir : dossier de sortie pour NetCDF
    zarr_output_dir : dossier de sortie pour Zarr
    fmt : 'netcdf', 'zarr' ou 'both'
    compression_level : niveau de compression pour netcdf (1-9)
    force_overwrite : si True, écrase les fichiers existants
    """
    day_name = directory_path.name  # ex: '2024010112'
    year = day_name[:4]

    # Définir les dossiers de sortie par défaut
    if nc_output_dir is None:
        nc_output_dir = Path(f'/dmidata/users/malegu/netcdf_{year}')
    if zarr_output_dir is None:
        zarr_output_dir = Path(f'/dmidata/projects/4dvarnet/data_{year}')

    nc_output_dir = Path(nc_output_dir)
    zarr_output_dir = Path(zarr_output_dir)
    nc_output_dir.mkdir(parents=True, exist_ok=True)
    zarr_output_dir.mkdir(parents=True, exist_ok=True)

    nc_output_path = nc_output_dir / f"{day_name}_x1"
    zarr_output_path = zarr_output_dir / f"{day_name}_x1"

    need_nc = fmt in ('netcdf', 'both') and (force_overwrite or not nc_output_path.with_suffix('.nc').exists())
    need_zarr = fmt in ('zarr', 'both') and (force_overwrite or not zarr_output_path.with_suffix('.zarr').exists())
    if not (need_nc or need_zarr):
        print (f"Files already exist for {day_name}, skipping.")
        return []

    # Sinon c'est qu'on doit créer au moins un des deux formats, donc on se met a lire les données
    try:
        lat, lon, time, analysed_st, analysis_error, sea_ice_fraction = read_netcdf(directory_path / f"{day_name}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc")
        sat_data = read_sat_ascii(directory_path, day_name)
        surfmask, oi_data = read_oi_surfmask_ascii(directory_path, day_name)
        ds = create_full_dataset(lon, lat, time, sat_data, surfmask, oi_data, analysed_st, analysis_error, sea_ice_fraction)
        saved_formats = save_datasets(ds, nc_output_path=nc_output_path, zarr_output_path=zarr_output_path, save_format=fmt, compression_level=compression_level, force_overwrite=force_overwrite)
        return saved_formats
    except Exception as e:
        raise RuntimeError(f"Error processing {day_name}: {e}")
    return []

def process_year(year, nc_output_dir=None, zarr_output_dir=None):
    """
    Traite toutes les journées d'une année.
    year : int, année à traiter
    nc_output_dir : dossier de sortie pour NetCDF
    zarr_output_dir : dossier de sortie pour Zarr
    """
    source_dir = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist or is not a directory.")
    day_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not day_dirs:
        print(f"No day directories found in {source_dir}.")
        return
    for day_dir in tqdm(sorted(day_dirs), desc=f"Processing year {year}", unit="day"):
        try:
            saved_formats = process_one_day(day_dir, nc_output_dir=nc_output_dir, zarr_output_dir=zarr_output_dir, fmt='both', compression_level=6, force_overwrite=False)
            if saved_formats:
                tqdm.write(f"{day_dir.name}: {', '.join(saved_formats)} created.")
        except Exception as e:
            tqdm.write(f"Error processing {day_dir.name}: {e}")
    return

def _process_day_wrapper(args):
    """Wrapper function for multiprocessing - must be at module level to be pickleable"""
    day_dir, nc_output_dir, zarr_output_dir = args
    try:
        saved_formats = process_one_day(day_dir, nc_output_dir=nc_output_dir, zarr_output_dir=zarr_output_dir, fmt='both', compression_level=6, force_overwrite=False)
        return (day_dir.name, saved_formats, None)
    except Exception as e:
        return (day_dir.name, [], str(e))

def process_year_parallel(year, nc_output_dir=None, zarr_output_dir=None, nb_workers=4):
    """Version parallélisée du traitement"""
    from multiprocessing import Pool
    source_dir = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    day_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not day_dirs:
        print(f"No day directories found in {source_dir}.")
        return
    print(f"Traitement parallèle: {len(day_dirs)} jours avec {nb_workers} workers")

    args_list = [(day_dir, nc_output_dir, zarr_output_dir) for day_dir in day_dirs]

    with Pool(processes=nb_workers) as pool:
        results = list(tqdm(pool.imap(_process_day_wrapper, args_list), total=len(day_dirs), desc=f"Year {year}", unit="jour"))

    success = sum(1 for _, formats, err in results if formats and not err)
    errors = sum(1 for _, _, err in results if err)
    print(f"\nRésumé: {success} jours OK, {errors} erreurs")

    if errors > 0:
        print("\nErreurs détaillées:")
        for day_name, formats, err in results:
            if err:
                print(f"  {day_name}: {err}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python converter.py YEAR [--parallel NB_WORKERS]")
        sys.exit(1)

    year = int(sys.argv[1])
    nc_output_dir = Path(f'/dmidata/users/malegu/netcdf_{year}')
    zarr_output_dir = Path(f'/dmidata/projects/4dvarnet/data_{year}')
    nc_output_dir.mkdir(parents=True, exist_ok=True)
    zarr_output_dir.mkdir(parents=True, exist_ok=True)

    # Check parallèle
    if '--parallel' in sys.argv:
        idx = sys.argv.index('--parallel')
        nb_workers = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 4
        process_year_parallel(year, nc_output_dir, zarr_output_dir, nb_workers)
    else:
        process_year(year, nc_output_dir, zarr_output_dir)

    print(20*"=")
    print(f"Processing for year {year} completed.")
    print(20*"=")