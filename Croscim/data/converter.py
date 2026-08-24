import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc
from tqdm import tqdm

try:
    from .ascii_files import SATELLITES, resolve_satellite_ascii
except ImportError:
    from ascii_files import SATELLITES, resolve_satellite_ascii

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
    data = {}
    for satellite in SATELLITES:
        for statistic in ("av", "std"):
            path = resolve_satellite_ascii(
                directory, day, satellite, statistic
            )
            _, values = read_ascii(path)
            data[f"{satellite}_{statistic}"] = values

        mean = data[f"{satellite}_av"]
        uncertainty = data[f"{satellite}_std"]
        if (
            np.isfinite(mean).any()
            and np.isfinite(uncertainty).any()
            and np.array_equal(mean, uncertainty, equal_nan=True)
        ):
            raise ValueError(
                f"{satellite.upper()} mean and uncertainty are identical for "
                f"{day}"
            )
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


def is_complete_zarr(path):
    """A consolidated metadata file is written only after a successful save."""
    path = Path(path)
    return path.is_dir() and (path / ".zmetadata").is_file()


def remove_path(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

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


def save_datasets(ds, nc_output_path=None, zarr_output_path=None, save_format="both", compression_level=4, force_overwrite=False, chunk_size=1536):
    """
    Ici on sauvegarde le dataset.
    On peut choisir le format : netcdf, zarr ou both
    Le niveau de compression peut être ajusté entre 1 et 9.
    nc_output_path: chemin pour NetCDF (sans extension)
    zarr_output_path: chemin pour Zarr (sans extension)
    chunk_size: taille des chunks spatiaux (défaut 1536×1536 pour optimiser nombre de fichiers)
    """
    formats = []
    if save_format in ("netcdf", "both") and nc_output_path is not None:
        nc_path = Path(nc_output_path).with_suffix('.nc')

        # Optimisation: chunks 1536×1536 pour réduire nombre de fichiers
        # Avec shuffling pour meilleure compression
        encoding = {}
        for var in ds.data_vars:
            var_dims = ds[var].dims
            if 'lat' in var_dims and 'lon' in var_dims:
                # Variables 2D spatiales: chunks 512×512
                encoding[var] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': (chunk_size, chunk_size)
                }
            else:
                # Variables 1D (lat, lon, time): pas de chunking spatial
                encoding[var] = {
                    'zlib': True,
                    'complevel': compression_level
                }

        ds.to_netcdf(nc_path, format='NETCDF4', encoding=encoding)
        formats.append('NetCDF')
    if save_format in ("zarr", "both") and zarr_output_path is not None:
        zarr_path = Path(zarr_output_path).with_suffix('.zarr')
        temp_path = zarr_path.with_name(f".{zarr_path.name}.tmp-{os.getpid()}")
        remove_path(temp_path)
        encoding = {var: {'chunks': (chunk_size, chunk_size)} for var in ds.data_vars}
        try:
            ds.to_zarr(temp_path, mode='w', encoding=encoding, consolidated=True)
            remove_path(zarr_path)
            temp_path.rename(zarr_path)
        except BaseException:
            remove_path(temp_path)
            raise
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

    # Définir les dossiers de sortie par défaut uniquement pour les formats demandés.
    if fmt in ('netcdf', 'both') and nc_output_dir is None:
        nc_output_dir = Path(f'/nwp/sst_malegu/data_{year}')
    if fmt in ('zarr', 'both') and zarr_output_dir is None:
        zarr_output_dir = Path(f'/nwp/sst_malegu/data_{year}')

    if nc_output_dir is not None:
        nc_output_dir = Path(nc_output_dir)
        nc_output_dir.mkdir(parents=True, exist_ok=True)
    if zarr_output_dir is not None:
        zarr_output_dir = Path(zarr_output_dir)
        zarr_output_dir.mkdir(parents=True, exist_ok=True)

    nc_output_path = nc_output_dir / f"{day_name}_x1" if nc_output_dir is not None else None
    zarr_output_path = zarr_output_dir / f"{day_name}_x1" if zarr_output_dir is not None else None

    need_nc = fmt in ('netcdf', 'both') and (force_overwrite or not nc_output_path.with_suffix('.nc').exists())
    zarr_path = zarr_output_path.with_suffix('.zarr') if zarr_output_path is not None else None
    if zarr_path is not None and zarr_path.exists() and not is_complete_zarr(zarr_path):
        remove_path(zarr_path)
    need_zarr = fmt in ('zarr', 'both') and (force_overwrite or not is_complete_zarr(zarr_path))
    if not (need_nc or need_zarr):
        print (f"Files already exist for {day_name}, skipping.")
        return []

    # Sinon c'est qu'on doit créer au moins un des deux formats, donc on se met a lire les données
    ds = None
    try:
        lat, lon, time, analysed_st, analysis_error, sea_ice_fraction = read_netcdf(directory_path / f"{day_name}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc")
        sat_data = read_sat_ascii(directory_path, day_name)
        surfmask, oi_data = read_oi_surfmask_ascii(directory_path, day_name)
        ds = create_full_dataset(lon, lat, time, sat_data, surfmask, oi_data, analysed_st, analysis_error, sea_ice_fraction)
        saved_formats = save_datasets(ds, nc_output_path=nc_output_path, zarr_output_path=zarr_output_path, save_format=fmt, compression_level=compression_level, force_overwrite=force_overwrite)
        return saved_formats
    except Exception as e:
        raise RuntimeError(f"Error processing {day_name}: {e}")
    finally:
        if ds is not None:
            ds.close()

def process_year(year, nc_output_dir=None, zarr_output_dir=None, source_dir=None):
    """
    Traite toutes les journées d'une année.
    year : int, année à traiter
    nc_output_dir : dossier de sortie pour NetCDF
    zarr_output_dir : dossier de sortie pour Zarr
    """
    # Déterminer le format de sortie
    if nc_output_dir and zarr_output_dir:
        fmt = 'both'
    elif zarr_output_dir:
        fmt = 'zarr'
    elif nc_output_dir:
        fmt = 'netcdf'
    else:
        raise ValueError("Au moins un chemin de sortie (nc_output_dir ou zarr_output_dir) doit être fourni")
    compression_level = 4

    if source_dir is None:
        source_dir = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    else:
        source_dir = Path(source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist or is not a directory.")
    day_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not day_dirs:
        print(f"No day directories found in {source_dir}.")
        return

    print(f"Format de sortie: {fmt}")

    for day_dir in tqdm(sorted(day_dirs), desc=f"Processing year {year}", unit="day"):
        try:
            saved_formats = process_one_day(day_dir, nc_output_dir=nc_output_dir, zarr_output_dir=zarr_output_dir, fmt=fmt, compression_level=compression_level, force_overwrite=False)
            if saved_formats:
                tqdm.write(f"{day_dir.name}: {', '.join(saved_formats)} created.")
        except Exception as e:
            tqdm.write(f"Error processing {day_dir.name}: {e}")
    return

def _process_day_wrapper(args):
    """Wrapper function for multiprocessing - must be at module level to be pickleable"""
    day_dir, nc_output_dir, zarr_output_dir, fmt, compression_level = args
    try:
        saved_formats = process_one_day(day_dir, nc_output_dir=nc_output_dir, zarr_output_dir=zarr_output_dir, fmt=fmt, compression_level=compression_level, force_overwrite=False)
        return (day_dir.name, saved_formats, None)
    except Exception as e:
        return (day_dir.name, [], str(e))

def process_year_parallel(year, nc_output_dir=None, zarr_output_dir=None, nb_workers=4, source_dir=None):
    """Version parallélisée du traitement"""
    from multiprocessing import Pool

    # Déterminer le format de sortie
    if nc_output_dir and zarr_output_dir:
        fmt = 'both'
    elif zarr_output_dir:
        fmt = 'zarr'
    elif nc_output_dir:
        fmt = 'netcdf'
    else:
        raise ValueError("Au moins un chemin de sortie (nc_output_dir ou zarr_output_dir) doit être fourni")
    compression_level = 4

    if source_dir is None:
        source_dir = Path(f'/dmidata/projects/4dvarnet/squash_{year}_extract')
    else:
        source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
    day_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not day_dirs:
        print(f"No day directories found in {source_dir}.")
        return
    print(f"Traitement parallèle: {len(day_dirs)} jours avec {nb_workers} workers")
    print(f"Format de sortie: {fmt}")

    args_list = [(day_dir, nc_output_dir, zarr_output_dir, fmt, compression_level) for day_dir in day_dirs]

    # Each day temporarily holds several global grids. Recycling workers avoids
    # allocator and library state accumulating over an entire year.
    with Pool(processes=nb_workers, maxtasksperchild=1) as pool:
        results = list(tqdm(pool.imap_unordered(_process_day_wrapper, args_list), total=len(day_dirs), desc=f"Year {year}", unit="jour"))

    success = sum(1 for _, formats, err in results if formats and not err)
    errors = sum(1 for _, _, err in results if err)
    print(f"\nRésumé: {success} jours OK, {errors} erreurs")

    if errors > 0:
        print("\nErreurs détaillées:")
        for day_name, formats, err in results:
            if err:
                print(f"  {day_name}: {err}")
        raise RuntimeError(f"{errors} day(s) failed during x1 conversion")

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Convert SST data to NetCDF/Zarr')
    parser.add_argument('year', type=int, help='Year to process')
    parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel workers')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: /nwp/sst_malegu/data_{YEAR})')
    parser.add_argument('--zarr-output-dir', type=str, help='Zarr output directory (optional, for backward compatibility)')
    parser.add_argument('--source-dir', type=str, help='Extracted source directory (default: /dmidata/projects/4dvarnet/squash_{YEAR}_extract)')

    args = parser.parse_args()
    year = args.year

    # Déterminer les chemins de sortie
    if args.zarr_output_dir and not args.output_dir:
        # Zarr uniquement
        nc_output_dir = None
        zarr_output_dir = Path(args.zarr_output_dir)
    elif args.output_dir and not args.zarr_output_dir:
        # NetCDF uniquement (ou défaut)
        nc_output_dir = Path(args.output_dir)
        zarr_output_dir = None
    elif args.zarr_output_dir and args.output_dir:
        # Les deux
        nc_output_dir = Path(args.output_dir)
        zarr_output_dir = Path(args.zarr_output_dir)
    else:
        # Défaut : NetCDF
        nc_output_dir = Path(f'/nwp/sst_malegu/data_{year}')
        zarr_output_dir = None

    if nc_output_dir:
        nc_output_dir.mkdir(parents=True, exist_ok=True)
    if zarr_output_dir:
        zarr_output_dir.mkdir(parents=True, exist_ok=True)

    # Traitement parallèle ou séquentiel
    if args.parallel:
        process_year_parallel(year, nc_output_dir, zarr_output_dir, args.parallel, source_dir=args.source_dir)
    else:
        process_year(year, nc_output_dir, zarr_output_dir, source_dir=args.source_dir)

    print(20*"=")
    print(f"Processing for year {year} completed.")
    print(20*"=")
