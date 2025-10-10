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
    """
    ascii_files = {
        "aasti_av": f"{day}_aasti_ist_l2p_av.asc",
        "aasti_std": f"{day}_aasti_ist_l2p_std_av.asc",
        "avhrr_av": f"{day}_avhrr_c3s_l3u_av.asc",
        "avhrr_std": f"{day}_avhrr_c3s_l3u_std_av.asc",
        "pmw_av": f"{day}_pmw_cci_l2p_av.asc",
        "pmw_std": f"{day}_pmw_cci_l2p_std_av.asc",
        "slstr_av": f"{day}_slstr_c3s_l3u_av.asc",
        "slstr_std": f"{day}_slstr_c3s_l3u_std_av.asc",
    }
    data = {}
    for sat, filename in ascii_files.items():
        path = directory / filename
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


def save_datasets(ds, output_path, save_format="both", compression_level=6, force_overwrite=False):
    """
    Ici on sauvegarde le dataset.
    On peut choisir le format : netcdf, zarr ou both
    Le niveau de compression est choisi par default à 6. Il peut être ajusté entre 1 et 9.
    """
    formats = []
    if save_format in ("netcdf", "both"):
        nc_path = output_path.with_suffix('.nc')
        comp = dict(zlib=True, complevel=compression_level)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(nc_path, format='NETCDF4', encoding=encoding)
        formats.append('NetCDF')
    if save_format in ("zarr", "both"):
        zarr_path = output_path.with_suffix('.zarr')
        ds.to_zarr(zarr_path, mode='w')
        formats.append('Zarr')
    return formats

def process_one_day(directory_path, output_dir, fmt="netcdf", compression_level=6, force_overwrite=False):
    """
    Traite une journée complète de données.
    directory : Path vers le dossier de la journée
    fmt : 'netcdf', 'zarr' ou 'both'
    compression_level : niveau de compression pour netcdf (1-9)
    force_overwrite : si True, écrase les fichiers existants
    """
    # ici je veux en premier lieu verifier si les fichiers existent déjà (le netcdf et le zarr)
    day_name = directory_path.name  # ex: '2024010112'
    if output_dir is None:
        year = day_name[:4]
        output_dir = Path('/dmidata/users/malegu/data/daily_output') / year
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{day_name}_13vars"
    
    need_nc = fmt in ('netcdf', 'both') and (force_overwrite or not output_path.with_suffix('.nc').exists())
    need_zarr = fmt in ('zarr', 'both') and (force_overwrite or not output_path.with_suffix('.zarr').exists())
    if not (need_nc or need_zarr):
        print (f"Files already exist for {day_name}, skipping.")
        return []
    
    # Sinon c'est qu'on doit créer au moins un des deux formats, donc on se met a lire les données
    try:
        lat, lon, time, analysed_st, analysis_error, sea_ice_fraction = read_netcdf(directory_path / f"{day_name}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc")
        sat_data = read_sat_ascii(directory_path, day_name)
        surfmask, oi_data = read_oi_surfmask_ascii(directory_path, day_name)
        ds = create_full_dataset(lon, lat, time, sat_data, surfmask, oi_data, analysed_st, analysis_error, sea_ice_fraction)
        saved_formats = save_datasets(ds, output_path, save_format=fmt, compression_level=compression_level, force_overwrite=force_overwrite)
        return saved_formats
    except Exception as e:
        raise RuntimeError(f"Error processing {day_name}: {e}")
    return []

def process_year(year, output_dir):
    """
    Traite toutes les journées d'une année.
    year : int, année à traiter
    output_dir : Path vers le dossier de sortie
    """
    source_dir = Path(f'/dmidata/users/malegu/data/squash_{year}_extract')
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist or is not a directory.")
    day_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not day_dirs:
        print(f"No day directories found in {source_dir}.")
        return
    for day_dir in tqdm(sorted(day_dirs), desc=f"Processing year {year}", unit="day"):
        try:
            saved_formats = process_one_day(day_dir, fmt='netcdf', output_dir=output_dir, compression_level=6, force_overwrite=False)
            if saved_formats:
                tqdm.write(f"{day_dir.name}: {', '.join(saved_formats)} created.")
        except Exception as e:
            tqdm.write(f"Error processing {day_dir.name}: {e}")
    return

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 converter.py YEAR")
        sys.exit(1)
    year = int(sys.argv[1])
    output_base_dir = Path(f'/dmidata/users/malegu/data/netcdf_{year}')
    process_year(year, output_base_dir)
    
    print(20*"=")
    print(f"Processing for year {year} completed.")
    print(20*"=")