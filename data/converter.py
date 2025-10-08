import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc

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

