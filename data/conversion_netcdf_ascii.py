import numpy as np
import xarray as xr
from pathlib import Path
import netCDF4 as nc

def read_reference_netcdf(netcdf_path):
    print(f"Lecture du NetCDF : {netcdf_path}")
    ds = xr.open_dataset(netcdf_path)
    lon = ds.lon.values
    lat = ds.lat.values
    time = ds.time.values
    print(f"Grille trouvée : {len(lon)} longitudes x {len(lat)} latitudes")
    print(f"Longitude : {lon.min():.3f} à {lon.max():.3f}")
    print(f"Latitude : {lat.min():.3f} à {lat.max():.3f}")
    ds.close()
    return lon, lat, time


def read_ascii_file(asc_path, verbose=True):
    """
    Lit un fichier .asc et retourne les données sous forme d'array numpy
    verbose: si True, affiche les détails de lecture
    """
    if verbose:
        print(f"Lecture du fichier ASCII : {asc_path}")
    with open(asc_path, 'r') as f: lines = f.readlines()
    header1 = lines[0].strip().split()
    header2 = lines[1].strip().split()
    header3 = lines[2].strip().split()
    if verbose:
        print(f"Dimensions ASCII : {header2[0]} x {header2[1]}")
        print(f"Valeur manquante : {header3[1]}")

    data_lines = lines[3:]
    
    data = []
    for line in data_lines:
        row = [float(x) for x in line.strip().split()]
        data.append(row)
    data = np.array(data)
    data[data == 99.0] = np.nan
    
    if verbose:
        print(f"Shape : {data.shape}")
    
    return data

def create_netcdf(data, lon, lat, time, output_path, variable_name, sensor_name):
    """
    Crée un fichier NetCDF avec xarray (version sécurisée)
    """
    print(f"Création du NetCDF : {output_path}")
    
    try:
        # Force la fermeture de tous les fichiers ouverts
        xr.backends.file_manager.FILE_CACHE.clear()
        
        # Crée le dataset xarray avec des types explicites
        ds = xr.Dataset(
            {variable_name: (["time", "lat", "lon"], 
                           data[np.newaxis, :, :].astype(np.float32))},
            coords={
                "lon": (["lon"], lon.astype(np.float32)),
                "lat": (["lat"], lat.astype(np.float32)), 
                "time": (["time"], time)
            }
        )
        
        # Ajoute les attributs
        ds[variable_name].attrs = {
            "long_name": f"{sensor_name} Sea Surface Temperature",
            "standard_name": "sea_surface_temperature", 
            "units": "celsius",
            "_FillValue": np.nan,
            "sensor": sensor_name
        }
        
        ds.lon.attrs = {
            "standard_name": "longitude",
            "units": "degrees_east"
        }
        
        ds.lat.attrs = {
            "standard_name": "latitude",
            "units": "degrees_north"
        }
        
        # Sauvegarde avec compression
        encoding = {variable_name: {'zlib': True, 'complevel': 6, 'dtype': 'float32'}}
        ds.to_netcdf(output_path, encoding=encoding)
        
        # Ferme explicitement
        ds.close()
        del ds
        
        print(f"NetCDF créé : {output_path}")
        
    except Exception as e:
        print(f"Erreur lors de la création du NetCDF : {e}")
        raise

def process_single_day(day_name="2024010112", root_dir="/dmidata/users/malegu/data/squashfs-root"):
    """
    Traite un seul jour de données
    day_name: nom du dossier (ex: "2024010112")
    root_dir: dossier racine contenant les jours
    """
    base_dir = Path(root_dir) / day_name
    reference_nc = base_dir / "20240101120000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc"
    
    if not reference_nc.exists():
        print(f"ERREUR : NetCDF de référence introuvable : {reference_nc}")
        return
    lon, lat, time = read_reference_netcdf(reference_nc)
    
    # Fichiers à traiter avec leurs noms de variables
    asc_files = [
        (f"{day_name}_pmw_cci_l2p_av.asc", "pmw_sst", "PMW"),
        (f"{day_name}_aasti_ist_l2p_av.asc", "aasti_sst", "AASTI"),
        (f"{day_name}_avhrr_c3s_l3u_av.asc", "avhrr_sst", "AVHRR"), 
        (f"{day_name}_slstr_c3s_l3u_av.asc", "slstr_sst", "SLSTR"),
        (f"{day_name}_aasti_ist_l2p_std_av.asc", "aasti_sst_std", "AASTI_STD"),
        (f"{day_name}_avhrr_c3s_l3u_std_av.asc", "avhrr_sst_std", "AVHRR_STD"),
        (f"{day_name}_pmw_cci_l2p_std_av.asc", "pmw_sst_std", "PMW_STD"),
        (f"{day_name}_slstr_c3s_l3u_std_av.asc", "slstr_sst_std", "SLSTR_STD")
    ]
    
    # Crée le dossier netcdf pour organiser les sorties
    netcdf_dir = base_dir.parent / "netcdf" / base_dir.name
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dossier de sortie : {netcdf_dir}")
    
    # Traite un seul fichier à la fois pour éviter les problèmes de mémoire  
    for asc_filename, var_name, sensor in asc_files:
        print(f"\n=== Traitement de {asc_filename} ===")
        asc_path = base_dir / asc_filename
        if not asc_path.exists():
            print(f"ATTENTION : Fichier introuvable : {asc_path}")
            continue
            
        try:
            data = read_ascii_file(asc_path)
            output_path = netcdf_dir / f"{asc_filename.replace('.asc', '.nc')}"
            create_netcdf(data, lon, lat, time, output_path, var_name, sensor)
            print(f"Terminé : {asc_filename}")
        except Exception as e:
            print(f"Erreur avec {asc_filename}: {e}")
        
        print("-" * 50)

def main():
    # Par défaut, traite le premier jour de janvier 2024
    process_single_day()

if __name__ == "__main__":
    main()