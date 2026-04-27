import numpy as np
import xarray as xr
from pathlib import Path
from datetime import date
from tqdm import tqdm

# Importe les fonctions du script de base
from conversion_netcdf_ascii import read_reference_netcdf, read_ascii_file

def create_annual_dataset(data_dict, lon, lat, time_coords, output_path, format_type="both"):
    """
    Crée un dataset annuel avec toutes les variables et la dimension temporelle
    Sauvegarde en NetCDF, Zarr, ou les deux selon format_type
    
    data_dict: dictionnaire {variable_name: array 3D (time, lat, lon)}
    format_type: "netcdf", "zarr", ou "both"
    """
    import gc
    import time
    from pathlib import Path
    
    print(f"Création du dataset annuel : {output_path}")
    print(f"Format(s) à sauvegarder : {format_type}")
    
    try:
        # Force la fermeture de tous les fichiers ouverts
        xr.backends.file_manager.FILE_CACHE.clear()
        
        # Nettoyage mémoire
        gc.collect()
        
        print("Préparation des variables...")
        
        # Crée le dataset xarray avec toutes les variables
        data_vars = {}
        for var_name, data_array in data_dict.items():
            print(f"  Ajout de {var_name}: {data_array.shape}")
            # Force float32 et s'assure que les données sont contiguës
            data_vars[var_name] = (["time", "lat", "lon"], 
                                   np.ascontiguousarray(data_array, dtype=np.float32))
        
        print("Création du dataset xarray...")
        ds = xr.Dataset(
            data_vars,
            coords={
                "lon": (["lon"], np.ascontiguousarray(lon, dtype=np.float32)),
                "lat": (["lat"], np.ascontiguousarray(lat, dtype=np.float32)), 
                "time": (["time"], np.ascontiguousarray(time_coords, dtype=np.int32))
            }
        )
        
        print("Ajout des attributs...")
        # Ajoute les attributs pour chaque variable
        sensor_info = {
            "aasti_av": ("AASTI Sea Surface Temperature", "AASTI"),
            "aasti_std": ("AASTI Sea Surface Temperature Standard Deviation", "AASTI"),
            "avhrr_av": ("AVHRR Sea Surface Temperature", "AVHRR"),
            "avhrr_std": ("AVHRR Sea Surface Temperature Standard Deviation", "AVHRR"),
            "pmw_av": ("PMW Sea Surface Temperature", "PMW"),
            "pmw_std": ("PMW Sea Surface Temperature Standard Deviation", "PMW"),
            "slstr_av": ("SLSTR Sea Surface Temperature", "SLSTR"),
            "slstr_std": ("SLSTR Sea Surface Temperature Standard Deviation", "SLSTR")
        }
        
        for var_name in data_vars.keys():
            if var_name in sensor_info:
                long_name, sensor = sensor_info[var_name]
                ds[var_name].attrs = {
                    "long_name": long_name,
                    "standard_name": "sea_surface_temperature", 
                    "units": "celsius",
                    "_FillValue": np.nan,
                    "sensor": sensor
                }
        
        ds.lon.attrs = {
            "standard_name": "longitude",
            "units": "degrees_east"
        }
        
        ds.lat.attrs = {
            "standard_name": "latitude",
            "units": "degrees_north"
        }
        
        ds.time.attrs = {
            "standard_name": "time",
            "units": "days since 2024-01-01"
        }
        
        # Sauvegarde selon le format choisi
        results = {}
        
        if format_type in ["netcdf", "both"]:
            print("Sauvegarde NetCDF...")
            start_time = time.time()
            
            # Configuration NetCDF
            netcdf_path = str(output_path).replace('.zarr', '.nc')
            encoding_nc = {}
            for var_name in data_vars.keys():
                encoding_nc[var_name] = {
                    'zlib': True, 
                    'complevel': 6,
                    'dtype': 'float32',
                    'chunksizes': None
                }
            
            ds.to_netcdf(netcdf_path, encoding=encoding_nc, engine='netcdf4', format='NETCDF4_CLASSIC')
            
            netcdf_time = time.time() - start_time
            netcdf_size = Path(netcdf_path).stat().st_size / (1024**2)  # MB
            results['netcdf'] = {'path': netcdf_path, 'time': netcdf_time, 'size_mb': netcdf_size}
            print(f"  NetCDF créé : {netcdf_path} ({netcdf_size:.1f} MB en {netcdf_time:.1f}s)")
        
        if format_type in ["zarr", "both"]:
            print("Sauvegarde Zarr...")
            start_time = time.time()
            
            # Configuration Zarr avec compression optimisée
            zarr_path = str(output_path).replace('.nc', '.zarr')
            encoding_zarr = {}
            for var_name in data_vars.keys():
                encoding_zarr[var_name] = {
                    'chunks': (10, 360, 720),  # Format tuple pour Zarr
                    'dtype': 'float32'
                }
            
            # Supprime le dossier zarr s'il existe
            import shutil
            if Path(zarr_path).exists():
                shutil.rmtree(zarr_path)
                
            ds.to_zarr(zarr_path, encoding=encoding_zarr, consolidated=True)
            
            zarr_time = time.time() - start_time
            # Taille du dossier zarr
            zarr_size = sum(f.stat().st_size for f in Path(zarr_path).rglob('*') if f.is_file()) / (1024**2)
            results['zarr'] = {'path': zarr_path, 'time': zarr_time, 'size_mb': zarr_size}
            print(f"  Zarr créé : {zarr_path} ({zarr_size:.1f} MB en {zarr_time:.1f}s)")
        
        # Ferme explicitement et nettoie
        ds.close()
        del ds, data_vars
        gc.collect()
        
        # Affiche le résumé de comparaison
        if len(results) > 1:
            print(f"\n=== COMPARAISON DES FORMATS ===")
            for fmt, info in results.items():
                print(f"{fmt.upper()}: {info['size_mb']:.1f} MB, {info['time']:.1f}s")
            if 'netcdf' in results and 'zarr' in results:
                ratio = results['netcdf']['size_mb'] / results['zarr']['size_mb']
                print(f"Gain Zarr: {ratio:.1f}x plus compact que NetCDF")
        
        return results
        
    except Exception as e:
        print(f"Erreur lors de la création du dataset : {e}")
        # Force le nettoyage même en cas d'erreur
        try:
            if 'ds' in locals():
                ds.close()
                del ds
        except:
            pass
        gc.collect()
        raise

def main():
    # Dossier racine contenant tous les jours
    root_dir = Path("/dmidata/users/malegu/data/squashfs-root")
    
    # Récupère tous les dossiers de jours (triés)
    day_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    print(f"Trouvé {len(day_dirs)} jours de données")
    
    # NetCDF de 30 jours pour le maître de stage
    day_dirs = day_dirs[:30]  # 30 jours pour démonstration
    # day_dirs = day_dirs[:100]  # TEST: 100 jours (problème mémoire)
    # day_dirs = day_dirs  # Année complète
    print(f"Traitement de {len(day_dirs)} jours (DEMO maître de stage)")
    
    # Définit les fichiers à traiter (sans le préfixe date)
    asc_patterns = {
        "aasti_av": "_aasti_ist_l2p_av.asc",
        "aasti_std": "_aasti_ist_l2p_std_av.asc", 
        "avhrr_av": "_avhrr_c3s_l3u_av.asc",
        "avhrr_std": "_avhrr_c3s_l3u_std_av.asc",
        "pmw_av": "_pmw_cci_l2p_av.asc",
        "pmw_std": "_pmw_cci_l2p_std_av.asc",
        "slstr_av": "_slstr_c3s_l3u_av.asc",
        "slstr_std": "_slstr_c3s_l3u_std_av.asc"
    }
    
    # Lit la grille de référence depuis le premier jour
    first_day = day_dirs[0]
    reference_nc = first_day / "20240101120000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc"
    
    if not reference_nc.exists():
        print(f"ERREUR : NetCDF de référence introuvable : {reference_nc}")
        return
        
    lon, lat, _ = read_reference_netcdf(reference_nc)
    
    # Initialise les tableaux 3D pour stocker toutes les données
    data_dict = {}
    for var_name in asc_patterns.keys():
        data_dict[var_name] = np.full((len(day_dirs), len(lat), len(lon)), np.nan, dtype=np.float32)
    
    # Crée les coordonnées temporelles (jours depuis 2024-01-01)
    time_coords = []
    
    # Parcourt tous les jours avec une progress bar
    missing_files = 0
    errors = 0
    
    for day_idx, day_dir in enumerate(tqdm(day_dirs, desc="Traitement des jours", unit="jour")):
        day_name = day_dir.name
        
        # Extrait la date pour les coordonnées temporelles
        # Format: 2024010112 -> 2024-01-01
        year = int(day_name[:4])
        month = int(day_name[4:6])
        day = int(day_name[6:8])
        
        # Calcule les jours depuis 2024-01-01
        date_obj = date(year, month, day)
        ref_date = date(2024, 1, 1)
        days_since_ref = (date_obj - ref_date).days
        time_coords.append(days_since_ref)
        
        # Traite chaque type de fichier pour ce jour
        day_missing = 0
        day_errors = 0
        
        for var_name, pattern in asc_patterns.items():
            asc_filename = day_name + pattern
            asc_path = day_dir / asc_filename
            
            if not asc_path.exists():
                day_missing += 1
                missing_files += 1
                continue
                
            try:
                # Mode silencieux pour éviter trop d'output
                data = read_ascii_file(asc_path, verbose=False)
                data_dict[var_name][day_idx, :, :] = data
            except Exception as e:
                day_errors += 1
                errors += 1
        
        # Met à jour la description de la progress bar avec les statistiques
        tqdm.write(f"Jour {day_name}: {8-day_missing-day_errors}/8 fichiers OK, {day_missing} manquants, {day_errors} erreurs")
    
    # Crée le dossier de sortie dans le répertoire de travail
    output_dir = Path("/home/malegu/data/netcdf_annual")
    output_dir.mkdir(exist_ok=True)
    
    # Crée le dataset annuel (NetCDF et/ou Zarr)
    output_path = output_dir / f"sst_annual_2024_{len(day_dirs)}days.nc"
    print(f"\nCréation du dataset annuel...")
    
    # CHOIX DU FORMAT : "netcdf", "zarr", ou "both"
    format_choice = "both"
    
    results = create_annual_dataset(data_dict, lon, lat, np.array(time_coords), output_path, format_choice)
    
    print(f"\n=== TERMINÉ ===")
    print(f"Fichiers créés :")
    for fmt, info in results.items():
        print(f"  {fmt.upper()}: {info['path']} ({info['size_mb']:.1f} MB)")
    print(f"Variables : {list(data_dict.keys())}")
    print(f"Dimensions : {len(day_dirs)} jours x {len(lat)} lat x {len(lon)} lon")
    print(f"Statistiques : {missing_files} fichiers manquants, {errors} erreurs")

if __name__ == "__main__":
    main()
