import numpy as np
import xarray as xr
from pathlib import Path
from datetime import date
from tqdm import tqdm
import zarr

# Importe les fonctions du script de base
from conversion_netcdf_ascii import read_reference_netcdf, read_ascii_file

def create_annual_zarr_streaming(day_dirs, asc_patterns, lon, lat, output_path):
    """
    Crée un fichier Zarr annuel en streaming (jour par jour)
    Évite les problèmes de mémoire en n'ayant qu'un jour en RAM à la fois
    """
    print(f"Création du Zarr annuel en streaming : {output_path}")
    
    # Supprime le fichier existant s'il y en a un
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Dimensions
    n_days = len(day_dirs)
    n_lat, n_lon = len(lat), len(lon)
    n_vars = len(asc_patterns)
    
    print(f"Dimensions finales : {n_days} jours × {n_lat} lat × {n_lon} lon × {n_vars} variables")
    
    # Crée les coordonnées temporelles
    time_coords = []
    for day_dir in day_dirs:
        day_name = day_dir.name
        year = int(day_name[:4])
        month = int(day_name[4:6])
        day = int(day_name[6:8])
        
        date_obj = date(year, month, day)
        ref_date = date(2024, 1, 1)
        days_since_ref = (date_obj - ref_date).days
        time_coords.append(days_since_ref)
    
    # Configuration optimisée des chunks pour l'année complète
    chunk_time = min(30, n_days)  # 30 jours par chunk temporel
    chunk_lat = 180  # Chunks spatiaux plus petits pour l'efficacité
    chunk_lon = 360
    
    print(f"Configuration chunks : {chunk_time} jours × {chunk_lat} lat × {chunk_lon} lon")
    
    # Crée le store Zarr
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    
    # Crée les coordonnées
    root.create_dataset('time', data=np.array(time_coords, dtype=np.int32), 
                       chunks=(chunk_time,), compressor=zarr.Blosc(cname='zstd', clevel=6))
    root.create_dataset('lat', data=lat.astype(np.float32), 
                       chunks=(chunk_lat,), compressor=zarr.Blosc(cname='zstd', clevel=6))
    root.create_dataset('lon', data=lon.astype(np.float32), 
                       chunks=(chunk_lon,), compressor=zarr.Blosc(cname='zstd', clevel=6))
    
    # Crée les arrays pour chaque variable avec compression optimale
    compressor = zarr.Blosc(cname='zstd', clevel=6, shuffle=2)  # zstd + bit shuffle
    
    variables = {}
    for var_name in asc_patterns.keys():
        variables[var_name] = root.create_dataset(
            var_name,
            shape=(n_days, n_lat, n_lon),
            chunks=(chunk_time, chunk_lat, chunk_lon),
            dtype=np.float32,
            compressor=compressor,
            fill_value=np.nan
        )
    
    # Ajoute les métadonnées
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
    
    # Métadonnées globales
    root.attrs['title'] = 'Annual Sea Surface Temperature Dataset'
    root.attrs['source'] = 'DMI satellite data conversion'
    root.attrs['creation_date'] = str(date.today())
    root.attrs['dimensions'] = f"{n_days} days x {n_lat} lat x {n_lon} lon"
    
    # Métadonnées des coordonnées
    root['time'].attrs['standard_name'] = 'time'
    root['time'].attrs['units'] = 'days since 2024-01-01'
    root['lat'].attrs['standard_name'] = 'latitude'
    root['lat'].attrs['units'] = 'degrees_north'
    root['lon'].attrs['standard_name'] = 'longitude'
    root['lon'].attrs['units'] = 'degrees_east'
    
    # Métadonnées des variables
    for var_name in asc_patterns.keys():
        if var_name in sensor_info:
            long_name, sensor = sensor_info[var_name]
            variables[var_name].attrs['long_name'] = long_name
            variables[var_name].attrs['standard_name'] = 'sea_surface_temperature'
            variables[var_name].attrs['units'] = 'celsius'
            variables[var_name].attrs['_FillValue'] = np.nan
            variables[var_name].attrs['sensor'] = sensor
    
    # Traitement jour par jour avec progress bar
    missing_files = 0
    errors = 0
    
    for day_idx, day_dir in enumerate(tqdm(day_dirs, desc="Écriture Zarr streaming", unit="jour")):
        day_name = day_dir.name
        
        day_missing = 0
        day_errors = 0
        
        # Traite chaque variable pour ce jour
        for var_name, pattern in asc_patterns.items():
            asc_filename = day_name + pattern
            asc_path = day_dir / asc_filename
            
            if not asc_path.exists():
                day_missing += 1
                missing_files += 1
                continue
                
            try:
                # Lit le fichier (mode silencieux)
                data = read_ascii_file(asc_path, verbose=False)
                
                # Écrit directement dans le zarr (pas de stockage en RAM)
                variables[var_name][day_idx, :, :] = data.astype(np.float32)
                
            except Exception as e:
                day_errors += 1
                errors += 1
                tqdm.write(f"Erreur {day_name}/{var_name}: {e}")
        
        # Progress update occasionnel
        if day_idx % 10 == 0 or day_missing > 0 or day_errors > 0:
            success = 8 - day_missing - day_errors
            tqdm.write(f"Jour {day_name}: {success}/8 fichiers OK, {day_missing} manquants, {day_errors} erreurs")
    
    # Ferme le store
    store.close()
    
    print(f"\n=== ZARR STREAMING TERMINÉ ===")
    print(f"Fichier créé : {output_path}")
    print(f"Variables : {list(asc_patterns.keys())}")
    print(f"Statistiques : {missing_files} fichiers manquants, {errors} erreurs")
    
    return output_path

def create_xarray_zarr_streaming(day_dirs, asc_patterns, lon, lat, output_path):
    """
    Alternative avec xarray : crée le zarr vide puis l'écrit par chunks
    """
    print(f"Création Zarr avec xarray streaming : {output_path}")
    
    # Crée les coordonnées temporelles
    time_coords = []
    for day_dir in day_dirs:
        day_name = day_dir.name
        year = int(day_name[:4])
        month = int(day_name[4:6])
        day = int(day_name[6:8])
        
        date_obj = date(year, month, day)
        ref_date = date(2024, 1, 1)
        days_since_ref = (date_obj - ref_date).days
        time_coords.append(days_since_ref)
    
    # Crée un dataset vide avec les bonnes dimensions
    n_days = len(day_dirs)
    data_vars = {}
    
    for var_name in asc_patterns.keys():
        # Crée des arrays vides remplis de NaN
        data_vars[var_name] = (["time", "lat", "lon"], 
                              np.full((n_days, len(lat), len(lon)), np.nan, dtype=np.float32))
    
    ds = xr.Dataset(
        data_vars,
        coords={
            "lon": (["lon"], lon.astype(np.float32)),
            "lat": (["lat"], lat.astype(np.float32)), 
            "time": (["time"], np.array(time_coords, dtype=np.int32))
        }
    )
    
    # Ajoute les métadonnées (même code que plus haut)
    # ... (métadonnées)
    
    # Configuration optimisée pour xarray + zarr
    encoding = {}
    for var_name in asc_patterns.keys():
        encoding[var_name] = {
            'chunks': (30, 180, 360),  # 30 jours × 180 lat × 360 lon
            'compressor': zarr.Blosc(cname='zstd', clevel=6),
            'dtype': 'float32'
        }
    
    # Sauvegarde le squelette
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)
    
    ds.to_zarr(output_path, encoding=encoding, consolidated=True)
    ds.close()
    
    # Maintenant remplit jour par jour
    print("Remplissage des données jour par jour...")
    
    missing_files = 0
    errors = 0
    
    for day_idx, day_dir in enumerate(tqdm(day_dirs, desc="Remplissage streaming", unit="jour")):
        day_name = day_dir.name
        
        # Ouvre le zarr en mode append
        ds_zarr = xr.open_zarr(output_path)
        
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
                data = read_ascii_file(asc_path, verbose=False)
                # Mise à jour du zarr
                ds_zarr[var_name][day_idx, :, :] = data.astype(np.float32)
                
            except Exception as e:
                day_errors += 1
                errors += 1
        
        ds_zarr.close()
    
    print(f"Streaming xarray terminé : {missing_files} manquants, {errors} erreurs")
    return output_path

def main():
    # Configuration
    root_dir = Path("/dmidata/users/malegu/data/squashfs-root")
    output_dir = Path("/home/malegu/data/netcdf_annual")
    output_dir.mkdir(exist_ok=True)
    
    # Récupère tous les dossiers de jours
    day_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    print(f"Trouvé {len(day_dirs)} jours de données")
    
    # TEST : choix du nombre de jours
    day_dirs = day_dirs[:100]  # TEST: 100 jours
    # day_dirs = day_dirs  # Année complète !
    print(f"Traitement de {len(day_dirs)} jours (streaming)")
    
    # Définit les fichiers à traiter
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
    
    # Lit la grille de référence
    first_day = day_dirs[0]
    reference_nc = first_day / "20240101120000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc"
    
    if not reference_nc.exists():
        print(f"ERREUR : NetCDF de référence introuvable : {reference_nc}")
        return
        
    lon, lat, _ = read_reference_netcdf(reference_nc)
    
    # Crée le fichier Zarr en streaming
    output_path = output_dir / f"sst_annual_2024_{len(day_dirs)}days_streaming.zarr"
    
    print(f"\n=== DÉMARRAGE STREAMING ZARR ===")
    print(f"Mémoire requise : ~{len(lat) * len(lon) * 4 / (1024**2):.1f} MB par jour")
    print(f"Fichier final estimé : ~{len(day_dirs) * 100:.0f} MB")
    
    result_path = create_annual_zarr_streaming(day_dirs, asc_patterns, lon, lat, output_path)
    
    # Affiche la taille finale
    final_size = sum(f.stat().st_size for f in result_path.rglob('*') if f.is_file()) / (1024**2)
    print(f"\n=== RÉSULTAT FINAL ===")
    print(f"Fichier Zarr : {result_path} ({final_size:.1f} MB)")
    
    # Test de lecture rapide
    print("\nTest de lecture...")
    try:
        ds = xr.open_zarr(result_path)
        print(f"Lecture OK : {ds.dims}")
        print(f"Variables : {list(ds.data_vars.keys())}")
        print(f"Période : {ds.time.min().values} à {ds.time.max().values} jours")
        ds.close()
    except Exception as e:
        print(f"Erreur de lecture : {e}")

if __name__ == "__main__":
    main()
