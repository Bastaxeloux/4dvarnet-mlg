#!/usr/bin/env python3

"""
Script pour déboguer la géométrie des patches et comprendre 
pourquoi seuls les premiers 19 pas de temps sont générés.
"""

import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from data import XrDataset

def analyze_patch_geometry():
    """Analyser la géométrie des patches temporels."""
    
    # Configuration simplifiée - utiliser directement les données
    ds = xr.open_dataset("data/natl_gf_w_5nadirs.nc")
    input_da = ds['ssh']  # Variable SSH
    
    # Domaine de test (vraie configuration de base.yaml)
    domain_test = {
        'time': slice('2012-10-01', '2012-12-20'),
        'lat': slice(33., 43.),
        'lon': slice(-65., -55.)
    }
    
    # Filtrer selon le domaine de test
    test_data = input_da.sel(**domain_test)
    print(f"Données de test shape: {test_data.shape}")
    print(f"Dimensions temporelles: {test_data.time.values[:5]} ... {test_data.time.values[-5:]}")
    
    # Créer le dataset avec la configuration des patches
    xrds_kw = {
        'patch_dims': {'time': 15, 'lat': 240, 'lon': 240},
        'strides': {'time': 1, 'lat': 200, 'lon': 200}
    }
    test_ds = XrDataset(test_data, **xrds_kw)
    
    print(f"\nConfiguration des patches:")
    print(f"patch_dims: {test_ds.patch_dims}")
    print(f"strides: {test_ds.strides}")
    print(f"ds_size: {test_ds.ds_size}")
    print(f"Nombre total de patches: {len(test_ds)}")
    
    # Analyser quelques patches pour comprendre leurs coordonnées temporelles
    print(f"\nAnalyse des coordonnées temporelles des premiers patches:")
    
    # Obtenir les coordonnées de tous les patches
    coords = test_ds.get_coords()
    
    for i in range(min(10, len(coords))):
        time_coords = coords[i]['time'].values
        print(f"Patch {i}: temps {time_coords[0]} à {time_coords[-1]} "
              f"(indices {np.where(test_data.time == time_coords[0])[0][0]} à "
              f"{np.where(test_data.time == time_coords[-1])[0][0]})")
    
    print(f"\n... et les derniers patches:")
    for i in range(max(0, len(coords)-5), len(coords)):
        time_coords = coords[i]['time'].values
        print(f"Patch {i}: temps {time_coords[0]} à {time_coords[-1]} "
              f"(indices {np.where(test_data.time == time_coords[0])[0][0]} à "
              f"{np.where(test_data.time == time_coords[-1])[0][0]})")
    
    # Calculer la couverture temporelle totale
    all_times = set()
    for coord in coords:
        for t in coord['time'].values:
            all_times.add(t)
    
    all_times = sorted(list(all_times))
    print(f"\nCouverture temporelle totale:")
    print(f"Premier temps couvert: {all_times[0]}")
    print(f"Dernier temps couvert: {all_times[-1]}")
    print(f"Nombre de pas de temps uniques couverts: {len(all_times)}")
    
    # Vérifier les indices temporels
    original_times = test_data.time.values
    covered_indices = [np.where(original_times == t)[0][0] for t in all_times]
    print(f"Indices temporels couverts: {min(covered_indices)} à {max(covered_indices)}")
    
if __name__ == "__main__":
    analyze_patch_geometry()
