#!/usr/bin/env python3

"""
Script pour analyser l'ordre et la correspondance des patches
avec les coordonnées temporelles dans la reconstruction.
"""

import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from data import XrDataset

def analyze_patch_mapping():
    """Analyser comment les patches sont mappés aux coordonnées temporelles."""
    
    # Configuration des données
    ds = xr.open_dataset("data/natl_gf_w_5nadirs.nc")
    input_da = ds['ssh']
    
    domain_test = {
        'time': slice('2012-10-01', '2012-12-20'),
        'lat': slice(33., 43.),
        'lon': slice(-65., -55.)
    }
    
    test_data = input_da.sel(**domain_test)
    print(f"Données de test shape: {test_data.shape}")
    
    # Créer le dataset
    xrds_kw = {
        'patch_dims': {'time': 15, 'lat': 240, 'lon': 240},
        'strides': {'time': 1, 'lat': 200, 'lon': 200}
    }
    test_ds = XrDataset(test_data, **xrds_kw)
    
    print(f"Nombre total de patches: {len(test_ds)}")
    
    # Simuler les batches comme dans le modèle
    batch_size = 4
    patch_to_batch_mapping = {}
    batch_info = []
    
    for i in range(0, len(test_ds), batch_size):
        batch_end = min(i + batch_size, len(test_ds))
        batch_patches = list(range(i, batch_end))
        batch_info.append({
            'batch_idx': len(batch_info),
            'patches': batch_patches,
            'size': len(batch_patches)
        })
        
        for patch_idx in batch_patches:
            patch_to_batch_mapping[patch_idx] = len(batch_info) - 1
    
    print(f"Nombre de batches: {len(batch_info)}")
    
    # Obtenir les coordonnées pour chaque patch
    coords = test_ds.get_coords()
    
    # Analyser les coordonnées temporelles de chaque patch
    print(f"\nMapping patch -> coordonnées temporelles:")
    
    for i in range(min(20, len(coords))):  # Premiers 20 patches
        time_coords = coords[i]['time'].values
        start_time = time_coords[0]
        end_time = time_coords[-1]
        
        # Trouver les indices temporels
        start_idx = np.where(test_data.time == start_time)[0][0]
        end_idx = np.where(test_data.time == end_time)[0][0]
        
        batch_idx = patch_to_batch_mapping[i]
        
        print(f"Patch {i:2d} (Batch {batch_idx}): t={start_idx:2d}-{end_idx:2d} "
              f"({str(start_time)[:10]} à {str(end_time)[:10]})")
    
    if len(coords) > 20:
        print("...")
        for i in range(len(coords)-5, len(coords)):  # Derniers 5 patches
            time_coords = coords[i]['time'].values
            start_time = time_coords[0]
            end_time = time_coords[-1]
            
            start_idx = np.where(test_data.time == start_time)[0][0]
            end_idx = np.where(test_data.time == end_time)[0][0]
            
            batch_idx = patch_to_batch_mapping[i]
            
            print(f"Patch {i:2d} (Batch {batch_idx}): t={start_idx:2d}-{end_idx:2d} "
                  f"({str(start_time)[:10]} à {str(end_time)[:10]})")
    
    # Maintenant simulons le processus de reconstruction avec marquage
    print(f"\n=== SIMULATION DE LA RECONSTRUCTION ===")
    
    # Créer des données factices où chaque patch a une valeur unique
    import torch
    tensor_batches = []
    
    for batch_info_item in batch_info:
        batch_tensors = []
        
        for patch_idx in batch_info_item['patches']:
            # Créer une donnée où seule la position centrale temporelle 
            # a la valeur du patch_idx + 1, reste à 0
            fake_data = np.zeros((1, 15, 240, 240), dtype=np.float32)
            center_time_idx = 7  # Centre de la fenêtre de 15 (index 7)
            fake_data[0, center_time_idx, :, :] = patch_idx + 1
            batch_tensors.append(torch.from_numpy(fake_data))
        
        tensor_batches.append(batch_tensors)
    
    # Reconstruction
    rec_da = test_ds.reconstruct(tensor_batches)
    print(f"Reconstruction shape: {rec_da.shape}")
    
    # Analyser quels pas de temps ont des valeurs non-nulles
    print(f"\nAnalyse des données reconstruites:")
    for t_idx in range(rec_da.shape[1]):  # Parcourir le temps
        data_slice = rec_da.isel(v0=0, time=t_idx).values
        max_val = np.max(data_slice)
        if max_val > 0:
            print(f"t={t_idx:2d} ({str(rec_da.time.values[t_idx])[:10]}): "
                  f"max_val={max_val:.0f}")
        else:
            # Compter les zéros consécutifs
            if t_idx == 0 or np.max(rec_da.isel(v0=0, time=t_idx-1).values) > 0:
                zero_start = t_idx
            if t_idx == rec_da.shape[1]-1 or np.max(rec_da.isel(v0=0, time=t_idx+1).values) > 0:
                if 'zero_start' in locals():
                    if t_idx > zero_start:
                        print(f"t={zero_start:2d}-{t_idx:2d}: tous zéros")
                    else:
                        print(f"t={t_idx:2d}: zéro")

if __name__ == "__main__":
    analyze_patch_mapping()
