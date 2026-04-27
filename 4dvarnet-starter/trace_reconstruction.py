#!/usr/bin/env python3

"""
Script pour tracer exactement ce qui se passe dans la reconstruction
et comprendre pourquoi seuls 19 pas de temps sont générés.
"""

import numpy as np
import xarray as xr
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / "src"))

from data import XrDataset

def trace_reconstruction_process():
    """Tracer le processus de reconstruction en détail."""
    
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
    
    # Créer le dataset avec la configuration des patches
    xrds_kw = {
        'patch_dims': {'time': 15, 'lat': 240, 'lon': 240},
        'strides': {'time': 1, 'lat': 200, 'lon': 200}
    }
    test_ds = XrDataset(test_data, **xrds_kw)
    
    print(f"Nombre de patches: {len(test_ds)}")
    
    # Simuler ce qui se passe dans la reconstruction
    # Créer des données factices pour chaque patch
    fake_batches = []
    batch_size = 4
    
    for i in range(0, len(test_ds), batch_size):
        batch_items = []
        batch_end = min(i + batch_size, len(test_ds))
        
        for j in range(i, batch_end):
            # Créer une donnée factice de la forme attendue
            # En réalité cela vient du modèle
            fake_data = np.ones((1, 15, 240, 240), dtype=np.float32) * (j + 1)
            batch_items.append(fake_data)
        
        fake_batches.append(batch_items)
        print(f"Batch {len(fake_batches)-1}: {len(batch_items)} items (patches {i} à {batch_end-1})")
    
    # Maintenant testons la reconstruction
    # Convertir en format tensor-like
    import torch
    tensor_batches = []
    for batch in fake_batches:
        tensor_batch = [torch.from_numpy(item) for item in batch]
        tensor_batches.append(tensor_batch)
    
    print(f"\nLancement de la reconstruction avec {len(tensor_batches)} batches...")
    
    # Tester la reconstruction
    try:
        rec_da = test_ds.reconstruct(tensor_batches)
        print(f"Reconstruction réussie!")
        print(f"Shape de la reconstruction: {rec_da.shape}")
        print(f"Dimensions: {rec_da.dims}")
        print(f"Coordonnées temporelles: {len(rec_da.time)} pas de temps")
        print(f"Premier temps: {rec_da.time.values[0]}")
        print(f"Dernier temps: {rec_da.time.values[-1]}")
        
        # Vérifier s'il y a des valeurs non-nulles
        non_zero_times = []
        for i, t in enumerate(rec_da.time.values):
            data_slice = rec_da.isel(time=i)
            if np.any(data_slice.values != 0):
                non_zero_times.append((i, t))
        
        print(f"\nPas de temps avec des données non-nulles: {len(non_zero_times)}")
        if len(non_zero_times) > 0:
            print(f"Premiers pas de temps non-nuls: {non_zero_times[:5]}")
            if len(non_zero_times) > 5:
                print(f"Derniers pas de temps non-nuls: {non_zero_times[-5:]}")
                
    except Exception as e:
        print(f"Erreur lors de la reconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trace_reconstruction_process()
