import sys
import torch
import numpy as np
import hydra
from hydra import compose, initialize_config_module

def test_specific_sample():
    """Charge un sample précis du train dataset et vérifie l'imbrication."""
    print("\n" + "="*80)
    print("TEST SAMPLE SPÉCIFIQUE DU TRAIN DATASET")
    with initialize_config_module(version_base="1.1", config_module="config"):
        cfg = compose(config_name="main", overrides=["xp=SST/multires_lite"])
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup(stage='fit')
    train_ds = dm.train_ds
    
    print(f"\nTrain Dataset Info:")
    print(f"   Length: {len(train_ds)}")
    print(f"   Multires: {train_ds.multires}")

    target_idx = 401604
    
    print(f"\nLoading sample idx={target_idx}...")
    print(f"   (Celui qui montre: x1=-26.38, x3=-133.08, x10=-179.98)")
    
    # Charger le sample
    sample = train_ds[target_idx]
    
    print("\nCOORDONNÉES OBTENUES:")
    print("-" * 80)
    
    for res in [10, 3, 1]:
        key = f"patch_x{res}"
        if key in sample:
            patch = sample[key]
            lon = patch.lon_geo[0, 0].item()
            lat = patch.lat_geo[0, 0].item()
            lon_max = patch.lon_geo[0, -1].item()
            lat_max = patch.lat_geo[-1, 0].item()
            
            print(f"\n{key}:")
            print(f"  lon_geo[0,0]   = {lon:.2f}")
            print(f"  lat_geo[0,0]   = {lat:.2f}")
            print(f"  lon_geo[0,-1]  = {lon_max:.2f}")
            print(f"  lat_geo[-1,0]  = {lat_max:.2f}")
            print(f"  Shape: {patch.lon_geo.shape}")
    
    # Vérifier l'imbrication
    print("\n" + "="*80)
    print("VÉRIFICATION D'IMBRICATION")
    print("="*80)
    
    x1_lon_min = sample['patch_x1'].lon_geo.min().item()
    x1_lon_max = sample['patch_x1'].lon_geo.max().item()
    x1_lat_min = sample['patch_x1'].lat_geo.min().item()
    x1_lat_max = sample['patch_x1'].lat_geo.max().item()
    
    x3_lon_min = sample['patch_x3'].lon_geo.min().item()
    x3_lon_max = sample['patch_x3'].lon_geo.max().item()
    x3_lat_min = sample['patch_x3'].lat_geo.min().item()
    x3_lat_max = sample['patch_x3'].lat_geo.max().item()
    
    x10_lon_min = sample['patch_x10'].lon_geo.min().item()
    x10_lon_max = sample['patch_x10'].lon_geo.max().item()
    x10_lat_min = sample['patch_x10'].lat_geo.min().item()
    x10_lat_max = sample['patch_x10'].lat_geo.max().item()
    
    print(f"\nx1 bounds: lon=[{x1_lon_min:.2f}, {x1_lon_max:.2f}], lat=[{x1_lat_min:.2f}, {x1_lat_max:.2f}]")
    print(f"x3 bounds: lon=[{x3_lon_min:.2f}, {x3_lon_max:.2f}], lat=[{x3_lat_min:.2f}, {x3_lat_max:.2f}]")
    print(f"x10 bounds: lon=[{x10_lon_min:.2f}, {x10_lon_max:.2f}], lat=[{x10_lat_min:.2f}, {x10_lat_max:.2f}]")
    
    # Check if x1 ⊂ x3
    x1_in_x3_lon = (x1_lon_min >= x3_lon_min - 0.1) and (x1_lon_max <= x3_lon_max + 0.1)
    x1_in_x3_lat = (x1_lat_min >= x3_lat_min - 0.1) and (x1_lat_max <= x3_lat_max + 0.1)
    x1_in_x3 = x1_in_x3_lon and x1_in_x3_lat
    
    # Check if x3 ⊂ x10
    x3_in_x10_lon = (x3_lon_min >= x10_lon_min - 0.1) and (x3_lon_max <= x10_lon_max + 0.1)
    x3_in_x10_lat = (x3_lat_min >= x10_lat_min - 0.1) and (x3_lat_max <= x10_lat_max + 0.1)
    x3_in_x10 = x3_in_x10_lon and x3_in_x10_lat
    
    print(f"\nx1 inclus x3 ? {'OUI' if x1_in_x3 else 'NON'}")
    if not x1_in_x3:
        if not x1_in_x3_lon:
            print(f"   Longitude: x1=[{x1_lon_min:.2f}, {x1_lon_max:.2f}] NOT IN x3=[{x3_lon_min:.2f}, {x3_lon_max:.2f}]")
        if not x1_in_x3_lat:
            print(f"   Latitude:  x1=[{x1_lat_min:.2f}, {x1_lat_max:.2f}] NOT IN x3=[{x3_lat_min:.2f}, {x3_lat_max:.2f}]")
    
    print(f"\nx3 inclus x10 ? {'OUI' if x3_in_x10 else 'NON'}")
    if not x3_in_x10:
        if not x3_in_x10_lon:
            print(f"   Longitude: x3=[{x3_lon_min:.2f}, {x3_lon_max:.2f}] NOT IN x10=[{x10_lon_min:.2f}, {x10_lon_max:.2f}]")
        if not x3_in_x10_lat:
            print(f"   Latitude:  x3=[{x3_lat_min:.2f}, {x3_lat_max:.2f}] NOT IN x10=[{x10_lat_min:.2f}, {x10_lat_max:.2f}]")
    
    print("\nTEST COMPARATIF: UN AUTRE SAMPLE ALÉATOIRE")
    
    random_idx = np.random.randint(0, len(train_ds))
    print(f"\nLoading sample idx={random_idx}...")
    
    sample2 = train_ds[random_idx]
    
    print("\nCOORDONNÉES:")
    for res in [10, 3, 1]:
        key = f"patch_x{res}"
        if key in sample2:
            patch = sample2[key]
            lon = patch.lon_geo[0, 0].item()
            lat = patch.lat_geo[0, 0].item()
            print(f"{key}: lon_geo[0,0]={lon:.2f}, lat_geo[0,0]={lat:.2f}")
    
    return True

if __name__ == "__main__":
    result = test_specific_sample()
    sys.exit(0 if result else 1)
