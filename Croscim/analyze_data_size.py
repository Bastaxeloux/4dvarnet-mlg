#!/usr/bin/env python3
"""
Script pour analyser la taille des données chargées vs taille théorique minimale
"""
import zarr
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, '/home/malegu/4D-MLG/Croscim')
from contrib.SST.data_multires import XrDatasetMultiResTrain, BaseDataModuleMultiRes
from contrib.SST.data import VAR_GROUPS

from datetime import datetime

def get_size_mb(obj):
    """Calcule la taille en MB d'un objet numpy/dict récursivement"""
    if isinstance(obj, np.ndarray):
        return obj.nbytes / 1e6
    elif isinstance(obj, dict):
        return sum(get_size_mb(v) for v in obj.values())
    elif hasattr(obj, '__dict__'):
        # Pour les TrainingItem (NamedTuple a des attributs)
        total = 0
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, (np.ndarray, dict)):
                        total += get_size_mb(attr)
                except:
                    pass
        return total
    else:
        return 0

def analyze_zarr_file():
    """Analyse la taille d'un fichier Zarr journalier"""
    print("ANALYSE FICHIER ZARR JOURNALIER")
    mounted_dir = Path("/home/malegu/4D-MLG/Croscim/data/mounted")
    zarr_files = list(mounted_dir.glob("**/*_x1.zarr"))
    
    if not zarr_files:
        print(f"Aucun fichier Zarr trouvé dans {mounted_dir}")
        return None
    
    zarr_path = zarr_files[0]
    print(f"Analyse de: {zarr_path}")
    
    store = zarr.open(str(zarr_path), mode='r')
    
    total_size = 0
    print("\nVariables dans le fichier:")
    for var_name in store.array_keys():
        arr = store[var_name]
        size_mb = arr.nbytes / 1e6
        total_size += size_mb
        print(f"  {var_name:20s}: shape={arr.shape}, dtype={arr.dtype}, size={size_mb:8.2f} MB")
    
    print(f"\n  TOTAL FICHIER: {total_size:.2f} MB")
    return total_size

def analyze_patch_size():
    print("\nTAILLE THEORIQUE D'UN PATCH 256x256 sur 15 TIMESTEPS")
    patch_size = 256
    n_timesteps = 15

    n_sat_vars = 0
    for sat_name, vars_list in VAR_GROUPS.items():
        n_sat_vars += len(vars_list)
        print(f"  {sat_name}: {len(vars_list)} vars")
    
    # sea_ice_fraction
    n_covariates = 1
    # lat, lon, time, surfmask
    n_spatial = 4
    # tgt_sst + (tgt_slstr_av, tgt_aasti_av)
    n_target = 3
    n_coords = 3
    
    # Taille pour x1 (full resolution)
    size_x1_3d = n_sat_vars * n_timesteps * patch_size * patch_size * 4 / 1e6
    size_x1_2d = (n_spatial + n_target) * patch_size * patch_size * 4 / 1e6
    size_x1_coords = (n_timesteps + patch_size + patch_size) * 8 / 1e6
    size_x1_covariate = n_covariates * n_timesteps * patch_size * patch_size * 4 / 1e6
    size_x1_total = size_x1_3d + size_x1_2d + size_x1_coords + size_x1_covariate
    
    print(f"\n  Patch x1 (256x256, 15 timesteps):")
    print(f"    Satellite vars (3D): {size_x1_3d:.2f} MB")
    print(f"    Covariate (3D): {size_x1_covariate:.2f} MB")
    print(f"    Spatial/target (2D): {size_x1_2d:.2f} MB")
    print(f"    Coords (1D): {size_x1_coords:.2f} MB")
    print(f"    TOTAL x1: {size_x1_total:.2f} MB")
    
    # Pour multi-résolution: x1 + x3 + x10
    size_multires = size_x1_total * 3  # Approximation (x3 et x10 même taille de patch)
    print(f"\n  Multi-résolution (x1 + x3 + x10): {size_multires:.2f} MB")
    
    return size_x1_total, size_multires

def analyze_actual_data():
    """Charge réellement un sample et mesure sa taille"""
    print("\nTAILLE REELLE D'UN SAMPLE CHARGE")
    
    dm = BaseDataModuleMultiRes(
        sst_daily_paths="/home/malegu/4D-MLG/Croscim/data/mounted",
        covariates_paths=[],
        multires=[10, 3, 1],
        precomputed=True,
        mask_path=None,
        domain_name="sst_multires",
        res=5.0,
        pads=[False, False, False],
        domains={
            'train': {'time': slice('2024-01-01', '2024-01-10')},
            'val': {'time': slice('2024-01-11', '2024-01-15')},
            'test': {'time': slice('2024-01-16', '2024-01-20')},
        },
        xrds_kw={
            'patch_dims': {'time': 15, 'lat': 256, 'lon': 256},
            'strides': {'time': 3, 'lat': 64, 'lon': 64},
            'subsel_patch': False 
        },
        dl_kw={'batch_size': 1, 'num_workers': 0},
        tgt_vars=["slstr_av", "aasti_av"],
        norm_stats={
            'aasti': {'av': {'mean': -7.228, 'std': 13.843, 'type': 'zscore'}},
            'slstr': {'av': {'mean': 20.406, 'std': 9.136, 'type': 'zscore'}},
            'tgt_sst': {'mean': 11.683, 'std': 17.667, 'type': 'zscore'}
        },
        norm_stats_covs={
            'sea_ice_fraction': {'min': 0.0, 'max': 1.0, 'type': 'minmax'}
        }
    )
    
    dm.setup('fit')
    dataset = dm.train_ds
    
    print(f"\n  Dataset length: {len(dataset)}")
    print(f"  Loading sample 0...")
    sample = dataset[0]
    print(f"\n  Sample keys: {list(sample.keys())}")
    
    total_size = 0
    for res_key, res_data in sample.items():
        size = get_size_mb(res_data)
        total_size += size
        
        # Show structure
        if hasattr(res_data, '_fields'):
            print(f"\n  {res_key} (TrainingItem):")
            for field in res_data._fields:
                val = getattr(res_data, field)
                if isinstance(val, np.ndarray):
                    field_size = val.nbytes / 1e6
                    print(f"    {field:20s}: shape={val.shape:30s}, dtype={val.dtype}, size={field_size:8.2f} MB")
        elif isinstance(res_data, dict):
            print(f"\n  {res_key} (dict):")
            for key, val in res_data.items():
                if isinstance(val, np.ndarray):
                    field_size = val.nbytes / 1e6
                    print(f"    {key:20s}: shape={val.shape:30s}, dtype={val.dtype}, size={field_size:8.2f} MB")
        
        print(f"  {res_key} TOTAL: {size:.2f} MB")
    
    print(f"\n  TOTAL SAMPLE (all resolutions): {total_size:.2f} MB")
    
    return total_size

if __name__ == "__main__":
    print("\n")
    file_size = analyze_zarr_file()
    patch_size_x1, patch_size_multi = analyze_patch_size()
    try:
        actual_size = analyze_actual_data()
        print("\nCOMPARAISON")
        if file_size:
            print(f"  Fichier Zarr complet: {file_size:.2f} MB")
            print(f"  Patch théorique x1: {patch_size_x1:.2f} MB ({100*patch_size_x1/file_size:.2f}% du fichier)")
            print(f"  Patch théorique multi-res: {patch_size_multi:.2f} MB")
        print(f"  Patch réel chargé: {actual_size:.2f} MB")
        print(f"  Overhead: {actual_size - patch_size_multi:.2f} MB ({100*(actual_size - patch_size_multi)/patch_size_multi:.1f}%)")
    except Exception as e:
        print(f"\nErreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
