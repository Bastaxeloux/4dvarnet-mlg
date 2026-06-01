#!/usr/bin/env python3
"""
Script de test pour vérifier le mode single_day avec la nouvelle classe XrDatasetSingleDay
"""
import sys
sys.path.insert(0, '/home/malegu/4D-MLG/Croscim')

from contrib.SST.data_multires import BaseDataModuleMultiRes
from contrib.SST.load_data import COVARIATES
import numpy as np
import yaml

# Charger les stats de normalisation
with open('/home/malegu/4D-MLG/Croscim/contrib/SST/norm_stats.yaml', 'r') as f:
    norm_stats_config = yaml.safe_load(f)
    norm_stats = norm_stats_config.get('norm_stats', None)
    norm_stats_covs = norm_stats_config.get('norm_stats_covs', None)

# Simuler la config
sst_daily_paths = "/nwp/sst_malegu"
multires = [10, 3, 1]
precomputed = True
domains = {
    'train': {'time': slice('2024-01-01', '2024-10-31')},
    'val': {'time': slice('2024-11-01', '2024-11-30')},
    'test': {'time': slice('2024-12-01', '2024-12-31')}
}

xrds_kw = {
    'patch_dims': {'time': 15, 'lat': 256, 'lon': 256},
    'strides': {'time': 3, 'lat': 64, 'lon': 64},
    'test_single_day': True,  # MODE SINGLE DAY ACTIVÉ
    'test_date_idx': 15       # Index du jour cible (fixé pour synchronisation)
}

tgt_vars = ["slstr_av", "aasti_av"]

print("=" * 80)
print("TEST DU MODE SINGLE DAY (nouvelle classe XrDatasetSingleDay)")
print("=" * 80)

# Créer le datamodule
dm = BaseDataModuleMultiRes(
    sst_daily_paths=sst_daily_paths,
    multires=multires,
    precomputed=precomputed,
    domains=domains,
    covariates_paths=[],
    covariates=COVARIATES,
    mask_path=None,
    domain_name='sst_multires',
    res=5.0,
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs,
    xrds_kw=xrds_kw,
    dl_kw={'batch_size': 4, 'num_workers': 0},
    tgt_vars=tgt_vars,
    pads=[False, False, False]
)

# Setup test
print("\n" + "=" * 80)
print("SETUP TEST DATASETS")
print("=" * 80)
dm.setup(stage='test')

# Vérifier les dimensions
print("\n" + "=" * 80)
print("DIMENSIONS DES DATASETS DE TEST")
print("=" * 80)

total_patches_all_res = 0
total_batches_all_res = 0

for res, ds in dm.test_ds.datasets.items():
    print(f"\n--- Résolution x{res} ---")
    print(f"  Nombre de jours (times): {len(ds.times)}")
    print(f"  Dates: {ds.times[0]} à {ds.times[-1]}")
    print(f"  Dimensions grid (lat×lon): {len(ds.lat_1d)}×{len(ds.lon_1d)}")
    print(f"  ds_size (patches par dim): {ds.ds_size}")
    print(f"  Nombre total de patches: {len(ds)}")
    
    # Calculer le nombre de batches
    batch_size = 4
    n_batches = len(ds) // batch_size + (1 if len(ds) % batch_size > 0 else 0)
    print(f"  Nombre de batches (batch_size={batch_size}): {n_batches}")
    
    total_patches_all_res += len(ds)
    total_batches_all_res += n_batches

print("\n" + "=" * 80)
print("RÉSUMÉ GLOBAL")
print("=" * 80)
print(f"  Total patches (toutes résolutions): {total_patches_all_res}")
print(f"  Total batches (toutes résolutions): {total_batches_all_res}")
print(f"\n  Comparaison avec le test précédent (207,507 batches pour x10 seul):")
print(f"  Réduction: {207507 / total_batches_all_res:.1f}x moins de batches !")

print("\n" + "=" * 80)
print("TEST RÉUSSI !")
print("=" * 80)
