#!/usr/bin/env python3
"""Test rapide du fix multi-résolution"""
import sys
sys.path.insert(0, '/home/malegu/4D-MLG/Croscim')

from contrib.SST.data_multires import BaseDataModuleMultiRes

print("="*70)
print("TEST DU FIX MULTI-RÉSOLUTION")
print("="*70)

# Config minimale pour tester
config = {
    'sst_daily_paths': '/nwp/sst_malegu',
    'covariates_paths': [],
    'multires': [10, 3, 1],
    'precomputed': True,
    'mask_path': None,
    'domain_name': 'test',
    'res': 5.0,
    'domains': {
        'train': {'time': {'_target_': 'builtins.slice', '_args_': ['2024-12-01', '2024-12-02']}},
        'val': {'time': {'_target_': 'builtins.slice', '_args_': ['2024-12-03', '2024-12-04']}},
        'test': {'time': {'_target_': 'builtins.slice', '_args_': ['2024-12-05', '2024-12-06']}}
    },
    'xrds_kw': {
        'patch_dims': {'time': 15, 'lat': 256, 'lon': 256},
        'strides': {'time': 7, 'lat': 100, 'lon': 100}
    },
    'dl_kw': {'batch_size': 1, 'num_workers': 0},
    'tgt_vars': ['tgt_sst']
}

try:
    print("\nCréation du DataModule...")
    dm = BaseDataModuleMultiRes(**config)
    
    print("\nSetup...")
    dm.setup('test')
    
    print("\n" + "="*70)
    print("RÉSULTAT:")
    print("="*70)
    
    # Vérifier les datasets test
    test_dls = dm.test_dataloader()
    print(f"\nNombre de test dataloaders: {len(test_dls)}")
    
    for res, dl in test_dls.items():
        ds = dl.dataset
        print(f"\n{res}:")
        print(f"  Nombre de fichiers: {len(ds.sst_daily_paths)}")
        print(f"  Premier fichier: {ds.sst_daily_paths[0]}")
        print(f"  Grid shape: {len(ds.lat_1d)}x{len(ds.lon_1d)}")
        print(f"  Nombre de patches: {len(ds)}")
        
    print("\n" + "="*70)
    print("TEST RÉUSSI !")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
