import numpy as np
import sys
from pathlib import Path
import warnings
import yaml
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from contrib.SST.data_multires import XrDatasetMultiResTrain
from contrib.SST.data import TrainingItem, BaseDataModule
from contrib.SST.load_data import VAR_GROUPS, COVARIATES

def check_patch_nesting(patch_inner, patch_outer, tolerance_deg=2.0):
    lat_inner_min = float(patch_inner['lat_geo'].min())
    lat_inner_max = float(patch_inner['lat_geo'].max())
    lon_inner_min = float(patch_inner['lon_geo'].min())
    lon_inner_max = float(patch_inner['lon_geo'].max())
    
    lat_outer_min = float(patch_outer['lat_geo'].min())
    lat_outer_max = float(patch_outer['lat_geo'].max())
    lon_outer_min = float(patch_outer['lon_geo'].min())
    lon_outer_max = float(patch_outer['lon_geo'].max())
    
    # Calculer les débordements (positif = débordement, négatif = OK)
    lat_error_min = lat_outer_min - lat_inner_min  # Si négatif, inner dépasse en bas
    lat_error_max = lat_inner_max - lat_outer_max  # Si positif, inner dépasse en haut
    lon_error_min = lon_outer_min - lon_inner_min
    lon_error_max = lon_inner_max - lon_outer_max
    
    lat_error = max(lat_error_min, lat_error_max)
    lon_error = max(lon_error_min, lon_error_max)
    
    # Tolérance stricte pour "parfait"
    strict_tol = 0.1
    
    if lat_error <= strict_tol and lon_error <= strict_tol:
        return True, lat_error, lon_error, "PERFECT"
    elif lat_error <= tolerance_deg and lon_error <= tolerance_deg:
        return False, lat_error, lon_error, "ACCEPTABLE"
    else:
        return False, lat_error, lon_error, "FAILED"


def test_nested_patches(num_samples=100, tolerance_deg=2.0):
    """
    Teste l'imbrication des patches multi-résolution.
    """
    print("=" * 80)
    print("TEST D'IMBRICATION DES PATCHES MULTI-RÉSOLUTION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Nombre de samples: {num_samples}")
    print(f"  - Tolérance acceptable: {tolerance_deg}°")
    print(f"  - Tolérance parfaite: 0.1°")
    print()
    
    # 1. Charger la config (même pattern que test_encompassing_patches)
    print("Chargement de la configuration...")
    
    config_path = project_root / "config" / "xp" / "SST" / "multires_lite.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dm_config = config['datamodule']
    sst_path = dm_config['sst_daily_paths']
    
    print(f"Chemin SST: {sst_path}")
    
    # 2. Créer le dataset (même pattern que test_encompassing_patches)
    print("Création du dataset...")
    
    import pandas as pd
    import xarray as xr
    
    # Créer liste de fichiers pour les 3 résolutions (prendre quelques jours de janvier 2024)
    dates = [f"202401{day:02d}12" for day in range(1, 16)]  # 15 jours
    sst_paths_dict = {
        1: [f"{sst_path}/data_2024/{date}_x1.zarr" for date in dates],
        3: [f"{sst_path}/data_2024/{date}_x3.zarr" for date in dates],
        10: [f"{sst_path}/data_2024/{date}_x10.zarr" for date in dates],
    }
    
    print(f"Fichiers configurés: {len(dates)} jours")
    print(f"Résolutions: {list(sst_paths_dict.keys())}")
    
    # Créer un masque bidon (tout à 1)
    mask = xr.DataArray(
        np.ones((3600, 7200), dtype=np.float32),
        dims=['lat', 'lon'])
    times_array = np.array([pd.to_datetime(date, format='%Y%m%d%H') for date in dates])
    
    # Créer le dataset
    dataset = XrDatasetMultiResTrain(
        sst_daily_paths=sst_paths_dict,
        tgt_vars=['slstr_av', 'aasti_av'],
        mask=mask,  
        times=times_array, 
        patch_dims={'time': 15, 'lat': 256, 'lon': 256},
        domain_limits=None,
        strides={'time': 3, 'lat': 64, 'lon': 64},
        resize=1,
        multires=[1, 3, 10], 
        precomputed=True, 
        load_data=False,
        enable_patch_filtering=False,  
        res=5.0,  
        pad=False,
        stride_test=False)
    
    print(f"Dataset créé: {len(dataset)} samples disponibles")
    print(f"Multires: {dataset.multires}")
    print()
    
    # 3. Tirer des samples aléatoires et vérifier l'imbrication
    print("Test des samples aléatoires...")
    print("-" * 80)
    
    results = {
        'perfect': 0,      # Imbrication parfaite (< 0.1°)
        'acceptable': 0,   # Imbrication acceptable (< 2°)
        'failed': 0,       # Échec (> 2°)
    }
    
    failed_samples = []  # Pour afficher les détails des échecs
    
    for i in range(num_samples):
        # Tirer un index aléatoire
        idx = np.random.randint(0, len(dataset))
        
        try:
            # Appeler __getitem__
            sample = dataset[idx]
            
            # Vérifier que les 3 résolutions sont présentes
            if 'patch_x1' not in sample or 'patch_x3' not in sample or 'patch_x10' not in sample:
                print(f"Sample {i} (idx={idx}): ERREUR - Patches manquants!")
                results['failed'] += 1
                continue
            
            patch_x1 = sample['patch_x1']
            patch_x3 = sample['patch_x3']
            patch_x10 = sample['patch_x10']
            is_perfect_13, lat_err_13, lon_err_13, status_13 = check_patch_nesting(patch_x1, patch_x3, tolerance_deg)
            is_perfect_310, lat_err_310, lon_err_310, status_310 = check_patch_nesting(patch_x3, patch_x10, tolerance_deg)

            if status_13 == "PERFECT" and status_310 == "PERFECT":
                results['perfect'] += 1
                status = "PERFECT"
            elif status_13 == "FAILED" or status_310 == "FAILED":
                results['failed'] += 1
                status = "FAILED"
                failed_samples.append({
                    'sample': i,
                    'idx': idx,
                    'lat_err_13': lat_err_13,
                    'lon_err_13': lon_err_13,
                    'lat_err_310': lat_err_310,
                    'lon_err_310': lon_err_310,
                    'patch_x1': patch_x1,
                    'patch_x3': patch_x3,
                    'patch_x10': patch_x10,
                })
            else:
                results['acceptable'] += 1
                status = "ACCEPTABLE"
            
            # Afficher le résultat
            if i < 10 or status == "FAILED":  # Afficher les 10 premiers + tous les échecs
                print(f"Sample {i:3d} (idx={idx:6d}): {status}")
                print(f"  x1 in x3: lat_err={lat_err_13:+6.2f}°, lon_err={lon_err_13:+6.2f}° [{status_13}]")
                print(f"  x3 in x10: lat_err={lat_err_310:+6.2f}°, lon_err={lon_err_310:+6.2f}° [{status_310}]")
        
        except Exception as e:
            print(f"Sample {i} (idx={idx}): EXCEPTION - {str(e)}")
            results['failed'] += 1
            import traceback
            traceback.print_exc()
    
    # 3. Résumé des résultats
    print()
    print("=" * 80)
    print("RÉSULTATS")
    print("=" * 80)
    print(f"Samples testés: {num_samples}")
    print(f"  Parfait (< 0.1°):        {results['perfect']:3d} ({100*results['perfect']/num_samples:5.1f}%)")
    print(f"  Acceptable (< {tolerance_deg}°):     {results['acceptable']:3d} ({100*results['acceptable']/num_samples:5.1f}%)")
    print(f"  Échec (> {tolerance_deg}°):          {results['failed']:3d} ({100*results['failed']/num_samples:5.1f}%)")
    print()
    
    # 4. Détails des échecs
    if failed_samples:
        print("=" * 80)
        print(f"DÉTAILS DES {len(failed_samples)} ÉCHECS")
        print("=" * 80)
        
        for fail in failed_samples[:5]:  # Afficher les 5 premiers échecs en détail
            print(f"\nSample {fail['sample']} (idx={fail['idx']}):")
            print(f"  x1 bounds: lat=[{fail['patch_x1']['lat_geo'].min():.2f}, {fail['patch_x1']['lat_geo'].max():.2f}], "
                  f"lon=[{fail['patch_x1']['lon_geo'].min():.2f}, {fail['patch_x1']['lon_geo'].max():.2f}]")
            print(f"  x3 bounds: lat=[{fail['patch_x3']['lat_geo'].min():.2f}, {fail['patch_x3']['lat_geo'].max():.2f}], "
                  f"lon=[{fail['patch_x3']['lon_geo'].min():.2f}, {fail['patch_x3']['lon_geo'].max():.2f}]")
            print(f"  x10 bounds: lat=[{fail['patch_x10']['lat_geo'].min():.2f}, {fail['patch_x10']['lat_geo'].max():.2f}], "
                  f"lon=[{fail['patch_x10']['lon_geo'].min():.2f}, {fail['patch_x10']['lon_geo'].max():.2f}]")
            print(f"  x1⊂x3 errors: lat={fail['lat_err_13']:+.2f}°, lon={fail['lon_err_13']:+.2f}°")
            print(f"  x3⊂x10 errors: lat={fail['lat_err_310']:+.2f}°, lon={fail['lon_err_310']:+.2f}°")
    
    # 5. Verdict final
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    success_rate = (results['perfect'] + results['acceptable']) / num_samples * 100
    
    if success_rate >= 95:
        print(f"TEST RÉUSSI ({success_rate:.1f}% de succès)")
        print("   -> __getitem__() et extract_encompassing_patch() fonctionnent correctement!")
        return True
    elif success_rate >= 80:
        print(f"TEST PARTIEL ({success_rate:.1f}% de succès)")
        print("   -> Il y a quelques problèmes d'imbrication, à investiguer.")
        return False
    else:
        print(f"TEST ÉCHOUÉ ({success_rate:.1f}% de succès)")
        print("   -> __getitem__() ou extract_encompassing_patch() sont cassés!")
        return False


if __name__ == "__main__":
    success = test_nested_patches(num_samples=100, tolerance_deg=2.0)
    sys.exit(0 if success else 1)
