import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path
import warnings
import yaml
import functools as ft
warnings.filterwarnings('ignore')
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from contrib.SST.data_multires import XrDatasetMultiResTrain
from contrib.SST.data import TrainingItem, BaseDataModule
from contrib.SST.load_data import VAR_GROUPS, COVARIATES
from omegaconf import OmegaConf

def plot_patch_bounds(ax, lat_bounds, lon_bounds, color, label, alpha=0.3, linewidth=3):
    """
    Plot un rectangle représentant les bounds géographiques d'un patch (SANS TEXTE).
    
    Args:
        ax: matplotlib axis
        lat_bounds: tuple (lat_min, lat_max)
        lon_bounds: tuple (lon_min, lon_max)
        color: couleur du rectangle
        label: label pour la légende
        alpha: transparence du remplissage
        linewidth: épaisseur du contour
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    rect = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',  # Pas de remplissage, juste le contour
        label=label
    )
    ax.add_patch(rect)


def verify_nesting(lat_x1, lon_x1, lat_x3, lon_x3, lat_x10, lon_x10):
    """
    Vérifie que x1 inclus dans x3 inclus dans x10 géographiquement.
    Returns:
        bool: True si l'imbrication est correcte
        str: Message de diagnostic
    """
    tol = 0.2  # Tolérance
    # Vérifier x1 inclus dans x3 (avec tolérance)
    x1_in_x3_lat = (lat_x3[0] - tol <= lat_x1[0]) and (lat_x1[1] <= lat_x3[1] + tol)
    x1_in_x3_lon = (lon_x3[0] - tol <= lon_x1[0]) and (lon_x1[1] <= lon_x3[1] + tol)    
    # Vérifier x3 inclus dans x10 (avec tolérance)
    x3_in_x10_lat = (lat_x10[0] - tol <= lat_x3[0]) and (lat_x3[1] <= lat_x10[1] + tol)
    x3_in_x10_lon = (lon_x10[0] - tol <= lon_x3[0]) and (lon_x3[1] <= lon_x10[1] + tol)
    
    all_good = x1_in_x3_lat and x1_in_x3_lon and x3_in_x10_lat and x3_in_x10_lon
    
    msg = []
    if not x1_in_x3_lat:
        msg.append(f"x1 NOT IN x3 (lat): x1={lat_x1}, x3={lat_x3}")
    if not x1_in_x3_lon:
        msg.append(f"x1 NOT IN x3 (lon): x1={lon_x1}, x3={lon_x3}")
    if not x3_in_x10_lat:
        msg.append(f"x3 NOT IN x10 (lat): x3={lat_x3}, x10={lat_x10}")
    if not x3_in_x10_lon:
        msg.append(f"x3 NOT IN x10 (lon): x3={lon_x3}, x10={lon_x10}")
    
    if all_good:
        msg.append("Imbrication géographique CORRECTE !")
    
    return all_good, "\n".join(msg)


def test_patch_at_index(dataset, idx, position_name):
    """
    Teste l'extraction d'un patch à un indice donné.
    
    Args:
        dataset: Instance de XrDatasetMultiResTrain
        idx: Index du patch à tester
        position_name: Nom de la position (ex: "Nord", "Milieu", etc.)
    
    Returns:
        dict: Résultats du test avec patches et bounds
    """
    print(f"\n{'='*80}")
    print(f"TEST: Position {position_name} (idx={idx})")
    print(f"{'='*80}")
    
    try:
        sample = dataset[idx]
        assert 'patch_x1' in sample, "patch_x1 manquant!"
        assert 'patch_x3' in sample, "patch_x3 manquant!"
        assert 'patch_x10' in sample, "patch_x10 manquant!"
        
        # Le dataset applique déjà saved_postpro_fn automatiquement
        # Pas besoin de l'appliquer manuellement
        
        # Extraire bounds géographiques
        # NOTE: Après postpro_fn, patch_x1/x3/x10 sont des TrainingItem (namedtuple), pas des dict
        patch_x1 = sample['patch_x1']
        patch_x3 = sample['patch_x3']
        patch_x10 = sample['patch_x10']
        
        # Accéder aux attributs du TrainingItem (pas comme des dict)
        lat_x1 = (float(patch_x1.lat_geo.min()), float(patch_x1.lat_geo.max()))
        lon_x1 = (float(patch_x1.lon_geo.min()), float(patch_x1.lon_geo.max()))
        
        lat_x3 = (float(patch_x3.lat_geo.min()), float(patch_x3.lat_geo.max()))
        lon_x3 = (float(patch_x3.lon_geo.min()), float(patch_x3.lon_geo.max()))
        
        lat_x10 = (float(patch_x10.lat_geo.min()), float(patch_x10.lat_geo.max()))
        lon_x10 = (float(patch_x10.lon_geo.min()), float(patch_x10.lon_geo.max()))
        
        # Vérifier tailles
        shape_x1 = patch_x1.lat_geo.shape
        shape_x3 = patch_x3.lat_geo.shape
        shape_x10 = patch_x10.lat_geo.shape
        
        print(f"\nTAILLES:")
        print(f"  x1:  {shape_x1} {'OK' if shape_x1 == (256, 256) else 'WRONG'}")
        print(f"  x3:  {shape_x3} {'OK' if shape_x3 == (256, 256) else 'WRONG'}")
        print(f"  x10: {shape_x10} {'OK' if shape_x10 == (256, 256) else 'WRONG'}")
        
        # Afficher bounds
        print(f"\nBOUNDS GÉOGRAPHIQUES:")
        print(f"  x1:  lat=[{lat_x1[0]:6.2f}, {lat_x1[1]:6.2f}]  lon=[{lon_x1[0]:7.2f}, {lon_x1[1]:7.2f}]  span=({lat_x1[1]-lat_x1[0]:.2f}° x {lon_x1[1]-lon_x1[0]:.2f}°)")
        print(f"  x3:  lat=[{lat_x3[0]:6.2f}, {lat_x3[1]:6.2f}]  lon=[{lon_x3[0]:7.2f}, {lon_x3[1]:7.2f}]  span=({lat_x3[1]-lat_x3[0]:.2f}° x {lon_x3[1]-lon_x3[0]:.2f}°)")
        print(f"  x10: lat=[{lat_x10[0]:6.2f}, {lat_x10[1]:6.2f}]  lon=[{lon_x10[0]:7.2f}, {lon_x10[1]:7.2f}]  span=({lat_x10[1]-lat_x10[0]:.2f}° x {lon_x10[1]-lon_x10[0]:.2f}°)")
        
        # Vérifier imbrication
        is_nested, nesting_msg = verify_nesting(lat_x1, lon_x1, lat_x3, lon_x3, lat_x10, lon_x10)
        print(f"\nIMBRICATION:")
        print(nesting_msg)
        
        return {
            'success': True,
            'position': position_name,
            'idx': idx,
            'lat_x1': lat_x1,
            'lon_x1': lon_x1,
            'lat_x3': lat_x3,
            'lon_x3': lon_x3,
            'lat_x10': lat_x10,
            'lon_x10': lon_x10,
            'shapes': {
                'x1': shape_x1,
                'x3': shape_x3,
                'x10': shape_x10
            },
            'is_nested': is_nested,
            'patches': sample
        }
        
    except Exception as e:
        print(f"\nERREUR lors de l'extraction du patch:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'position': position_name,
            'idx': idx,
            'error': str(e)
        }


def plot_all_results(results, output_file='test_encompassing_patches.png'):
    """
    Crée une figure avec carte globale SST et rectangles des patchs.
    
    Args:
        results: Liste de dicts avec résultats de test
        output_file: Nom du fichier de sortie
    """
    n_tests = len([r for r in results if r['success']])
    if n_tests == 0:
        print("\nAucun test réussi, pas de plot généré.")
        return
    
    # Créer figure avec subplots pour chaque test
    fig, axes = plt.subplots(n_tests, 1, figsize=(20, 8 * n_tests))
    if n_tests == 1:
        axes = [axes]  # Pour avoir toujours une liste
    
    plot_idx = 0
    for result in results:
        if not result['success']:
            continue
        
        ax = axes[plot_idx]
        patch_x10 = result['patches']['patch_x10']
        sst_data = None
        for var_name in ['tgt_sst']:
            # Accéder aux attributs du TrainingItem avec hasattr/getattr
            if hasattr(patch_x10, var_name):
                sst_data = getattr(patch_x10, var_name)
                print(f"    Utilisation de '{var_name}' pour la visualisation")
                break
        
        if sst_data is None:
            print(f"AUCUNE variable SST trouvée!")
        
        if sst_data is not None:
            # Prendre le timestep central pour visualisation
            if sst_data.ndim == 3:  # (time, lat, lon)
                sst_slice = sst_data[2, :, :]
            else:
                sst_slice = sst_data
            
            # DIAGNOSTIC: Afficher les statistiques de chaque variable SST
            print(f"\n  DIAGNOSTIC SST pour {result['position']}:")
            
            # Stocker les slices pour analyse ultérieure
            var_slices = {}
            for var_name in ['tgt_sst', 'slstr_av', 'aasti_av']:
                # Accéder aux attributs du TrainingItem avec hasattr/getattr
                if hasattr(patch_x10, var_name):
                    var_data = getattr(patch_x10, var_name)
                    if var_data.ndim == 3:
                        var_slice = var_data[2, :, :]
                    else:
                        var_slice = var_data
                    var_slices[var_name] = var_slice
                    
                    n_valid = np.sum(~np.isnan(var_slice))
                    n_total = var_slice.size
                    pct = 100 * n_valid / n_total
                    print(f"    {var_name:12s}: {n_valid:5d}/{n_total} valid ({pct:5.1f}%)", end="")
                    if n_valid > 0:
                        print(f"  range=[{np.nanmin(var_slice):6.2f}, {np.nanmax(var_slice):6.2f}]")
                    else:
                        print(f"  ALL NaN")
                else:
                    print(f"    {var_name:12s}: NOT AVAILABLE")
            
            # Analyser le chevauchement spatial
            if 'slstr_av' in var_slices and 'aasti_av' in var_slices:
                slstr_valid = ~np.isnan(var_slices['slstr_av'])
                aasti_valid = ~np.isnan(var_slices['aasti_av'])
                
                overlap = np.sum(slstr_valid & aasti_valid)
                slstr_only = np.sum(slstr_valid & ~aasti_valid)
                aasti_only = np.sum(~slstr_valid & aasti_valid)
                
                print(f"\n    Chevauchement spatial:")
                print(f"      SLSTR seul    : {slstr_only:5d} pixels")
                print(f"      AASTI seul    : {aasti_only:5d} pixels")
                print(f"      Chevauchement : {overlap:5d} pixels")
                
                if 'tgt_sst' in var_slices:
                    expected_valid = slstr_only + aasti_only + overlap
                    actual_valid = np.sum(~np.isnan(var_slices['tgt_sst']))
                    print(f"\n    Validation fusion:")
                    print(f"      Pixels attendus : {expected_valid}")
                    print(f"      Pixels observés : {actual_valid}")
                    if expected_valid == actual_valid:
                        print(f"      Nombre correct")
                    else:
                        print(f"      Différence = {actual_valid - expected_valid}")
            
            # Plot SST en fond
            # Accéder aux attributs du TrainingItem
            lat_geo_x10 = patch_x10.lat_geo
            lon_geo_x10 = patch_x10.lon_geo
            
            im = ax.pcolormesh(
                lon_geo_x10, lat_geo_x10, sst_slice,
                cmap='RdYlBu_r', 
                vmin=-3, vmax=2,
                alpha=0.7,
                shading='auto')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label('SST (°C)', fontsize=10)
            
        plot_patch_bounds(ax, result['lat_x10'], result['lon_x10'], 'red', 'x10 (50km)', linewidth=4)
        plot_patch_bounds(ax, result['lat_x3'], result['lon_x3'], 'orange', 'x3 (15km)', linewidth=3)
        plot_patch_bounds(ax, result['lat_x1'], result['lon_x1'], 'lime', 'x1 (5km)', linewidth=2)
        
        # Configuration
        ax.set_xlabel('Longitude (°)', fontsize=14)
        ax.set_ylabel('Latitude (°)', fontsize=14)
        ax.set_title(
            f"{result['position']} (idx={result['idx']}) - "
            f"{'NESTED' if result['is_nested'] else 'NOT NESTED'}",
            fontsize=16, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Fixer limites au patch x10 pour voir tout le contexte
        ax.set_xlim(result['lon_x10'][0], result['lon_x10'][1])
        ax.set_ylim(result['lat_x10'][0], result['lat_x10'][1])
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure sauvegardée: {output_file}")


def main():
    print("\nInitialisation du dataset...")
    norm_stats_path = project_root / 'contrib' / 'SST' / 'norm_stats.yaml'
    print(f"Chargement des statistiques depuis {norm_stats_path}...")
    with open(norm_stats_path, 'r') as f:
        norm_stats = yaml.safe_load(f)
    
    # Créer liste de fichiers pour les 3 résolutions
    dates = [f"202401{day:02d}12" for day in range(1, 16)]
    sst_paths_dict = {
        1: [f"/nwp/sst_malegu/data_2024/{date}_x1.zarr" for date in dates],
        3: [f"/nwp/sst_malegu/data_2024/{date}_x3.zarr" for date in dates],
        10: [f"/nwp/sst_malegu/data_2024/{date}_x10.zarr" for date in dates],
    }

    mask = xr.DataArray(
        np.ones((3600, 7200), dtype=np.float32),
        dims=['lat', 'lon'])
    times = np.array([pd.to_datetime(date, format='%Y%m%d%H') for date in dates])
    
    # Créer un dataset temporaire pour pouvoir appeler post_fn()
    # On va créer le dataset sans postpro_fn, puis injecter norm_stats et créer postpro_fn
    dataset = XrDatasetMultiResTrain(
        sst_daily_paths=sst_paths_dict,
        tgt_vars=['slstr_av', 'aasti_av'],
        mask=mask,  
        times=times, 
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
    
    total_patches = len(dataset)
    print(f"   Dataset chargé: {total_patches} patchs disponibles")
    
    # il faut créer un mini DataModule pour obtenir post_fn()
    mini_datamodule = BaseDataModule(sst_paths=sst_paths_dict[1], covariates_paths=[], covariates=COVARIATES,
        tgt_vars=['slstr_av', 'aasti_av'], mask_path=None, domain_name='test', domains={}, xrds_kw={}, 
        dl_kw={}, norm_stats=norm_stats['norm_stats'],  # Extraire la sous-clé 'norm_stats'
        norm_stats_covs=norm_stats.get('norm_stats_covs', {}),  # Extraire 'norm_stats_covs'
        res=5.0, resize=1)
    
    # Obtenir post_fn() et l'injecter (comme setup() en training)
    postpro_fn = mini_datamodule.post_fn(rand_obs=False)
    dataset.saved_postpro_fn = postpro_fn
    print("   postpro_fn créé via mini DataModule et injecté")
    
    # ATTENTION : indices semblent etre inversés car idx faibles = Antarctique (Sud), idx élevés = Arctique (Nord)
    idx_south = 5  # Pôle SUD (Antarctique)
    idx_middle = total_patches // 2  # Patch au milieu
    idx_north = total_patches - 7  # Pôle NORD (Arctique)
    idx_west = 1417  # À ajuster manuellement
    idx_east = 4322  # À ajuster manuellement
    
    test_cases = [
        (idx_north, "NORD"),
        (idx_middle, "MILIEU"),
        (idx_south, "SUD"),
        (idx_west, "OUEST (à ajuster)"),
        (idx_east, "EST (à ajuster)")]
    
    print(f"\nIndices de test:")
    for idx, name in test_cases:
        print(f"   {name:30s} : idx={idx}")
    
    results = []
    for idx, position_name in test_cases:
        result = test_patch_at_index(dataset, idx, position_name)
        results.append(result)

    print(f"\n{'='*80}")
    print(f"RÉSUMÉ DES TESTS")
    print(f"{'='*80}")
    
    n_success = sum(1 for r in results if r['success'])
    n_nested = sum(1 for r in results if r.get('is_nested', False))
    
    print(f"\nTests réussis: {n_success}/{len(results)}")
    print(f"Patchs correctement imbriqués: {n_nested}/{n_success}")
    
    if n_success < len(results):
        print(f"\nÉchecs:")
        for r in results:
            if not r['success']:
                print(f"   {r['position']:30s} (idx={r['idx']}): {r['error']}")
    if n_nested < n_success:
        print(f"\nPatchs NON imbriqués correctement:")
        for r in results:
            if r['success'] and not r.get('is_nested', False):
                print(f"   {r['position']:30s} (idx={r['idx']})")
    if n_success > 0:
        plot_all_results(results)

if __name__ == "__main__":
    main()
