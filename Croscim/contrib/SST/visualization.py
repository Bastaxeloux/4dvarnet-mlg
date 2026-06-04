import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm

LAND_COLOR = '#8B4513'


def get_batch_field(batch, field_name):
    """Get field from batch (supports both dict and NamedTuple)."""
    if isinstance(batch, dict):
        return batch[field_name]
    else:
        return getattr(batch, field_name)


def _denormalize_zscore(arr, norm_stats):
    if arr is None:
        return None
    if not norm_stats:
        return arr
    mean = norm_stats.get('mean')
    std = norm_stats.get('std')
    if mean is None or std is None:
        return arr
    return arr * std + mean


def _robust_limits(arrays, default=(-2, 2), percentiles=(2, 98), min_span=1.0):
    values = []
    for arr in arrays:
        if arr is None:
            continue
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            values.append(valid)
    if not values:
        return default

    values = np.concatenate(values)
    vmin, vmax = np.percentile(values, percentiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < min_span:
        center = np.nanmean(values)
        half_span = min_span / 2
        return center - half_span, center + half_span
    return vmin, vmax


def _format_train_context(train_res=None, lr=None):
    parts = []
    if train_res is not None:
        parts.append(f"training x{train_res}")
    if lr is not None:
        parts.append(f"lr={lr:.2e}")
    return " | ".join(parts)


def _absolute_error_cmap():
    cmap = plt.get_cmap('Reds').copy()
    cmap.set_bad(color='white')
    return cmap


def _surfmask_cmap_norm():
    cmap = ListedColormap([LAND_COLOR, '#1E90FF', '#87CEEB', '#FFFFFF'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    return cmap, BoundaryNorm(bounds, cmap.N), bounds


def _overlay_land(ax, surfmask, origin='upper'):
    if surfmask is None:
        return
    land_mask = np.asarray(surfmask) == 0
    if not np.any(land_mask):
        return
    land_overlay = np.ma.masked_where(~land_mask, np.ones_like(land_mask, dtype=float))
    ax.imshow(
        land_overlay,
        cmap=ListedColormap([LAND_COLOR]),
        vmin=0,
        vmax=1,
        interpolation='nearest',
        origin=origin,
        alpha=1.0,
    )


def _extract_surfmask(xr_data, t_idx=None, lat_slice=None, lon_slice=None):
    if 'surfmask' not in xr_data:
        return None

    da = xr_data['surfmask']
    indexers = {}
    if 'time' in da.dims and t_idx is not None:
        indexers['time'] = t_idx
    if lat_slice is not None:
        indexers['lat'] = lat_slice
    if lon_slice is not None:
        indexers['lon'] = lon_slice

    surfmask = da.isel(**indexers).values
    surfmask = np.squeeze(surfmask)
    if surfmask.ndim != 2:
        return None
    return np.flipud(surfmask)


def plot_test_reconstruction(xr_data, save_dir):
    """
    Visualise la reconstruction complète du test (carte entière).
    
    Args:
        xr_data: xarray.Dataset avec les variables pred_sst, tgt_sst, etc.
        save_dir: Path où sauvegarder les figures
    """
    import xarray as xr
    
    # Prendre le timestep central (jour cible)
    t_mid = len(xr_data.time) // 2
    
    # Extraire les variables et FLIP vertical (origine géographique: nord en haut)
    pred_sst = np.flipud(xr_data['pred_sst'].isel(time=t_mid).values)  # (lat, lon)
    tgt_sst = np.flipud(xr_data['tgt_sst'].isel(time=t_mid).values)  # (lat, lon)
    surfmask = _extract_surfmask(xr_data, t_idx=t_mid)
    
    # Calculer l'erreur
    error = np.abs(pred_sst - tgt_sst)
    
    # Créer la figure avec 4 panneaux
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    # Limites de couleur (test datasets are denormalized before plotting).
    vmin_sst, vmax_sst = _robust_limits([pred_sst, tgt_sst], default=(-2, 30))
    _, vmax_err = _robust_limits([error], default=(0, 5), percentiles=(0, 98), min_span=1.0)
    vmin_err = 0
    err_cmap = _absolute_error_cmap()
    
    # Plot 1: Target (SANS colorbar)
    axes[0].imshow(tgt_sst, cmap='RdYlBu_r', interpolation='nearest', 
                   vmin=vmin_sst, vmax=vmax_sst, origin='upper')
    _overlay_land(axes[0], surfmask)
    axes[0].set_title('Target SST (SLSTR + AASTI fusionné)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude index', fontsize=12)
    axes[0].set_ylabel('Latitude index', fontsize=12)
    
    # Plot 2: Prediction (SANS colorbar)
    axes[1].imshow(pred_sst, cmap='RdYlBu_r', interpolation='nearest',
                   vmin=vmin_sst, vmax=vmax_sst, origin='upper')
    _overlay_land(axes[1], surfmask)
    axes[1].set_title('Predicted SST (4DVar reconstruction)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude index', fontsize=12)
    axes[1].set_ylabel('Latitude index', fontsize=12)
    
    # Plot 3: Input observations (SANS colorbar)
    nan_mask = np.isnan(tgt_sst).astype(float)
    axes[2].imshow(nan_mask, cmap='gray', interpolation='nearest', origin='upper')
    _overlay_land(axes[2], surfmask)
    axes[2].set_title('Data gaps (white=missing, black=valid)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude index', fontsize=12)
    axes[2].set_ylabel('Latitude index', fontsize=12)
    
    # Plot 4: Erreur absolue (SANS colorbar)
    axes[3].imshow(error, cmap=err_cmap, interpolation='nearest',
                   vmin=vmin_err, vmax=vmax_err, origin='upper')
    _overlay_land(axes[3], surfmask)
    axes[3].set_title('Absolute Error |Pred - Target|', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Longitude index', fontsize=12)
    axes[3].set_ylabel('Latitude index', fontsize=12)
    
    # Statistiques
    valid_mask = np.isfinite(pred_sst) & np.isfinite(tgt_sst)
    if valid_mask.sum() > 0:
        rmse = np.sqrt(np.nanmean(error[valid_mask]**2))
        mae = np.nanmean(error[valid_mask])
        bias = np.nanmean((pred_sst - tgt_sst)[valid_mask])
        
        stats_text = f'RMSE: {rmse:.4f} | MAE: {mae:.4f} | Bias: {bias:.4f}'
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Sauvegarder avec timestamp
    timestamp = xr_data.time.values[t_mid]
    date_str = str(timestamp)[:10].replace('-', '')
    filename = f'test_reconstruction_{date_str}.jpg'
    plt.savefig(save_dir / filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")



def save_validation_patches(
        batches_list,
        preds_list,
        save_dir,
        epoch,
        sst_norm_stats=None,
        pmw_norm_stats=None,
        train_res=None,
        lr=None):
    """
    Sauvegarde une grille de tous les patches de validation (16 patches).
    Pour chaque patch: affiche Input (PMW) et Prediction côte à côte.
    
    Args:
        batches_list: Liste de batches de validation (4 batches de 4 patches chacun)
        preds_list: Liste des prédictions correspondantes
        save_dir: Répertoire de sauvegarde
        epoch: Numéro d'epoch
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Collecter tous les patches
    all_targets = []
    all_preds = []
    all_pmws = []
    all_surfmasks = []

    for batch, pred in zip(batches_list, preds_list):
        # Si batch est un dict multi-résolution, extraire le patch haute résolution
        if isinstance(batch, dict) and 'patch_x1' in batch:
            batch = batch['patch_x1']

        # Utiliser tgt_sst_full (complet) si disponible, sinon fallback sur tgt_sst (masqué)
        try:
            batch_tgt = get_batch_field(batch, 'tgt_sst_full')  # Version complète pour visualisation
        except (KeyError, AttributeError):
            batch_tgt = get_batch_field(batch, 'tgt_sst')  # Fallback sur version masquée
        # Shape: (B, T, H, W)
        batch_size = batch_tgt.shape[0]

        t_mid_target = batch_tgt.shape[1] // 2
        t_mid_pred = pred.shape[1] // 2

        for i in range(batch_size):
            target = batch_tgt[i, t_mid_target, :, :].cpu().numpy()
            prediction = pred[i, t_mid_pred, :, :].cpu().numpy() if pred.ndim == 4 else pred[i, t_mid_pred, 0, :, :].cpu().numpy()

            # PMW input
            try:
                pmw_av = get_batch_field(batch, 'pmw_av')
                pmw = pmw_av[i, t_mid_target, :, :].cpu().numpy()
            except (KeyError, AttributeError):
                pmw = np.full_like(target, np.nan)

            # Surfmask
            try:
                surfmask_tensor = get_batch_field(batch, 'surfmask')
                surfmask_2d = surfmask_tensor[i].cpu().numpy()
                if surfmask_2d.ndim == 3:  # (T, H, W)
                    surfmask_2d = surfmask_2d[0]
            except (KeyError, AttributeError):
                surfmask_2d = None

            all_targets.append(_denormalize_zscore(target, sst_norm_stats))
            all_preds.append(_denormalize_zscore(prediction, sst_norm_stats))
            all_pmws.append(_denormalize_zscore(pmw, pmw_norm_stats))
            all_surfmasks.append(surfmask_2d)
    
    n_patches = len(all_targets)

    # Grille: n_patches lignes × 5 colonnes (Input, Pred, Target, Error, Surfmask)
    fig, axes = plt.subplots(n_patches, 5, figsize=(20, 3 * n_patches))
    if n_patches == 1:
        axes = axes.reshape(1, -1)

    # Palette surfmask (réutilisée de plot_patch_analysis)
    cmap_discrete = ListedColormap(['#8B4513', '#1E90FF', '#87CEEB', '#FFFFFF'])
    sm_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    sm_norm = BoundaryNorm(sm_bounds, cmap_discrete.N)
    vmin_sst, vmax_sst = _robust_limits(all_pmws + all_preds + all_targets, default=(-2, 30))
    all_errors = [pred - target for pred, target in zip(all_preds, all_targets)]
    _, vmax_err = _robust_limits([np.abs(err) for err in all_errors], default=(0, 5), percentiles=(0, 98), min_span=1.0)
    vmax_err = max(abs(vmax_err), 1.0)

    for idx in range(n_patches):
        target = all_targets[idx]
        pred = all_preds[idx]
        pmw = all_pmws[idx]
        surfmask = all_surfmasks[idx]
        error = pred - target

        # Colonne 0: Input PMW
        im0 = axes[idx, 0].imshow(pmw, cmap='RdBu_r', vmin=vmin_sst, vmax=vmax_sst, interpolation='nearest')
        axes[idx, 0].set_title(f'Patch {idx+1} - Input PMW')
        axes[idx, 0].axis('off')
        plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04, label='SST (degC)')

        # Colonne 1: Prediction avec land en marron
        # Afficher la prédiction normalement
        im1 = axes[idx, 1].imshow(pred, cmap='RdBu_r', vmin=vmin_sst, vmax=vmax_sst, interpolation='nearest')
        # Overlay marron sur les zones de terre
        if surfmask is not None:
            land_mask = (surfmask == 0)
            land_overlay = np.ma.masked_where(~land_mask, np.ones_like(surfmask))
            axes[idx, 1].imshow(land_overlay, cmap=ListedColormap(['#8B4513']),
                               vmin=0, vmax=1, interpolation='nearest', alpha=1.0)
        axes[idx, 1].set_title(f'Prediction')
        axes[idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04, label='SST (degC)')

        # Colonne 2: Target
        im2 = axes[idx, 2].imshow(target, cmap='RdBu_r', vmin=vmin_sst, vmax=vmax_sst, interpolation='nearest')
        axes[idx, 2].set_title(f'Target')
        axes[idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04, label='SST (degC)')

        # Colonne 3: Error
        im3 = axes[idx, 3].imshow(error, cmap='seismic', vmin=-vmax_err, vmax=vmax_err, interpolation='nearest')
        axes[idx, 3].set_title(f'Error (Pred - Target)')
        axes[idx, 3].axis('off')
        plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046, pad=0.04, label='Error (degC)')

        # Colonne 4: Surfmask
        if surfmask is not None:
            im4 = axes[idx, 4].imshow(surfmask, cmap=cmap_discrete, norm=sm_norm,
                                      interpolation='nearest')
            axes[idx, 4].set_title(f'Surfmask')
            axes[idx, 4].axis('off')
            cbar4 = plt.colorbar(im4, ax=axes[idx, 4], fraction=0.046, pad=0.04,
                                 ticks=[0, 1, 2, 3], boundaries=sm_bounds, spacing='uniform')
            cbar4.set_ticklabels(['Land', 'Ocean', 'Ice-water', 'Ice'])
        else:
            axes[idx, 4].text(0.5, 0.5, 'No surfmask', ha='center', va='center',
                             transform=axes[idx, 4].transAxes)
            axes[idx, 4].axis('off')
    
    context = _format_train_context(train_res, lr)
    if context:
        plt.suptitle(f'Validation patches - Epoch {epoch} | {context}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
    else:
        plt.tight_layout()
    
    # Sauvegarder
    filename = f'validation_all_patches_epoch_{epoch:03d}.jpg'
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[VIZ VAL] Sauvegardé: {save_dir / filename}")


def save_validation_patches_multires(
        batches_list,
        preds_list,
        save_dir,
        epoch,
        sst_norm_stats=None,
        train_res=None,
        lr=None):
    """
    Sauvegarde une grille montrant la progression multi-résolution x10 → x3 → x1.
    Pour chaque patch: affiche les 3 résolutions côte à côte.

    Args:
        batches_list: Liste de batches de validation (4 batches de 4 patches chacun)
        preds_list: Liste des prédictions (dict avec patch_x10_on_x1, patch_x3_on_x1, patch_x1)
        save_dir: Répertoire de sauvegarde
        epoch: Numéro d'epoch
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collecter les prédictions à chaque résolution (toutes sur grille x1)
    all_x10 = []
    all_x3 = []
    all_x1 = []

    for pred_dict in preds_list:
        # Extraire les prédictions multi-résolution
        # patch_x10_on_x1: x10 interpolé sur grille x1
        # patch_x3_on_x1: x3 interpolé sur grille x1
        # patch_x1: prédiction finale x1

        x10_on_x1 = pred_dict.get('patch_x10_on_x1', {}).get('tgt_sst')
        x3_on_x1 = pred_dict.get('patch_x3_on_x1', {}).get('tgt_sst')
        x1 = pred_dict.get('patch_x1', {}).get('tgt_sst')

        if x1 is None:
            continue

        batch_size = x1.shape[0]
        t_mid = x1.shape[1] // 2

        for i in range(batch_size):
            # Extraire le timestep central pour chaque résolution
            pred_x1 = x1[i, t_mid, :, :].cpu().numpy() if x1 is not None else None
            pred_x3 = x3_on_x1[i, t_mid, :, :].cpu().numpy() if x3_on_x1 is not None else None
            pred_x10 = x10_on_x1[i, t_mid, :, :].cpu().numpy() if x10_on_x1 is not None else None

            all_x1.append(_denormalize_zscore(pred_x1, sst_norm_stats))
            all_x3.append(_denormalize_zscore(pred_x3, sst_norm_stats))
            all_x10.append(_denormalize_zscore(pred_x10, sst_norm_stats))

    n_patches = len(all_x1)
    if n_patches == 0:
        print(f"[VIZ MULTIRES] Aucun patch à visualiser")
        return

    # Créer la grille: n_patches lignes × 3 colonnes (x10, x3, x1)
    fig, axes = plt.subplots(n_patches, 3, figsize=(12, 3 * n_patches))
    if n_patches == 1:
        axes = axes.reshape(1, -1)

    # Calculer vmin/vmax global pour comparabilité entre résolutions
    vmin, vmax = _robust_limits(all_x10 + all_x3 + all_x1, default=(-2, 30))

    for idx in range(n_patches):
        pred_x10 = all_x10[idx]
        pred_x3 = all_x3[idx]
        pred_x1 = all_x1[idx]

        # Colonne 0: x10 (50km)
        if pred_x10 is not None:
            im0 = axes[idx, 0].imshow(pred_x10, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[idx, 0].set_title(f'Patch {idx+1} - x10 (50km)')
        else:
            axes[idx, 0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[idx, 0].transAxes)
            axes[idx, 0].set_title(f'Patch {idx+1} - x10 (50km)')
        axes[idx, 0].axis('off')

        # Colonne 1: x3 (15km)
        if pred_x3 is not None:
            im1 = axes[idx, 1].imshow(pred_x3, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[idx, 1].set_title(f'x3 (15km)')
        else:
            axes[idx, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[idx, 1].transAxes)
            axes[idx, 1].set_title(f'x3 (15km)')
        axes[idx, 1].axis('off')

        # Colonne 2: x1 (5km) - Finale
        if pred_x1 is not None:
            im2 = axes[idx, 2].imshow(pred_x1, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='nearest')
            axes[idx, 2].set_title(f'x1 (5km) - Final')
        else:
            axes[idx, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[idx, 2].transAxes)
            axes[idx, 2].set_title(f'x1 (5km) - Final')
        axes[idx, 2].axis('off')

    # Titre et layout
    context = _format_train_context(train_res, lr)
    title = f'Multi-Resolution Progression - Epoch {epoch}'
    if context:
        title = f'{title} | {context}'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])  # Laisser place pour colorbar et titre

    # Colorbar commune à droite
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    if pred_x1 is not None:
        fig.colorbar(im2, cax=cbar_ax, label='SST (degC)')

    # Sauvegarder
    filename = f'validation_multires_patches_epoch_{epoch:03d}.jpg'
    plt.savefig(save_dir / filename, dpi=150)
    plt.close()

    print(f"[VIZ MULTIRES] Sauvegardé: {save_dir / filename}")


def plot_patch_analysis(patches_data, save_dir, title_suffix="", sst_norm_stats=None):
    """
    Génère 1 PNG par patch avec 3 colonnes: Target, Pred, Surfmask.
    Organisé par résolution dans patches/res_x*/.
    
    Args:
        patches_data: Liste de dicts contenant {'target', 'prediction', 'surfmask', 'res', 'id'}
        save_dir: Répertoire de sauvegarde (ex: outputs/test_results/.../analysis/)
        title_suffix: Suffixe pour identifier la résolution (ex: "res_x10")
    """
    n_patches = len(patches_data)
    if n_patches == 0:
        return

    # Extraire résolution depuis title_suffix (ex: "res_x10" -> "x10")
    res_name = title_suffix.replace("res_", "") if "res_" in title_suffix else "unknown"
    
    # Créer sous-dossier pour cette résolution
    patches_dir = save_dir / "patches" / f"res_{res_name}"
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    # Track patches with no valid data
    nan_only_count = 0
    
    surfmask_cmap, surfmask_norm, surfmask_bounds = _surfmask_cmap_norm()

    # Générer 1 PNG par patch (1 ligne × 4 colonnes)
    for i, data in enumerate(patches_data):
        target = data['target']
        pred = data['prediction']
        target = _denormalize_zscore(target, sst_norm_stats)
        pred = _denormalize_zscore(pred, sst_norm_stats)
        patch_id = data.get('id', i)
        surfmask = data.get('surfmask')
        
        # Calculer RMSE
        error = pred - target
        valid_errors = error[~np.isnan(error)]
        if len(valid_errors) > 0:
            rmse = np.sqrt(np.mean(valid_errors**2))
            rmse_str = f'RMSE={rmse:.3f}'
        else:
            rmse_str = 'N/A'
            nan_only_count += 1
        
        # Créer figure 1 ligne × 4 colonnes
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        vmin_sst, vmax_sst = _robust_limits([target, pred], default=(-2, 30))
        _, vmax_err = _robust_limits([np.abs(error)], default=(0, 5), percentiles=(0, 98), min_span=1.0)
        vmax_err = max(abs(vmax_err), 1.0)

        # Colonne 1: Target
        im0 = axes[0].imshow(target, cmap='RdYlBu_r', interpolation='nearest', 
                            vmin=vmin_sst, vmax=vmax_sst, origin='upper')
        _overlay_land(axes[0], surfmask)
        axes[0].set_title(f'Target SST', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='SST (degC)')
        
        # Colonne 2: Prediction
        im1 = axes[1].imshow(pred, cmap='RdYlBu_r', interpolation='nearest', 
                            vmin=vmin_sst, vmax=vmax_sst, origin='upper')
        _overlay_land(axes[1], surfmask)
        axes[1].set_title(f'Prediction ({rmse_str})', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='SST (degC)')

        # Colonne 3: Error
        im2 = axes[2].imshow(error, cmap='seismic', interpolation='nearest',
                             vmin=-vmax_err, vmax=vmax_err, origin='upper')
        _overlay_land(axes[2], surfmask)
        axes[2].set_title('Error (Pred - Target)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Error (degC)')
        
        # Colonne 4: Surfmask
        if surfmask is not None:
            im3 = axes[3].imshow(surfmask, cmap=surfmask_cmap, norm=surfmask_norm,
                                interpolation='nearest', origin='upper')
            axes[3].set_title('Surfmask', fontsize=12, fontweight='bold')
            axes[3].axis('off')
            cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04,
                                 ticks=[0, 1, 2, 3], boundaries=surfmask_bounds, spacing='uniform')
            cbar3.set_ticklabels(['Land', 'Ocean', 'Ice-water', 'Ice'])
        else:
            axes[3].text(0.5, 0.5, 'No surfmask', ha='center', va='center',
                         transform=axes[3].transAxes, fontsize=10)
            axes[3].axis('off')
        
        # Titre global
        fig.suptitle(f'Patch {patch_id:03d} - Resolution {res_name}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f'patch_{patch_id:03d}.png'
        plt.savefig(patches_dir / filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Report
    if nan_only_count > 0:
        print(f"    {nan_only_count}/{n_patches} patches all NaN")
    
    print(f"    Saved {n_patches} patches in: {patches_dir.relative_to(save_dir.parent.parent)}")


def plot_global_netcdf_as_png(xr_data, res, save_dir):
    """
    Génère un PNG de la reconstruction globale depuis le NetCDF agrégé.
    2 colonnes: Target | Prediction (avec stats RMSE/MAE).
    
    Args:
        xr_data: xarray.Dataset avec pred_sst, tgt_sst
        res: Résolution (1, 3, ou 10)
        save_dir: Path de sauvegarde
    """
    # Timestep central
    t_mid = len(xr_data.time) // 2
    
    # Extraire et FLIP
    pred_sst = np.flipud(xr_data['pred_sst'].isel(time=t_mid).values)
    tgt_sst = np.flipud(xr_data['tgt_sst'].isel(time=t_mid).values)
    surfmask = _extract_surfmask(xr_data, t_idx=t_mid)
    
    # Stats
    valid_mask = np.isfinite(pred_sst) & np.isfinite(tgt_sst)
    if valid_mask.sum() > 0:
        error = pred_sst - tgt_sst
        rmse = np.sqrt(np.nanmean(error[valid_mask]**2))
        mae = np.nanmean(np.abs(error[valid_mask]))
        bias = np.nanmean(error[valid_mask])
        stats_text = f'RMSE: {rmse:.4f} | MAE: {mae:.4f} | Bias: {bias:+.4f}'
    else:
        stats_text = 'No valid data'
    
    # Créer figure 1 ligne × 2 colonnes
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    vmin, vmax = _robust_limits([pred_sst, tgt_sst], default=(-2, 30))
    
    # Colonne 1: Target
    im0 = axes[0].imshow(tgt_sst, cmap='RdYlBu_r', interpolation='nearest',
                        vmin=vmin, vmax=vmax, origin='upper')
    _overlay_land(axes[0], surfmask)
    axes[0].set_title(f'Target SST - x{res} (5km × {res})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude index', fontsize=11)
    axes[0].set_ylabel('Latitude index', fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='SST (degC)')
    
    # Colonne 2: Prediction
    im1 = axes[1].imshow(pred_sst, cmap='RdYlBu_r', interpolation='nearest',
                        vmin=vmin, vmax=vmax, origin='upper')
    _overlay_land(axes[1], surfmask)
    axes[1].set_title(f'Predicted SST - x{res}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude index', fontsize=11)
    axes[1].set_ylabel('Latitude index', fontsize=11)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='SST (degC)')
    
    # Stats en bas
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Sauvegarder
    timestamp = xr_data.time.values[t_mid]
    date_str = str(timestamp)[:10].replace('-', '')
    filename = f'global_reconstruction_x{res}_{date_str}.png'
    # CRITICAL: Reduce DPI for x1 (3600×7200 grid = huge!)
    dpi = 200 if res == 1 else 200
    plt.savefig(save_dir / filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {filename}")


def plot_test_reconstruction_split(xr_data, save_dir):
    """
    Génère 4 PNG séparés (1 subplot par fichier) SANS rectangles.
    Version individuelle de plot_test_reconstruction.
    
    Args:
        xr_data: xarray.Dataset avec pred_sst, tgt_sst
        save_dir: Path de sauvegarde
    """
    import xarray as xr
    
    t_mid = len(xr_data.time) // 2
    
    # Extraire et FLIP
    pred_sst = np.flipud(xr_data['pred_sst'].isel(time=t_mid).values)
    tgt_sst = np.flipud(xr_data['tgt_sst'].isel(time=t_mid).values)
    surfmask = _extract_surfmask(xr_data, t_idx=t_mid)
    error = np.abs(pred_sst - tgt_sst)
    nan_mask = np.isnan(tgt_sst).astype(float)
    
    # Stats
    valid_mask = np.isfinite(pred_sst) & np.isfinite(tgt_sst)
    if valid_mask.sum() > 0:
        rmse = np.sqrt(np.nanmean(error[valid_mask]**2))
        mae = np.nanmean(error[valid_mask])
        bias = np.nanmean((pred_sst - tgt_sst)[valid_mask])
        stats_text = f'RMSE: {rmse:.4f} | MAE: {mae:.4f} | Bias: {bias:+.4f}'
    else:
        stats_text = 'No valid data'
    
    timestamp = xr_data.time.values[t_mid]
    date_str = str(timestamp)[:10].replace('-', '')
    
    vmin_sst, vmax_sst = _robust_limits([pred_sst, tgt_sst], default=(-2, 30))
    vmin_err = 0
    _, vmax_err = _robust_limits([error], default=(0, 5), percentiles=(0, 98), min_span=1.0)
    err_cmap = _absolute_error_cmap()
    
    # PNG 1: Target
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(tgt_sst, cmap='RdYlBu_r', interpolation='nearest',
                   vmin=vmin_sst, vmax=vmax_sst, origin='upper')
    _overlay_land(ax, surfmask)
    ax.set_title('Target SST (SLSTR + AASTI fusionné)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude index', fontsize=12)
    ax.set_ylabel('Latitude index', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SST (degC)')
    plt.tight_layout()
    # CRITICAL: Low DPI for x1 (3600×7200 grid)
    dpi = 40 if tgt_sst.shape[0] > 2000 else 200
    plt.savefig(save_dir / f'test_target_sst_{date_str}.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # PNG 2: Prediction
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pred_sst, cmap='RdYlBu_r', interpolation='nearest',
                   vmin=vmin_sst, vmax=vmax_sst, origin='upper')
    _overlay_land(ax, surfmask)
    ax.set_title(f'Predicted SST - {stats_text}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude index', fontsize=12)
    ax.set_ylabel('Latitude index', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SST (degC)')
    plt.tight_layout()
    plt.savefig(save_dir / f'test_prediction_sst_{date_str}.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # PNG 3: Data gaps
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(nan_mask, cmap='gray', interpolation='nearest', origin='upper')
    _overlay_land(ax, surfmask)
    ax.set_title('Data gaps (white=missing, black=valid)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude index', fontsize=12)
    ax.set_ylabel('Latitude index', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Missing fraction')
    plt.tight_layout()
    plt.savefig(save_dir / f'test_data_gaps_{date_str}.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # PNG 4: Absolute Error
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(error, cmap=err_cmap, interpolation='nearest',
                   vmin=vmin_err, vmax=vmax_err, origin='upper')
    _overlay_land(ax, surfmask)
    ax.set_title('Absolute Error |Pred - Target|', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude index', fontsize=12)
    ax.set_ylabel('Latitude index', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error (degC)')
    plt.tight_layout()
    plt.savefig(save_dir / f'test_absolute_error_{date_str}.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved 4 split PNG files for {date_str}")


def plot_temporal_sequence(xr_data, save_dir, n_patches=7):
    """
    Génère des PNGs avec séquence temporelle pour 3 patches aléatoires 256x256.
    Pour CHAQUE patch : 3 lignes (Target, Pred, Error) x 5 colonnes (timesteps).
    
    Args:
        xr_data: xarray.Dataset avec pred_sst, tgt_sst sur 5 timesteps (grille complète)
        save_dir: Path de sauvegarde
        n_patches: Nombre de patches aléatoires à extraire (défaut: 3)
    """
    import xarray as xr
    
    n_times = len(xr_data.time)
    if n_times != 5:
        print(f"[VIZ WARNING] Expected 5 timesteps for temporal sequence, got {n_times}. Skipping.")
        return
    
    timestamp_central = xr_data.time.values[2]  # Jour 0
    date_str = str(timestamp_central)[:10].replace('-', '')
    
    # Grille complète
    nlat, nlon = xr_data.sizes['lat'], xr_data.sizes['lon']
    patch_size = 256
    
    # Extraire n_patches patches aléatoires (même position pour tous les timesteps)
    np.random.seed(42)  # Reproductibilité
    for patch_id in range(n_patches):
        # Position aléatoire valide
        lat_start = np.random.randint(0, max(1, nlat - patch_size))
        lon_start = np.random.randint(0, max(1, nlon - patch_size))
        lat_end = min(lat_start + patch_size, nlat)
        lon_end = min(lon_start + patch_size, nlon)
        
        # Extraire le patch pour tous les timesteps
        pred_all = []
        tgt_all = []
        surfmask = _extract_surfmask(
            xr_data,
            t_idx=2,
            lat_slice=slice(lat_start, lat_end),
            lon_slice=slice(lon_start, lon_end),
        )
        for t in range(n_times):
            pred = np.flipud(xr_data['pred_sst'].isel(time=t, lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).values)
            tgt = np.flipud(xr_data['tgt_sst'].isel(time=t, lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).values)
            pred_all.append(pred)
            tgt_all.append(tgt)
        
        error_all = [np.abs(p - t) for p, t in zip(pred_all, tgt_all)]
        
        # Créer figure 3 lignes x 5 colonnes
        fig, axes = plt.subplots(3, 5, figsize=(20, 9))
        
        vmin_sst, vmax_sst = _robust_limits(pred_all + tgt_all, default=(-2, 30))
        vmin_err = 0
        _, vmax_err = _robust_limits(error_all, default=(0, 5), percentiles=(0, 98), min_span=1.0)
        err_cmap = _absolute_error_cmap()
        
        # Labels des jours
        day_labels = ['Day -2', 'Day -1', 'Day 0 (target)', 'Day +1', 'Day +2']
        
        # Ligne 1: Target
        for col, (tgt, label) in enumerate(zip(tgt_all, day_labels)):
            axes[0, col].imshow(tgt, cmap='RdYlBu_r', interpolation='nearest',
                                    vmin=vmin_sst, vmax=vmax_sst, origin='upper')
            _overlay_land(axes[0, col], surfmask)
            axes[0, col].set_title(label, fontsize=11, fontweight='bold')
            axes[0, col].axis('off')
        axes[0, 0].set_ylabel('Target', fontsize=13, fontweight='bold', rotation=0, labelpad=40, va='center')
        
        # Ligne 2: Prediction
        for col, pred in enumerate(pred_all):
            axes[1, col].imshow(pred, cmap='RdYlBu_r', interpolation='nearest',
                                    vmin=vmin_sst, vmax=vmax_sst, origin='upper')
            _overlay_land(axes[1, col], surfmask)
            axes[1, col].axis('off')
        axes[1, 0].set_ylabel('Prediction', fontsize=13, fontweight='bold', rotation=0, labelpad=40, va='center')
        
        # Ligne 3: Error
        for col, error in enumerate(error_all):
            axes[2, col].imshow(error, cmap=err_cmap, interpolation='nearest',
                                    vmin=vmin_err, vmax=vmax_err, origin='upper')
            _overlay_land(axes[2, col], surfmask)
            axes[2, col].axis('off')
        axes[2, 0].set_ylabel('Error', fontsize=13, fontweight='bold', rotation=0, labelpad=40, va='center')
        
        # Colorbars communes à droite
        fig.subplots_adjust(right=0.92)
        
        # Colorbar pour SST (lignes 1-2)
        cbar_ax_sst = fig.add_axes([0.93, 0.375, 0.015, 0.55])
        sm_sst = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=vmin_sst, vmax=vmax_sst))
        sm_sst.set_array([])
        fig.colorbar(sm_sst, cax=cbar_ax_sst, label='SST (degC)')
        
        # Colorbar pour Error (ligne 3)
        cbar_ax_err = fig.add_axes([0.93, 0.11, 0.015, 0.25])
        sm_err = plt.cm.ScalarMappable(cmap=err_cmap, norm=plt.Normalize(vmin=vmin_err, vmax=vmax_err))
        sm_err.set_array([])
        fig.colorbar(sm_err, cax=cbar_ax_err, label='|Error| (degC)')
        
        # Titre global
        fig.suptitle(f'5-Day Temporal Sequence - Patch {patch_id+1}/{n_patches} (lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}])',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        
        # Sauvegarder
        filename = f'temporal_sequence_patch{patch_id+1:02d}_{date_str}.png'
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")


def plot_multires_comparison(aggregate_results, save_dir, n_patches=3):
    """
    Compare le MÊME patch (mêmes coordonnées lat/lon) à différentes résolutions.
    1 row x 3 columns: x10 | x3 | x1
    
    Args:
        aggregate_results: dict {patch_x10: {...}, patch_x3: {...}, patch_x1: {...}}
        save_dir: Path de sauvegarde
        n_patches: Nombre de patches à comparer (défaut: 3)
    """
    # Vérifier que toutes les résolutions sont disponibles
    resolutions = [10, 3, 1]
    data_by_res = {}
    for res in resolutions:
        key = f"patch_x{res}"
        if key not in aggregate_results or 'daw_0' not in aggregate_results[key]:
            print(f"[VIZ WARNING] Missing data for {key}, skipping multires comparison")
            return
        data_by_res[res] = aggregate_results[key]['daw_0']
    
    # Prendre le timestep central (jour 0)
    x10_data = data_by_res[10]
    x3_data = data_by_res[3]
    x1_data = data_by_res[1]
    
    t_mid_x10 = len(x10_data.time) // 2
    t_mid_x3 = len(x3_data.time) // 2
    t_mid_x1 = len(x1_data.time) // 2
    
    # Grilles complètes
    nlat_x1, nlon_x1 = x1_data.sizes['lat'], x1_data.sizes['lon']
    
    # Extraire n_patches patches aléatoires sur x1 (plus haute résolution)
    np.random.seed(123)
    patch_size = 256
    
    date_str_x1 = str(x1_data.time.values[t_mid_x1])[:10].replace('-', '')
    
    for patch_id in range(n_patches):
        # Position aléatoire sur x1
        lat_start_x1 = np.random.randint(0, max(1, nlat_x1 - patch_size))
        lon_start_x1 = np.random.randint(0, max(1, nlon_x1 - patch_size))
        lat_end_x1 = min(lat_start_x1 + patch_size, nlat_x1)
        lon_end_x1 = min(lon_start_x1 + patch_size, nlon_x1)
        
        # Extraire patch x1
        pred_x1 = np.flipud(x1_data['pred_sst'].isel(time=t_mid_x1, lat=slice(lat_start_x1, lat_end_x1), lon=slice(lon_start_x1, lon_end_x1)).values)
        tgt_x1 = np.flipud(x1_data['tgt_sst'].isel(time=t_mid_x1, lat=slice(lat_start_x1, lat_end_x1), lon=slice(lon_start_x1, lon_end_x1)).values)
        surfmask_x1 = _extract_surfmask(
            x1_data,
            t_idx=t_mid_x1,
            lat_slice=slice(lat_start_x1, lat_end_x1),
            lon_slice=slice(lon_start_x1, lon_end_x1),
        )
        
        # Correspondance spatiale approximative pour x3 et x10
        # x1 = 5km, x3 = 15km (3×), x10 = 50km (10×)
        lat_start_x3 = lat_start_x1 // 3
        lon_start_x3 = lon_start_x1 // 3
        lat_end_x3 = min(lat_start_x3 + patch_size // 3, x3_data.sizes['lat'])
        lon_end_x3 = min(lon_start_x3 + patch_size // 3, x3_data.sizes['lon'])
        
        lat_start_x10 = lat_start_x1 // 10
        lon_start_x10 = lon_start_x1 // 10
        lat_end_x10 = min(lat_start_x10 + patch_size // 10, x10_data.sizes['lat'])
        lon_end_x10 = min(lon_start_x10 + patch_size // 10, x10_data.sizes['lon'])
        
        # Extraire patchs x3 et x10
        pred_x3 = np.flipud(x3_data['pred_sst'].isel(time=t_mid_x3, lat=slice(lat_start_x3, lat_end_x3), lon=slice(lon_start_x3, lon_end_x3)).values)
        tgt_x3 = np.flipud(x3_data['tgt_sst'].isel(time=t_mid_x3, lat=slice(lat_start_x3, lat_end_x3), lon=slice(lon_start_x3, lon_end_x3)).values)
        surfmask_x3 = _extract_surfmask(
            x3_data,
            t_idx=t_mid_x3,
            lat_slice=slice(lat_start_x3, lat_end_x3),
            lon_slice=slice(lon_start_x3, lon_end_x3),
        )
        
        pred_x10 = np.flipud(x10_data['pred_sst'].isel(time=t_mid_x10, lat=slice(lat_start_x10, lat_end_x10), lon=slice(lon_start_x10, lon_end_x10)).values)
        tgt_x10 = np.flipud(x10_data['tgt_sst'].isel(time=t_mid_x10, lat=slice(lat_start_x10, lat_end_x10), lon=slice(lon_start_x10, lon_end_x10)).values)
        surfmask_x10 = _extract_surfmask(
            x10_data,
            t_idx=t_mid_x10,
            lat_slice=slice(lat_start_x10, lat_end_x10),
            lon_slice=slice(lon_start_x10, lon_end_x10),
        )
        
        # Créer figure 3 rows × 3 columns (Target + Pred + Surfmask)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        vmin, vmax = _robust_limits(
            [pred_x10, tgt_x10, pred_x3, tgt_x3, pred_x1, tgt_x1],
            default=(-2, 30)
        )
        surfmask_cmap, surfmask_norm, surfmask_bounds = _surfmask_cmap_norm()
        
        # Row 1: Targets
        axes[0, 0].imshow(tgt_x10, cmap='RdYlBu_r', interpolation='nearest', vmin=vmin, vmax=vmax, origin='upper')
        _overlay_land(axes[0, 0], surfmask_x10)
        axes[0, 0].set_title(f'Target x10 (50km)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(tgt_x3, cmap='RdYlBu_r', interpolation='nearest', vmin=vmin, vmax=vmax, origin='upper')
        _overlay_land(axes[0, 1], surfmask_x3)
        axes[0, 1].set_title(f'Target x3 (15km)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(tgt_x1, cmap='RdYlBu_r', interpolation='nearest', vmin=vmin, vmax=vmax, origin='upper')
        _overlay_land(axes[0, 2], surfmask_x1)
        axes[0, 2].set_title(f'Target x1 (5km)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Predictions
        axes[1, 0].imshow(pred_x10, cmap='RdYlBu_r', interpolation='nearest', vmin=vmin, vmax=vmax, origin='upper')
        _overlay_land(axes[1, 0], surfmask_x10)
        axes[1, 0].set_title(f'Prediction x10', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_x3, cmap='RdYlBu_r', interpolation='nearest', vmin=vmin, vmax=vmax, origin='upper')
        _overlay_land(axes[1, 1], surfmask_x3)
        axes[1, 1].set_title(f'Prediction x3', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(pred_x1, cmap='RdYlBu_r', interpolation='nearest', vmin=vmin, vmax=vmax, origin='upper')
        _overlay_land(axes[1, 2], surfmask_x1)
        axes[1, 2].set_title(f'Prediction x1', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        # Row 3: Surfmask
        for ax, surfmask, title in [
            (axes[2, 0], surfmask_x10, 'Surfmask x10'),
            (axes[2, 1], surfmask_x3, 'Surfmask x3'),
            (axes[2, 2], surfmask_x1, 'Surfmask x1'),
        ]:
            if surfmask is not None:
                ax.imshow(surfmask, cmap=surfmask_cmap, norm=surfmask_norm,
                          interpolation='nearest', origin='upper')
            else:
                ax.text(0.5, 0.5, 'No surfmask', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # Colorbars communes
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.38, 0.02, 0.50])
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='SST (degC)')

        cbar_ax_sm = fig.add_axes([0.92, 0.10, 0.02, 0.18])
        sm_surfmask = plt.cm.ScalarMappable(cmap=surfmask_cmap, norm=surfmask_norm)
        sm_surfmask.set_array([])
        cbar_sm = fig.colorbar(
            sm_surfmask,
            cax=cbar_ax_sm,
            ticks=[0, 1, 2, 3],
            boundaries=surfmask_bounds,
            spacing='uniform',
        )
        cbar_sm.set_ticklabels(['Land', 'Ocean', 'Ice-water', 'Ice'])
        
        # Titre global
        fig.suptitle(f'Multi-Resolution Comparison - Patch {patch_id+1}/{n_patches} (x1 lat[{lat_start_x1}:{lat_end_x1}])',
                    fontsize=14, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0, 0.90, 0.93])
        
        # Sauvegarder
        filename = f'multires_comparison_patch{patch_id+1:02d}_{date_str_x1}.png'
        plt.savefig(save_dir / filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filename}")



def plot_spectral_analysis(patches_data, save_dir, title_suffix=""):
    """
    Calcule et affiche la densité spectrale de puissance (PSD) moyenne.
    """
    psd_targets = []
    psd_preds = []
    
    def compute_radial_psd(img):
        # Fill NaNs with mean
        if np.isnan(img).any():
            mask = np.isnan(img)
            img = img.copy()
            img[mask] = np.nanmean(img)
        
        # Remove mean to avoid DC component dominance
        img = img - np.mean(img)
        
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)**2
        
        # Radial profile
        y, x = np.indices(img.shape)
        center = np.array(img.shape) / 2
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile
    
    for data in patches_data:
        target = data['target']
        pred = data['prediction']
        
        # Skip if too many NaNs
        if np.isnan(target).sum() > target.size * 0.5:
            continue
            
        psd_targets.append(compute_radial_psd(target))
        psd_preds.append(compute_radial_psd(pred))
    
    if not psd_targets:
        print("[VIZ] No valid patches for spectral analysis")
        return

    # Mean PSD
    mean_psd_target = np.mean(np.array(psd_targets), axis=0)
    mean_psd_pred = np.mean(np.array(psd_preds), axis=0)
    
    # Log scale for plotting
    mean_psd_target_log = 10 * np.log10(mean_psd_target + 1e-10)
    mean_psd_pred_log = 10 * np.log10(mean_psd_pred + 1e-10)
    
    freqs = np.arange(len(mean_psd_target))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, mean_psd_target_log, label='Target', linewidth=2)
    ax.plot(freqs, mean_psd_pred_log, label='Prediction', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Spatial Frequency (pixels^-1)')
    ax.set_ylabel('Power Spectrum (dB)')
    ax.set_title(f'Average Power Spectral Density - {title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    filename = f'spectral_analysis_{title_suffix}.jpg'
    plt.savefig(save_dir / filename, dpi=150)
    plt.close()
    print(f"    Saved: {filename}")
