import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def get_batch_field(batch, field_name):
    """Get field from batch (supports both dict and NamedTuple)."""
    if isinstance(batch, dict):
        return batch[field_name]
    else:
        return getattr(batch, field_name)


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
    
    # Extraire les variables
    pred_sst = xr_data['pred_sst'].isel(time=t_mid).values  # (lat, lon)
    tgt_sst = xr_data['tgt_sst'].isel(time=t_mid).values  # (lat, lon)
    
    # Calculer l'erreur
    error = np.abs(pred_sst - tgt_sst)
    
    # Créer la figure avec 4 panneaux
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    # Limites de couleur
    vmin_sst, vmax_sst = -2, 2
    vmin_err, vmax_err = 0, 1
    
    # Plot 1: Target
    im0 = axes[0].imshow(tgt_sst, cmap='RdYlBu_r', interpolation='nearest', 
                         vmin=vmin_sst, vmax=vmax_sst, origin='upper')
    axes[0].set_title('Target SST (SLSTR + AASTI fusionné)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude index', fontsize=12)
    axes[0].set_ylabel('Latitude index', fontsize=12)
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('SST (normalized)', fontsize=11)
    
    # Plot 2: Prediction
    im1 = axes[1].imshow(pred_sst, cmap='RdYlBu_r', interpolation='nearest',
                         vmin=vmin_sst, vmax=vmax_sst, origin='upper')
    axes[1].set_title('Predicted SST (4DVar reconstruction)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude index', fontsize=12)
    axes[1].set_ylabel('Latitude index', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('SST (normalized)', fontsize=11)
    
    # Plot 3: Input observations (clouds/gaps visible)
    # Note: Input data not available in xr_data (only pred_sst and tgt_sst)
    # Show NaN mask instead to visualize data gaps
    nan_mask = np.isnan(tgt_sst).astype(float)
    im2 = axes[2].imshow(nan_mask, cmap='gray', interpolation='nearest', origin='upper')
    axes[2].set_title('Data gaps (white=missing, black=valid)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude index', fontsize=12)
    axes[2].set_ylabel('Latitude index', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label('Missing data', fontsize=11)
    
    # Plot 4: Erreur absolue
    im3 = axes[3].imshow(error, cmap='hot', interpolation='nearest',
                         vmin=vmin_err, vmax=vmax_err, origin='upper')
    axes[3].set_title('Absolute Error |Pred - Target|', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Longitude index', fontsize=12)
    axes[3].set_ylabel('Latitude index', fontsize=12)
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.set_label('Error (normalized)', fontsize=11)
    
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



def save_validation_patches(batches_list, preds_list, save_dir, epoch):
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
    
    for batch, pred in zip(batches_list, preds_list):
        # Si batch est un dict multi-résolution, extraire le patch haute résolution
        if isinstance(batch, dict) and 'patch_x1' in batch:
            batch = batch['patch_x1']
        
        batch_tgt = get_batch_field(batch, 'tgt_sst')  # Shape: (B, T, H, W)
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
            
            all_targets.append(target)
            all_preds.append(prediction)
            all_pmws.append(pmw)
    
    n_patches = len(all_targets)
    # print(f"[VIZ VAL] Création figure avec {n_patches} patches de validation")
    
    # Créer une grille: 4 colonnes (Input/Target, Pred, Target, Error) × n_patches//4 lignes
    n_rows = (n_patches + 3) // 4  # Arrondir au supérieur
    n_cols = 4  # Input, Pred, Target, Error
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_patches):
        row = idx // 4
        col_offset = (idx % 4) * 1  # Pas de multiplication, on va utiliser les 4 colonnes différemment
        
        # Mais en fait, affichons 4 patches par ligne avec 4 colonnes par patch
        # Reformulons: n_patches lignes, 4 colonnes (Input, Pred, Target, Error)
    
    # Refaisons la grille plus simplement: n_patches lignes × 4 colonnes
    fig, axes = plt.subplots(n_patches, 4, figsize=(16, 3 * n_patches))
    if n_patches == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_patches):
        target = all_targets[idx]
        pred = all_preds[idx]
        pmw = all_pmws[idx]
        error = pred - target
        
        # Colonne 0: Input PMW
        im0 = axes[idx, 0].imshow(pmw, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
        axes[idx, 0].set_title(f'Patch {idx+1} - Input PMW')
        axes[idx, 0].axis('off')
        plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)
        
        # Colonne 1: Prediction
        im1 = axes[idx, 1].imshow(pred, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
        axes[idx, 1].set_title(f'Prediction')
        axes[idx, 1].axis('off')
        plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        # Colonne 2: Target
        im2 = axes[idx, 2].imshow(target, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
        axes[idx, 2].set_title(f'Target')
        axes[idx, 2].axis('off')
        plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)
        
        # Colonne 3: Error
        im3 = axes[idx, 3].imshow(error, cmap='seismic', vmin=-1, vmax=1, interpolation='nearest')
        axes[idx, 3].set_title(f'Error (Pred - Target)')
        axes[idx, 3].axis('off')
        plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = f'validation_all_patches_epoch_{epoch:03d}.jpg'
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[VIZ VAL] Sauvegardé: {save_dir / filename}")


def plot_patch_analysis(patches_data, save_dir, title_suffix=""):
    """
    Visualise une sélection de patches avec Target, Pred, Error et Surfmask.
    
    Args:
        patches_data: Liste de dicts contenant {'target', 'prediction', 'surfmask', 'coords', 'id'}
        save_dir: Répertoire de sauvegarde
        title_suffix: Suffixe pour le titre et le nom de fichier
    """
    n_patches = len(patches_data)
    if n_patches == 0:
        return

    # Limiter à 16 patches max par figure pour rester lisible
    if n_patches > 16:
        print(f"[VIZ] Too many patches ({n_patches}), plotting first 16 only.")
        patches_data = patches_data[:16]
        n_patches = 16

    # 4 colonnes: Target, Pred, Error, Surfmask (PMW supprimé pour économiser mémoire)
    fig, axes = plt.subplots(n_patches, 4, figsize=(16, 3.5 * n_patches))
    if n_patches == 1:
        axes = axes.reshape(1, -1)
    
    # Track patches with no valid data
    nan_only_patches = []
    
    for i, data in enumerate(patches_data):
        target = data['target']
        # Utiliser la sortie DIRECTE du stage en test si disponible (x3/x1), sinon la prédiction finale (x10)
        if 'pred_after_add' in data and data['pred_after_add'] is not None:
            pred = data['pred_after_add']
            pred_label = 'After out+itrp_coarse (Stage 2)'
        else:
            pred = data['prediction']
            pred_label = 'Prediction'
        patch_id = data.get('id', i)
        
        error = pred - target
        # Handle case where all values are NaN (e.g., land-only patches)
        valid_errors = error[~np.isnan(error)]
        if len(valid_errors) > 0:
            rmse = np.sqrt(np.mean(valid_errors**2))
            rmse_str = f'{rmse:.3f}'
        else:
            rmse_str = 'N/A'
            nan_only_patches.append(patch_id)
        
        # Plot Target
        im0 = axes[i, 0].imshow(target, cmap='RdYlBu_r', interpolation='nearest', 
                                vmin=-2, vmax=2, origin='upper')
        axes[i, 0].set_title(f'Patch {patch_id} - Target')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Plot Prediction
        im1 = axes[i, 1].imshow(pred, cmap='RdYlBu_r', interpolation='nearest', 
                                vmin=-2, vmax=2, origin='upper')
        axes[i, 1].set_title(f'{pred_label} (RMSE={rmse_str})')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        im2 = axes[i, 2].imshow(error, cmap='seismic', interpolation='nearest', 
                                vmin=-1, vmax=1, origin='upper')
        axes[i, 2].set_title(f'Error (Pred - Target)')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        if 'surfmask' in data and data['surfmask'] is not None:
            surfmask = data['surfmask']
            im3 = axes[i, 3].imshow(surfmask, cmap='tab10', interpolation='nearest', 
                                    vmin=0, vmax=3, origin='upper')
            axes[i, 3].set_title('Surfmask (0=land, 1=ocean)')
            axes[i, 3].axis('off')
            cbar3 = plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
            cbar3.set_ticklabels(['Land', 'Ocean', 'Ice-water', 'Ice'])
        else:
            axes[i, 3].text(0.5, 0.5, 'No surfmask', ha='center', va='center', transform=axes[i, 3].transAxes)
            axes[i, 3].axis('off')
    
    # Report patches with no valid data
    if len(nan_only_patches) > 0:
        print(f"   {len(nan_only_patches)}/{n_patches} patches all NaN: {nan_only_patches}")
    
    plt.tight_layout()
    filename = f'patch_analysis_{title_suffix}.jpg'
    plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
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



