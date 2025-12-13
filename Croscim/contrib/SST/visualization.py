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


def save_training_figures(batch, pred, epoch, run_dir, batch_idx=0):
    """Sauvegarde toutes les visualisations pour le batch de validation."""
    epoch_dir = Path(run_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = batch_idx
    
    # Visualisations existantes
    save_inputs_figure(batch, sample_idx, epoch_dir)
    save_target_pred_error(batch, pred, sample_idx, epoch_dir)
    save_error_histogram(batch, pred, epoch_dir)
    save_worst_patches(batch, pred, epoch_dir, top_k=3)
    
    # NOUVEAU: Sauvegarder les 4 patches du batch
    save_batch_4_patches(batch, pred, epoch_dir, batch_idx=batch_idx)
    
    print(f"[VIZ] Figures saved to {epoch_dir}")


def save_inputs_figure(batch, sample_idx, save_dir):
    """Visualise les inputs slstr_av et aasti_av (moyennes temporelles)."""
    try:
        # Si batch est un dict multi-résolution, extraire le patch haute résolution
        if isinstance(batch, dict) and 'patch_x1' in batch:
            batch = batch['patch_x1']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Accès direct aux champs du TrainingItem
        slstr_av = get_batch_field(batch, 'slstr_av')  # Shape: (B, T, H, W)
        aasti_av = get_batch_field(batch, 'aasti_av')  # Shape: (B, T, H, W)

        t_mid = slstr_av.shape[1] // 2
        slstr = slstr_av[sample_idx, t_mid, :, :].cpu().numpy()
        aasti = aasti_av[sample_idx, t_mid, :, :].cpu().numpy()
        im0 = axes[0].imshow(slstr, cmap='RdYlBu_r', interpolation='nearest')
        axes[0].set_title('SLSTR average (input)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(aasti, cmap='RdYlBu_r', interpolation='nearest')
        axes[1].set_title('AASTI average (input)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(save_dir / 'inputs_slstr_aasti.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    except (KeyError, AttributeError) as e:
        # Skip input visualization if fields not available
        print(f"[VIZ] Skipping input visualization: {e}")
        pass


def save_target_pred_error(batch, pred, sample_idx, save_dir):
    """Visualise target, prediction, PMW et erreur absolue (4 panneaux)."""
    # Si batch est un dict multi-résolution, extraire le patch haute résolution
    if isinstance(batch, dict) and 'patch_x1' in batch:
        batch = batch['patch_x1']

    # DIAGNOSTIC: Vérifier si pred existe et a des valeurs
    if pred is None:
        print(f"[VIZ ERROR] pred is None! Cannot visualize predictions.")
        return
    
    print(f"[VIZ] pred shape: {pred.shape}, dtype: {pred.dtype}")
    n_nan = torch.isnan(pred).sum().item() if isinstance(pred, torch.Tensor) else np.isnan(pred).sum()
    print(f"[VIZ] pred contains {n_nan}/{pred.numel() if isinstance(pred, torch.Tensor) else pred.size} NaN values")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Accès direct au champ tgt_sst du TrainingItem
    batch_tgt = get_batch_field(batch, 'tgt_sst')  # Shape: (B, T, H, W)
    
    # Calculer l'index du milieu pour la cible (qui a 15 jours)
    t_mid_target = batch_tgt.shape[1] // 2
    target = batch_tgt[sample_idx, t_mid_target, :, :].cpu().numpy()
    
    # Calculer l'index du milieu pour la prédiction (qui a 5 jours)
    t_mid_pred = pred.shape[1] // 2
    prediction = pred[sample_idx, t_mid_pred, :, :].cpu().numpy() if pred.ndim == 4 else pred[sample_idx, t_mid_pred, 0, :, :].cpu().numpy()
    
    # PMW (moyenne temporelle)
    try:
        pmw_av = get_batch_field(batch, 'pmw_av')  # Shape: (B, T, H, W)
        pmw = pmw_av[sample_idx, t_mid_target, :, :].cpu().numpy()
    except (KeyError, AttributeError):
        pmw = np.full_like(target, np.nan)
    
    error = np.abs(target - prediction)
    
    # Plot 1: Target (fusion slstr + aasti)
    im0 = axes[0].imshow(target, cmap='RdYlBu_r', interpolation='nearest', vmin=-2, vmax=2)
    axes[0].set_title('Target SST (SLSTR + AASTI fusionné)', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Plot 2: Prediction
    im1 = axes[1].imshow(prediction, cmap='RdYlBu_r', interpolation='nearest', vmin=-2, vmax=2)
    axes[1].set_title('Predicted SST', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Plot 3: PMW
    im2 = axes[2].imshow(pmw, cmap='RdYlBu_r', interpolation='nearest', vmin=-2, vmax=2)
    axes[2].set_title('PMW (input)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Plot 4: Erreur absolue
    im3 = axes[3].imshow(error, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[3].set_title('Absolute Error |Pred - Target|', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'target_pred_pmw_error.jpg', dpi=150, bbox_inches='tight')
    plt.close()


def save_error_histogram(batch, pred, save_dir):
    """Histogramme de la distribution des erreurs pour tout le batch."""
    # Si batch est un dict multi-résolution, extraire le patch haute résolution
    if isinstance(batch, dict) and 'patch_x1' in batch:
        batch = batch['patch_x1']

    batch_tgt = get_batch_field(batch, 'tgt_sst')  # Shape: (B, T, H, W)

    # Aligner les dimensions temporelles avant la soustraction
    target_timesteps = batch_tgt.shape[1]
    pred_timesteps = pred.shape[1]

    if target_timesteps > pred_timesteps:
        crop_total = target_timesteps - pred_timesteps
        start_idx = crop_total // 2
        end_idx = start_idx + pred_timesteps
        batch_tgt_cropped = batch_tgt[:, start_idx:end_idx, :, :]
    else:
        batch_tgt_cropped = batch_tgt

    if pred.ndim == 4:
        errors = (batch_tgt_cropped - pred).cpu().numpy()
    errors_flat = errors.flatten()
    errors_flat = errors_flat[~np.isnan(errors_flat)]

    # Gérer le cas où il n'y a aucune erreur valide à afficher
    if errors_flat.size == 0:
        mean_err, std_err = 0.0, 0.0
        title = 'Error Distribution (No valid data)'
    else:
        mean_err, std_err = errors_flat.mean(), errors_flat.std()
        title = f'Error Distribution (mean={mean_err:.3f}, std={std_err:.3f})'

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors_flat, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Error (°C)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'error_histogram.jpg', dpi=150, bbox_inches='tight')
    plt.close()


def save_worst_patches(batch, pred, save_dir, top_k=3):
    """Visualise les top K pires patches (plus grande erreur MSE moyenne)."""
    # Si batch est un dict multi-résolution, extraire le patch haute résolution
    if isinstance(batch, dict) and 'patch_x1' in batch:
        batch = batch['patch_x1']

    batch_tgt = get_batch_field(batch, 'tgt_sst')  # Shape: (B, T, H, W)

    # Aligner les dimensions temporelles avant la soustraction
    target_timesteps = batch_tgt.shape[1]
    pred_timesteps = pred.shape[1]
    if target_timesteps > pred_timesteps:
        crop_total = target_timesteps - pred_timesteps
        start_idx = crop_total // 2
        end_idx = start_idx + pred_timesteps
        batch_tgt_cropped = batch_tgt[:, start_idx:end_idx, :, :]
    else:
        batch_tgt_cropped = batch_tgt

    if pred.ndim == 4:
        errors_squared = (batch_tgt_cropped - pred) ** 2
    mse_per_patch = errors_squared.mean(dim=(1, 2, 3)).cpu().numpy()
    worst_indices = np.argsort(mse_per_patch)[-top_k:][::-1]

    fig, axes = plt.subplots(top_k, 3, figsize=(15, 5 * top_k))
    if top_k == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(worst_indices):
        # Utiliser les tenseurs alignés pour la visualisation
        t_mid_target = batch_tgt.shape[1] // 2
        target = batch_tgt[idx, t_mid_target, :, :].cpu().numpy()

        t_mid_pred = pred.shape[1] // 2
        prediction = pred[idx, t_mid_pred, :, :].cpu().numpy()

        error = np.abs(target - prediction)

        im0 = axes[i, 0].imshow(target, cmap='RdYlBu_r', interpolation='nearest')
        axes[i, 0].set_title(f'Worst #{i+1} - Target (MSE={mse_per_patch[idx]:.4f})')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        im1 = axes[i, 1].imshow(prediction, cmap='RdYlBu_r', interpolation='nearest')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        im2 = axes[i, 2].imshow(error, cmap='hot', interpolation='nearest')
        axes[i, 2].set_title('Absolute Error')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'worst_patches.jpg', dpi=150, bbox_inches='tight')
    plt.close()


def save_batch_4_patches(batch, pred, save_dir, batch_idx=0):
    """
    Sauvegarde les 4 patches d'un batch avec pred, target, PMW et erreur pour chaque patch.
    Chaque patch a 4 panneaux : Target, Pred, PMW, Error
    """
    # Si batch est un dict multi-résolution, extraire le patch haute résolution
    if isinstance(batch, dict) and 'patch_x1' in batch:
        batch = batch['patch_x1']
    
    batch_tgt = get_batch_field(batch, 'tgt_sst')  # Shape: (B, T, H, W)
    batch_size = min(batch_tgt.shape[0], 4)  # Prendre au max 4 patches
    
    # DIAGNOSTIC: Vérifier si les patches sont vraiment différents
    print(f"\n[VIZ DEBUG] save_batch_4_patches - batch_size={batch_size}, batch_tgt.shape={batch_tgt.shape}")
    t_mid = batch_tgt.shape[1] // 2
    for i in range(batch_size):
        patch_i = batch_tgt[i, t_mid, :, :].cpu().numpy()
        print(f"[VIZ DEBUG] Patch {i}: mean={patch_i[~np.isnan(patch_i)].mean():.4f}, "
              f"std={patch_i[~np.isnan(patch_i)].std():.4f}, "
              f"sum={patch_i[~np.isnan(patch_i)].sum():.2f}")
        # Comparer avec le premier patch
        if i > 0:
            patch_0 = batch_tgt[0, t_mid, :, :].cpu().numpy()
            diff = np.abs(patch_i - patch_0)
            max_diff = np.nanmax(diff)
            print(f"[VIZ DEBUG]   -> Diff vs patch 0: max={max_diff:.6f}")
    
    # Vérifier les coordonnées géographiques si disponibles
    try:
        lon_patch = get_batch_field(batch, 'lon_patch')
        lat_patch = get_batch_field(batch, 'lat_patch')
        print(f"[VIZ DEBUG] Geographic coords available:")
        for i in range(batch_size):
            lon_i = lon_patch[i].cpu().numpy() if lon_patch.ndim > 1 else lon_patch.cpu().numpy()
            lat_i = lat_patch[i].cpu().numpy() if lat_patch.ndim > 1 else lat_patch.cpu().numpy()
            print(f"[VIZ DEBUG] Patch {i}: lon=[{lon_i.min():.2f}, {lon_i.max():.2f}], "
                  f"lat=[{lat_i.min():.2f}, {lat_i.max():.2f}]")
    except Exception as e:
        print(f"[VIZ DEBUG] No geographic coords in batch: {e}")
    
    # Calculer les indices temporels (milieu)
    t_mid_target = batch_tgt.shape[1] // 2
    t_mid_pred = pred.shape[1] // 2
    
    # Créer une figure avec 4 lignes (patches) × 4 colonnes (Target, Pred, PMW, Error)
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Extraire les données pour ce patch
        target = batch_tgt[i, t_mid_target, :, :].cpu().numpy()
        prediction = pred[i, t_mid_pred, :, :].cpu().numpy() if pred.ndim == 4 else pred[i, t_mid_pred, 0, :, :].cpu().numpy()
        
        # PMW
        try:
            pmw_av = get_batch_field(batch, 'pmw_av')
            pmw = pmw_av[i, t_mid_target, :, :].cpu().numpy()
        except (KeyError, AttributeError):
            pmw = np.full_like(target, np.nan)
        
        error = np.abs(target - prediction)
        
        # Plot Target
        im0 = axes[i, 0].imshow(target, cmap='RdYlBu_r', interpolation='nearest', vmin=-2, vmax=2)
        axes[i, 0].set_title(f'Patch {i+1} - Target', fontsize=10)
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Plot Prediction
        im1 = axes[i, 1].imshow(prediction, cmap='RdYlBu_r', interpolation='nearest', vmin=-2, vmax=2)
        axes[i, 1].set_title(f'Patch {i+1} - Prediction', fontsize=10)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Plot PMW
        im2 = axes[i, 2].imshow(pmw, cmap='RdYlBu_r', interpolation='nearest', vmin=-2, vmax=2)
        axes[i, 2].set_title(f'Patch {i+1} - PMW', fontsize=10)
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Plot Error
        im3 = axes[i, 3].imshow(error, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        axes[i, 3].set_title(f'Patch {i+1} - Error', fontsize=10)
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'batch_{batch_idx}_4patches.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved 4 patches visualization to {save_dir / f'batch_{batch_idx}_4patches.jpg'}")


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
                         vmin=vmin_sst, vmax=vmax_sst, origin='lower')
    axes[0].set_title('Target SST (SLSTR + AASTI fusionné)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude index', fontsize=12)
    axes[0].set_ylabel('Latitude index', fontsize=12)
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('SST (normalized)', fontsize=11)
    
    # Plot 2: Prediction
    im1 = axes[1].imshow(pred_sst, cmap='RdYlBu_r', interpolation='nearest',
                         vmin=vmin_sst, vmax=vmax_sst, origin='lower')
    axes[1].set_title('Predicted SST (4DVar reconstruction)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude index', fontsize=12)
    axes[1].set_ylabel('Latitude index', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('SST (normalized)', fontsize=11)
    
    # Plot 3: PMW (si disponible)
    try:
        # Chercher les variables PMW dans le xarray
        pmw_vars = [v for v in xr_data.data_vars if 'pmw' in v.lower()]
        if pmw_vars:
            pmw_sst = xr_data[pmw_vars[0]].isel(time=t_mid).values
        else:
            pmw_sst = np.full_like(pred_sst, np.nan)
    except:
        pmw_sst = np.full_like(pred_sst, np.nan)
    
    im2 = axes[2].imshow(pmw_sst, cmap='RdYlBu_r', interpolation='nearest',
                         vmin=vmin_sst, vmax=vmax_sst, origin='lower')
    axes[2].set_title('PMW observations (input)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude index', fontsize=12)
    axes[2].set_ylabel('Latitude index', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label('SST (normalized)', fontsize=11)
    
    # Plot 4: Erreur absolue
    im3 = axes[3].imshow(error, cmap='hot', interpolation='nearest',
                         vmin=vmin_err, vmax=vmax_err, origin='lower')
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
    plt.savefig(save_dir / f'test_reconstruction_{date_str}.jpg', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[VIZ] Test reconstruction plot saved to {save_dir / f'test_reconstruction_{date_str}.jpg'}")

