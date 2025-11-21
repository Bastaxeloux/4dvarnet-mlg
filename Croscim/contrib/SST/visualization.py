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
    epoch_dir = Path(run_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    sample_idx = batch_idx
    save_inputs_figure(batch, sample_idx, epoch_dir)
    save_target_pred_error(batch, pred, sample_idx, epoch_dir)
    save_error_histogram(batch, pred, epoch_dir)
    save_worst_patches(batch, pred, epoch_dir, top_k=3)
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
    """Visualise target, prediction et erreur absolue."""
    # Si batch est un dict multi-résolution, extraire le patch haute résolution
    if isinstance(batch, dict) and 'patch_x1' in batch:
        batch = batch['patch_x1']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accès direct au champ tgt_sst du TrainingItem
    batch_tgt = get_batch_field(batch, 'tgt_sst')  # Shape: (B, T, H, W)
    
    # Calculer l'index du milieu pour la cible (qui a 15 jours)
    t_mid_target = batch_tgt.shape[1] // 2
    target = batch_tgt[sample_idx, t_mid_target, :, :].cpu().numpy()
    
    # Calculer l'index du milieu pour la prédiction (qui a 5 jours)
    t_mid_pred = pred.shape[1] // 2
    prediction = pred[sample_idx, t_mid_pred, :, :].cpu().numpy() if pred.ndim == 4 else pred[sample_idx, t_mid_pred, 0, :, :].cpu().numpy()
    error = np.abs(target - prediction)
    im0 = axes[0].imshow(target, cmap='RdYlBu_r', interpolation='nearest')
    axes[0].set_title('Target SST')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(prediction, cmap='RdYlBu_r', interpolation='nearest')
    axes[1].set_title('Predicted SST')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(error, cmap='hot', interpolation='nearest')
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_dir / 'target_pred_error.jpg', dpi=150, bbox_inches='tight')
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
