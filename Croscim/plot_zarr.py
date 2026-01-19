#!/usr/bin/env python3
"""Plot toutes les variables d'un fichier Zarr et sauvegarde les figures"""
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python plot_zarr.py <path_to_zarr>")
    sys.exit(1)

zarr_path = sys.argv[1]
print(f"\nOuverture: {zarr_path}")

# Créer dossier de sortie
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("figs") / "Zarr" / date_str
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Sortie: {output_dir}")

# Ouvrir zarr
ds = xr.open_zarr(zarr_path)
print(f"\nVariables trouvées: {len(ds.data_vars)}")

# Plot chaque variable
for i, var_name in enumerate(ds.data_vars, 1):
    print(f"  [{i}/{len(ds.data_vars)}] Plotting {var_name}...", end=" ")
    
    data = ds[var_name].values
    
    # Vérifier dimension
    if data.ndim == 2:
        # Image 2D simple
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot avec gestion NaN
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
        ax.set_title(f"{var_name}\nShape: {data.shape}", fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Stats
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f"Min: {valid_data.min():.2f}, Max: {valid_data.max():.2f}, Mean: {valid_data.mean():.2f}")
        
        plt.tight_layout()
        fig.savefig(output_dir / f"{var_name}.png", dpi=600, bbox_inches='tight')
        plt.close(fig)
        print("OK")
        
    elif data.ndim == 3:
        # Série temporelle ou multi-couches
        n_frames = data.shape[0]
        print(f"({n_frames} frames)...", end=" ")
        
        # Plot première et dernière frame
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        im1 = axes[0].imshow(data[0], cmap='RdBu_r', aspect='auto')
        axes[0].set_title(f"{var_name} - Frame 0")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(data[-1], cmap='RdBu_r', aspect='auto')
        axes[1].set_title(f"{var_name} - Frame {n_frames-1}")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        fig.savefig(output_dir / f"{var_name}_frames.png", dpi=600, bbox_inches='tight')
        plt.close(fig)
        print("OK")
        
    else:
        print(f"SKIP (ndim={data.ndim})")

# Fermer dataset
ds.close()

print(f"\nTerminé! {len(list(output_dir.glob('*.png')))} images créées dans {output_dir}")
