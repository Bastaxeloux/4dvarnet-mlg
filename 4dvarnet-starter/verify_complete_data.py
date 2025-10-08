#!/usr/bin/env python3

"""
Script pour vérifier que les données sont bien présentes 
dans tous les pas de temps et analyser les valeurs.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def check_all_timesteps():
    """Vérifier les valeurs réelles dans tous les pas de temps."""
    
    # Charger le fichier
    data = xr.open_dataset("test_output/test_logs/test_data.nc")
    out_data = data['out']
    
    print("Analyse détaillée des valeurs:")
    print("=" * 50)
    
    # Analyser quelques pas de temps clés
    key_times = [0, 10, 18, 19, 20, 30, 50, 70, 80]
    
    for t in key_times:
        if t < len(out_data.time):
            slice_data = out_data.isel(time=t).values
            
            print(f"\nPas de temps t={t}:")
            print(f"  Min: {np.min(slice_data):.6f}")
            print(f"  Max: {np.max(slice_data):.6f}")
            print(f"  Moyenne: {np.mean(slice_data):.6f}")
            print(f"  Std: {np.std(slice_data):.6f}")
            print(f"  Valeurs nulles: {np.sum(slice_data == 0)}/{slice_data.size}")
    
    # Créer des visualisations pour vérifier
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    times_to_plot = [0, 10, 18, 19, 20, 30, 50, 80]
    
    for i, t in enumerate(times_to_plot):
        if t < len(out_data.time):
            slice_data = out_data.isel(time=t).values
            
            im = axes[i].imshow(slice_data, cmap='viridis')
            axes[i].set_title(f't={t}\nMin={np.min(slice_data):.3f}, Max={np.max(slice_data):.3f}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('verification_all_times.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualisations sauvées dans: verification_all_times.png")
    
    # Analyser l'évolution temporelle des statistiques
    print(f"\nÉvolution temporelle (échantillon):")
    print("t  |   min    |   max    |  moyenne |   std")
    print("-" * 45)
    
    for t in range(0, len(out_data.time), 10):
        slice_data = out_data.isel(time=t).values
        print(f"{t:2d} | {np.min(slice_data):8.3f} | {np.max(slice_data):8.3f} | {np.mean(slice_data):8.3f} | {np.std(slice_data):6.3f}")

if __name__ == "__main__":
    check_all_timesteps()
