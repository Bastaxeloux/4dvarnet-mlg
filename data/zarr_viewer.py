#!/usr/bin/env python3
"""
Script de visualisation pour fichiers Zarr
Usage: python3 zarr_viewer.py mon_fichier.zarr
"""

import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def view_zarr(zarr_path, day_index=0, variable=None):
    """
    Visualise un fichier Zarr de manière interactive
    """
    print(f"Ouverture de {zarr_path}...")
    
    try:
        ds = xr.open_zarr(zarr_path)
        print(f"Dataset chargé !")
        print(f"Dimensions : {dict(ds.dims)}")
        print(f"Variables : {list(ds.data_vars.keys())}")
        print(f"Période : jour {ds.time.min().values} à {ds.time.max().values}")
        
        # Choix automatique de la variable si pas spécifiée
        if variable is None:
            variables = list(ds.data_vars.keys())
            variable = variables[0]  # Prendre la première
            print(f"Variable sélectionnée : {variable}")
        
        # Sélection du jour
        if day_index >= len(ds.time):
            day_index = 0
            print(f"Index jour ajusté à : {day_index}")
        
        print(f"Affichage : {variable}, jour {day_index}")
        
        # Extraction des données pour ce jour
        data = ds[variable].isel(time=day_index)
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot avec coordonnées
        im = data.plot(ax=ax, cmap='viridis', add_colorbar=True)
        
        # Titre informatif
        time_val = ds.time.isel(time=day_index).values
        ax.set_title(f"{variable} - Jour {day_index} (jour {time_val} depuis 2024-01-01)")
        
        # Amélioration de l'affichage
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Statistiques
        valid_data = data.where(~np.isnan(data))
        stats = f"Min: {valid_data.min().values:.2f}, Max: {valid_data.max().values:.2f}, Mean: {valid_data.mean().values:.2f}"
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Option pour sauvegarder
        output_name = f"{Path(zarr_path).stem}_{variable}_day{day_index}.png"
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Image sauvegardée : {output_name}")
        
        # Affichage
        plt.show()
        
        # Menu interactif simple
        print(f"\n=== NAVIGATION ===")
        print(f"Variables disponibles : {list(ds.data_vars.keys())}")
        print(f"Jours disponibles : 0 à {len(ds.time)-1}")
        
        ds.close()
        
    except Exception as e:
        print(f"Erreur : {e}")

def interactive_viewer(zarr_path):
    """
    Visualiseur interactif en mode console
    """
    ds = xr.open_zarr(zarr_path)
    variables = list(ds.data_vars.keys())
    
    print(f"\n=== VISUALISEUR ZARR INTERACTIF ===")
    print(f"Fichier : {zarr_path}")
    print(f"Variables : {variables}")
    print(f"Jours : 0 à {len(ds.time)-1}")
    
    while True:
        print(f"\n--- Menu ---")
        print("1. Afficher une carte")
        print("2. Exporter un jour en NetCDF")
        print("3. Statistiques")
        print("0. Quitter")
        
        choice = input("Choix : ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            var = input(f"Variable ({'/'.join(variables)}) : ").strip()
            if var not in variables:
                var = variables[0]
            
            day = input(f"Jour (0-{len(ds.time)-1}) : ").strip()
            try:
                day = int(day)
            except:
                day = 0
            
            view_zarr(zarr_path, day, var)
            
        elif choice == "2":
            day = input(f"Jour à exporter (0-{len(ds.time)-1}) : ").strip()
            try:
                day = int(day)
                day_data = ds.isel(time=day)
                output_file = f"zarr_day_{day}.nc"
                day_data.to_netcdf(output_file)
                print(f"NetCDF exporté : {output_file}")
                print(f"Visualiser avec : ncview {output_file}")
            except Exception as e:
                print(f"Erreur export : {e}")
                
        elif choice == "3":
            print(f"\n=== STATISTIQUES ===")
            for var in variables:
                data = ds[var]
                valid = data.where(~np.isnan(data))
                print(f"{var}:")
                print(f"  Min: {valid.min().values:.2f}")
                print(f"  Max: {valid.max().values:.2f}")
                print(f"  Mean: {valid.mean().values:.2f}")
                print(f"  Couverture: {(~np.isnan(data)).sum().values / data.size * 100:.1f}%")
    
    ds.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 zarr_viewer.py mon_fichier.zarr [jour] [variable]")
        print("   ou: python3 zarr_viewer.py mon_fichier.zarr --interactive")
        return
    
    zarr_path = sys.argv[1]
    
    if not Path(zarr_path).exists():
        print(f"Erreur : {zarr_path} n'existe pas")se retrouve ici si tu veu
        return
    
    if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
        interactive_viewer(zarr_path)
    else:
        day = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        var = sys.argv[3] if len(sys.argv) > 3 else None
        view_zarr(zarr_path, day, var)

if __name__ == "__main__":
    main()
