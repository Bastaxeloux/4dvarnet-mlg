#!/usr/bin/env python3

"""
Script pour analyser en détail les données partielles 
dans le fichier NetCDF généré et comprendre où/comment l'arrêt se produit.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def analyze_partial_data():
    """Analyser les données partielles dans le fichier NetCDF."""
    
    # Charger le fichier généré
    try:
        data = xr.open_dataset("test_output/test_logs/test_data.nc")
        print("Fichier NetCDF chargé avec succès")
        print(f"Variables disponibles: {list(data.keys())}")
        print(f"Dimensions: {data.dims}")
        
        # Prendre la variable 'out' (sortie du modèle)
        if 'out' in data:
            out_data = data['out']
            print(f"\nVariable 'out' shape: {out_data.shape}")
            print(f"Coordonnées temps: {len(out_data.time)} pas de temps")
            
            # Analyser la répartition des valeurs non-nulles
            print("\nAnalyse des valeurs non-nulles par pas de temps:")
            for i, t in enumerate(out_data.time.values):
                slice_data = out_data.isel(time=i)
                non_zero = np.sum(slice_data.values != 0)
                total = slice_data.size
                percentage = (non_zero / total) * 100
                print(f"t={i:2d} ({t}): {non_zero:8d}/{total:8d} ({percentage:5.1f}%) valeurs non-nulles")
                
                # Si on trouve des valeurs partielles, analysons plus en détail
                if 0 < non_zero < total:
                    print(f"  *** PARTIEL DETECTE à t={i} ***")
                    
                    # Analyser la distribution spatiale
                    data_2d = slice_data.values
                    non_zero_mask = data_2d != 0
                    
                    # Trouver les limites des données non-nulles
                    if np.any(non_zero_mask):
                        rows_with_data = np.any(non_zero_mask, axis=1)
                        cols_with_data = np.any(non_zero_mask, axis=0)
                        
                        first_row = np.where(rows_with_data)[0][0] if np.any(rows_with_data) else -1
                        last_row = np.where(rows_with_data)[0][-1] if np.any(rows_with_data) else -1
                        first_col = np.where(cols_with_data)[0][0] if np.any(cols_with_data) else -1
                        last_col = np.where(cols_with_data)[0][-1] if np.any(cols_with_data) else -1
                        
                        print(f"    Lignes avec données: {first_row} à {last_row}")
                        print(f"    Colonnes avec données: {first_col} à {last_col}")
                        
                        # Créer une visualisation de la zone partielle
                        plt.figure(figsize=(12, 5))
                        
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_2d, cmap='viridis')
                        plt.title(f'Données complètes t={i}')
                        plt.colorbar()
                        
                        plt.subplot(1, 2, 2)
                        plt.imshow(non_zero_mask, cmap='gray')
                        plt.title(f'Masque non-zéro t={i}')
                        plt.colorbar()
                        
                        plt.tight_layout()
                        plt.savefig(f'partial_data_t{i}.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"    Visualisation sauvée: partial_data_t{i}.png")
                
                # Arrêter l'analyse après les premiers pas de temps vides
                if i > 25 and non_zero == 0:
                    print("    (arrêt de l'analyse - pas de temps vides)")
                    break
        
        # Vérifier si le fichier semble tronqué ou corrompu
        print(f"\nVérification de l'intégrité du fichier:")
        print(f"Taille du fichier 'out': {out_data.nbytes / (1024*1024):.2f} MB")
        
        # Comparer avec les données d'entrée
        if 'inp' in data:
            inp_data = data['inp']
            print(f"Variable 'inp' shape: {inp_data.shape}")
            
            # Vérifier si inp et out ont la même structure temporelle
            inp_non_zero_times = []
            for i in range(len(inp_data.time)):
                if np.any(inp_data.isel(time=i).values != 0):
                    inp_non_zero_times.append(i)
            
            out_non_zero_times = []
            for i in range(len(out_data.time)):
                if np.any(out_data.isel(time=i).values != 0):
                    out_non_zero_times.append(i)
            
            print(f"Pas de temps non-nuls dans 'inp': {inp_non_zero_times}")
            print(f"Pas de temps non-nuls dans 'out': {out_non_zero_times}")
            
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return
    
if __name__ == "__main__":
    analyze_partial_data()
