import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def visualize_data():
    dataset = xr.open_dataset('test_output/test_logs/test_data.nc')
    n_time = dataset.dims['time']
    t1 = int(input(f"Quel instant temporel (0-{n_time-1}) ? "))
    t2 = int(input(f"Quel instant temporel (0-{n_time-1}) ? "))
    if t1 < 0 or t1 >= n_time or t2 < 0 or t2 >= n_time:
        print("Instant temporel invalide.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20,12))
    variables = ['inp', 'tgt', 'out']
    var_labels = ['Input (sat)', 'Target', 'Output']
    
    for i, t in enumerate([t1, t2]):
        for j, (var, var_label) in enumerate(zip(variables, var_labels)):
            ax = axes[i, j]
            data = dataset[var][t]
            print(f"Variable {var} à t={t}: {np.count_nonzero(~np.isnan(data))} valeurs non-NaN sur {data.size} total")
            
            data_np = data.values
            valid_mask = ~np.isnan(data_np)
            valid_data = data_np[valid_mask]
            if len(valid_data) > 0:
                vmin, vmax = valid_data.min(), valid_data.max()
            else:
                vmin, vmax = 0, 1
            im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f'{var_label} at t={t}')
            plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f'test_output/visualization_{t1}_{t2}.png', dpi=300, bbox_inches='tight')
    print("Visualisation sauvegardée: test_output/visualization_simple.png")


# def visualize_data():
#     ds = xr.open_dataset('test_output/test_logs/test_data.nc')
#     print(f"Shape des données: {ds.inp.shape}")
#     n_time = ds.dims['time']
#     # Trouvons le dernier instant avec des données non-nulles pour 'out'
#     last_valid_t = None
#     for t in range(n_time-1, -1, -1):
#         if np.count_nonzero(ds['out'][t].values) > 0:
#             last_valid_t = t
#             break
#     print(f"Dernier instant avec données valides: t={last_valid_t}")
    
#     time_indices = [0, last_valid_t//2, last_valid_t]
#     time_labels = ['Premier', f'Milieu', f'Dernier valide']
    
#     print("\n=== Diagnostics par instant ===")
#     for i, t_idx in enumerate(time_indices):
#         print(f"\nInstant {time_labels[i]} (t={t_idx}):")
#         for var in ['inp', 'tgt', 'out']:
#             data = ds[var][t_idx].values  # Convertir en numpy array
#             valid_mask = ~np.isnan(data)
#             valid_data = data[valid_mask]
#             if len(valid_data) > 0:
#                 print(f"  {var}: min={valid_data.min():.6f}, max={valid_data.max():.6f}, mean={valid_data.mean():.6f}, non-null={len(valid_data)}/{data.size}")
#             else:
#                 print(f"  {var}: TOUTES LES VALEURS SONT NaN !")
    
#     fig, axes = plt.subplots(3, 3, figsize=(12, 10))
#     variables = ['inp', 'tgt', 'out']
#     var_labels = ['Input (sat)', 'Target', 'Output']
    
#     for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels)):
#         for j, (var, var_label) in enumerate(zip(variables, var_labels)):
#             ax = axes[i, j]
#             data = ds[var][t_idx]
            
#             # Vérifier si les données sont valides
#             data_np = data.values
#             valid_mask = ~np.isnan(data_np)
#             valid_data = data_np[valid_mask]
                
#             print(f"Variable {var} à t={t}: {np.count_nonzero(~np.isnan(data))} valeurs non-NaN sur {data.size} total")
            
#             if len(valid_data) == 0:
#                 # Toutes les valeurs sont NaN
#                 ax.text(0.5, 0.5, 'Toutes NaN', ha='center', va='center', transform=ax.transAxes)
#                 ax.set_title(f'{var_label}\n{t_label} (t={t_idx}) - NaN')
#             else:
#                 # Utiliser vmin/vmax pour améliorer le contraste
#                 vmin, vmax = valid_data.min(), valid_data.max()
#                 if vmin == vmax:
#                     # Valeurs constantes
#                     im = ax.imshow(data, cmap='RdBu_r', aspect='auto')
#                     ax.set_title(f'{var_label}\n{t_label} (t={t_idx}) - Constant: {vmin:.6f}')
#                 else:
#                     im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
#                     ax.set_title(f'{var_label}\n{t_label} (t={t_idx})')
#                 plt.colorbar(im, ax=ax, shrink=0.8)
    
#     plt.tight_layout()
#     plt.savefig('visualization_simple.png', dpi=150, bbox_inches='tight')
#     print("Visualisation sauvegardée: visualization_simple.png")
    
#     print(f"\nStatistiques:")
#     for var in variables:
#         print(f"{var}: min={ds[var].min().values:.3f}, max={ds[var].max().values:.3f}, mean={ds[var].mean().values:.3f}")

if __name__ == '__main__':
    visualize_data()
