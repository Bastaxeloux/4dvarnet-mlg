"""
Test exact de l'interpolation x10 -> x3 comme dans le vrai training.
On copie-colle la fonction interpolate_torch du modèle pour tester EXACTEMENT
ce qui sera utilisé pendant l'entraînement.
"""

import sys
sys.path.append('/home/malegu/4D-MLG/Croscim')

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from contrib.SST.load_data import organize_by_resolution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def interpolate_torch(coarse_dict, lon_coarse, lat_coarse, lon_target, lat_target):
    result = {}
    
    for var, tensor in coarse_dict.items():
        if (tensor is None) or (var in ["time", "lat", "lon"]):
            continue
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        T, Hc, Wc = tensor.shape[1:]
        B = lat_target.shape[0]
        Hf = lat_target.shape[1]  # Hauteur de la grille cible
        Wf = lat_target.shape[2]  # Largeur de la grille cible
        out = np.zeros((B, T, Hf, Wf), dtype=np.float32)
        for b in range(B):
            lat_c_2d = lat_coarse[b].cpu().numpy() if hasattr(lat_coarse[b], "cpu") else lat_coarse[b]
            lon_c_2d = lon_coarse[b].cpu().numpy() if hasattr(lon_coarse[b], "cpu") else lon_coarse[b]
            lat_t_2d = lat_target[b].cpu().numpy() if hasattr(lat_target[b], "cpu") else lat_target[b]
            lon_t_2d = lon_target[b].cpu().numpy() if hasattr(lon_target[b], "cpu") else lon_target[b]
            lat_c_1d = lat_c_2d[:, 0]  # (Hc,) - extract first column
            lon_c_1d = lon_c_2d[0, :]  # (Wc,) - extract first row
            lat_t_1d = lat_t_2d[:, 0]  # (Hf,)
            lon_t_1d = lon_t_2d[0, :]  # (Wf,)
            Lon_t, Lat_t = np.meshgrid(lon_t_1d, lat_t_1d, indexing="xy")
            target_points = np.stack([Lat_t.ravel(), Lon_t.ravel()], axis=-1)  # (Hf*Wf, 2)
            tensor_b = tensor[b] 
            for t in range(T):
                data_t = tensor_b[t]
                f_interp = RegularGridInterpolator(
                    (lat_c_1d, lon_c_1d),
                    data_t, 
                    bounds_error=False, fill_value=np.nan)
                interp_vals = f_interp(target_points).reshape(Hf, Wf)
                out[b, t] = interp_vals
        result[var] = torch.tensor(out).to(device)
    return result


def load_patches(data_dir, num_patches=3):
    """Load x10 and x3 patches"""
    data_path = Path(data_dir)
    all_files = sorted(list(data_path.glob('*.zarr')))
    
    print(f"\nFound {len(all_files)} total files in {data_dir}")
    
    by_res = organize_by_resolution(all_files)
    print(f"Resolutions found: {list(by_res.keys())}")
    
    x10_files = by_res.get(10, [])[:num_patches]
    x3_files = by_res.get(3, [])[:num_patches]
    
    print(f"\nLoading {len(x10_files)} x10 patches...")
    print(f"Loading {len(x3_files)} x3 patches...")
    
    x10_patches = [xr.open_zarr(f) for f in x10_files]
    x3_patches = [xr.open_zarr(f) for f in x3_files]
    
    return x10_patches, x3_patches


def test_interpolation(x10_patch, x3_patch, shift_h=0, shift_w=0):
    """Test interpolation on one patch pair
    
    Args:
        x10_patch: x10 data
        x3_patch: x3 data
        shift_h: vertical shift in pixels (0 = no shift, 1 = shift down by 1 pixel)
        shift_w: horizontal shift in pixels (0 = no shift, 1 = shift right by 1 pixel)
    """
    
    # Prepare coarse dict (x10 data)
    # Utilise les observations (pas tgt_sst qui est créé pendant le training)
    coarse_dict = {}
    test_vars = ['aasti_av', 'avhrr_av', 'pmw_av', 'slstr_av']
    available_vars = []
    
    for var in test_vars:
        if var in x10_patch and var in x3_patch:
            data_x10 = x10_patch[var].values  # (H, W) - PAS de dimension temporelle!
            data_x3 = x3_patch[var].values
            # Vérifie que les données ne sont pas toutes NaN
            if not np.all(np.isnan(data_x10)) and not np.all(np.isnan(data_x3)):
                # Ajouter dimensions batch et temps: (H, W) → (1, 1, H, W)
                coarse_dict[var] = torch.from_numpy(data_x10).float().unsqueeze(0).unsqueeze(0)
                available_vars.append(var)
    
    if not available_vars:
        print("WARNING: No valid observation data found!")
        return None
    
    shift_str = f" (shift: h={shift_h}, w={shift_w})" if (shift_h != 0 or shift_w != 0) else ""
    print(f"Testing with variables: {available_vars}{shift_str}")
    
    # Prepare coordinates
    # Les fichiers zarr ont des coordonnées 1D (lat, lon), il faut créer des meshgrids 2D
    lon_x10_1d = x10_patch['lon'].values  # (W,)
    lat_x10_1d = x10_patch['lat'].values  # (H,)
    lon_x3_1d = x3_patch['lon'].values
    lat_x3_1d = x3_patch['lat'].values
    
    # Créer des grilles 2D
    lon_x10_2d, lat_x10_2d = np.meshgrid(lon_x10_1d, lat_x10_1d)  # (H, W)
    lon_x3_2d, lat_x3_2d = np.meshgrid(lon_x3_1d, lat_x3_1d)
    
    # Ajouter la dimension batch
    lon_x10 = torch.from_numpy(lon_x10_2d).float().unsqueeze(0)  # (1, H, W)
    lat_x10 = torch.from_numpy(lat_x10_2d).float().unsqueeze(0)
    lon_x3 = torch.from_numpy(lon_x3_2d).float().unsqueeze(0)
    lat_x3 = torch.from_numpy(lat_x3_2d).float().unsqueeze(0)
    
    print(f"\nInterpolating from x10 to x3:")
    print(f"  x10 grid: lat {lat_x10.shape}, lon {lon_x10.shape}")
    print(f"  x3 grid: lat {lat_x3.shape}, lon {lon_x3.shape}")
    
    # Interpolate using EXACT function from models.py
    interpolated_dict = interpolate_torch(coarse_dict, lon_x10, lat_x10, lon_x3, lat_x3)
    
    # Apply spatial shift if requested
    if shift_h != 0 or shift_w != 0:
        print(f"Applying spatial shift: h={shift_h}, w={shift_w}")
        for var_name in interpolated_dict.keys():
            data = interpolated_dict[var_name]  # (B, T, H, W) or (1, 1, H, W)
            # Apply roll to shift the data
            data_shifted = torch.roll(data, shifts=(shift_h, shift_w), dims=(2, 3))
            interpolated_dict[var_name] = data_shifted
    
    # Compute differences
    stats = {}
    for var_name, interp_tensor in interpolated_dict.items():
        if var_name in x3_patch:
            actual_x3 = torch.from_numpy(x3_patch[var_name].values).float()  # (H, W)
            
            # Remove batch and time dims: (1, 1, H, W) => (H, W)
            interp_np = interp_tensor.squeeze(0).squeeze(0).cpu().numpy()
            actual_np = actual_x3.cpu().numpy()
            
            diff = interp_np - actual_np
            
            stats[var_name] = {
                'min': np.nanmin(diff),
                'max': np.nanmax(diff),
                'mean': np.nanmean(diff),
                'std': np.nanstd(diff),
                'abs_max': np.nanmax(np.abs(diff)),
                'rmse': np.sqrt(np.nanmean(diff**2))
            }
            
            print(f"\n{var_name} difference statistics:")
            print(f"  Min: {stats[var_name]['min']:.6e}")
            print(f"  Max: {stats[var_name]['max']:.6e}")
            print(f"  Mean: {stats[var_name]['mean']:.6e}")
            print(f"  Std: {stats[var_name]['std']:.6e}")
            print(f"  Max(abs): {stats[var_name]['abs_max']:.6e}")
            print(f"  RMSE: {stats[var_name]['rmse']:.6e}")
    
    return stats, interpolated_dict, {k: x3_patch[k].values for k in interpolated_dict.keys() if k in x3_patch}, available_vars


def plot_results(interpolated_dict, x3_actual, available_vars, save_path):
    """Plot comparison"""
    var = available_vars[0]  # Utilise la première variable disponible
    if var not in interpolated_dict:
        print(f"Variable {var} not found")
        return
    
    print(f"Plotting variable: {var}")
    
    # Les données sont 2D (H, W), pas 3D
    interp_np = interpolated_dict[var].squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
    actual_np = x3_actual[var]  # (H, W)
    diff_np = interp_np - actual_np
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Interpolated
    im1 = axes[0].imshow(interp_np, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Interpolated x10→x3\n{var}')
    plt.colorbar(im1, ax=axes[0])
    
    # Actual
    im2 = axes[1].imshow(actual_np, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Actual x3\n{var}')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    vmax_diff = np.nanmax(np.abs(diff_np))
    im3 = axes[2].imshow(diff_np, cmap='seismic', aspect='auto',
                         vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title(f'Difference\nmax|diff|={vmax_diff:.2e}')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")


def main():
    print("="*80)
    print("TEST D'INTERPOLATION")
    print("="*80)
    
    data_dir = '/home/malegu/4D-MLG/Croscim/data/mounted/2024'
    num_patches = 3
    
    # Load data
    x10_patches, x3_patches = load_patches(data_dir, num_patches)
    
    if not x10_patches or not x3_patches:
        print("ERROR: No patches found!")
        return
    
    # Test configurations: (shift_h, shift_w, suffix)
    test_configs = [
        (0, 0, "no_shift"),
        (-1,1, "shift_1px_left_down"),
    ]
    
    for shift_h, shift_w, suffix in test_configs:
        print(f"\n\n{'#'*80}")
        print(f"TEST CONFIGURATION: {suffix}")
        print(f"{'#'*80}")
        
        # Test on all patches
        all_stats = []
        all_vars = []
        for i, (x10_patch, x3_patch) in enumerate(zip(x10_patches, x3_patches)):
            print(f"\n")
            print(f"PATCH {i+1}/{num_patches}")
            print(f"{'='*80}")
            
            result = test_interpolation(x10_patch, x3_patch, shift_h=shift_h, shift_w=shift_w)
            if result is None:
                print(f"Skipping patch {i+1} - no valid data")
                continue
            
            stats, interp_dict, x3_actual, available_vars = result
            all_stats.append(stats)
            all_vars.append(available_vars)
            
            # Plot first patch
            if i == 0:
                save_path = Path(f'/home/malegu/4D-MLG/Croscim/utils/interpolation_test_{suffix}.png')
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plot_results(interp_dict, x3_actual, available_vars, save_path)
        
        # Summary
        print(f"\n")
        print("RÉSUMÉ SUR TOUS LES PATCHES")
        if not all_stats:
            print("\nAUCUNE DONNÉE VALIDE TROUVÉE")
            continue
        
        # Utilise la première variable disponible
        var = all_vars[0][0] if all_vars else None
        if var and all_stats and var in all_stats[0]:
            rmses = [s[var]['rmse'] for s in all_stats if var in s]
            max_diffs = [s[var]['abs_max'] for s in all_stats if var in s]
            
            print(f"\nVariable testée: {var}")
            print(f"Nombre de patches valides: {len(all_stats)}")
            print(f"RMSE moyen: {np.mean(rmses):.6e}")
            print(f"Différence absolue max: {max(max_diffs):.6e}")
            
            # Critère de succès
            threshold = 1e-5
            if max(max_diffs) < threshold:
                print(f"\nTEST RÉUSSI: différence < {threshold:.0e}")
            else:
                print(f"\nATTENTION: différence ≥ {threshold:.0e}")
        else:
            print("\nAUCUNE VARIABLE COMMUNE TROUVÉE")

if __name__ == '__main__':
    main()