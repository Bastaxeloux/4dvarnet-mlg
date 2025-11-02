import os
import sys
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import tempfile
from contrib.SST.data_multires import BaseDataModuleMultiRes
from matplotlib.patches import Rectangle


os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.append('.')
DATA_DIR = "/dmidata/users/malegu/data/netcdf_2024"
OUTPUT_FILE = "figs/SST_multires/multires_sst.png"
NORM_STATS_PATH = "contrib/SST/norm_stats.yaml"
TIME_INDEX = 7
MULTIRES = [10, 3, 1]
VARIABLES = ['aasti_av', 'slstr_av', 'tgt_sst']
MIN_DATA_COVERAGE = 0.20


def get_data_from_batch(batch_dict, var_name, time_idx=0):
    result = {}
    for res_key, item in batch_dict.items():
        factor = int(res_key.split('_x')[-1])
        if isinstance(item, dict):
            if var_name not in item:
                print(f"Variable {var_name} non trouvée dans {res_key}")
                continue
            data = item[var_name]
            lon = item['lon']
            lat = item['lat']
        elif hasattr(item, var_name):
            data = getattr(item, var_name)
            lon = getattr(item, 'lon')
            lat = getattr(item, 'lat')
        else:
            print(f"Variable {var_name} non trouvée dans {res_key} (type: {type(item)})")
            continue
        
        # Convertir en numpy si c'est un tensor
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        if hasattr(lon, 'cpu'):
            lon = lon.cpu().numpy()
        if hasattr(lat, 'cpu'):
            lat = lat.cpu().numpy()
        
        # Gérer les dimensions
        if data.ndim == 4:  # (batch, time, lat, lon)
            data = data[0, time_idx, :, :]
        elif data.ndim == 3:  # (time, lat, lon)
            data = data[time_idx, :, :]
        elif data.ndim == 2:  # (lat, lon)
            pass
        else:
            print(f"Dimension inattendue pour {var_name} dans {res_key}: {data.shape}")
            continue
        # lon/lat shape
        if lon.ndim == 3:  # (batch, lat, lon)
            lon = lon[0, :, :]
            lat = lat[0, :, :]
        elif lon.ndim == 2:  # (lat, lon)
            pass
        else:
            print(f"Dimension inattendue pour lon/lat dans {res_key}: {lon.shape}")
            continue
        result[factor] = (data, lon, lat)
    return result


def plot_multires_comparison(batch_dict, variables, time_idx=0, save_path=None):
    n_vars = len(variables)
    n_res = len(MULTIRES)
    
    fig, axes = plt.subplots(
        n_vars, n_res,
        figsize=(5*n_res, 5*n_vars),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    if n_res == 1:
        axes = axes.reshape(-1, 1)
    
    # Pour chaque variable (ligne)
    for i, var_name in enumerate(variables):
        data_dict = get_data_from_batch(batch_dict, var_name, time_idx)
        # Pour chaque résolution (colonne)
        for j, res in enumerate(MULTIRES):
            ax = axes[i, j]
            if res not in data_dict:
                ax.text(0.5, 0.5, f'Pas de données\npour x{res}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{var_name} - x{res}")
                continue
            
            data, lon, lat = data_dict[res]
            ax.add_feature(cfeature.LAND, color='lightgray', zorder=0)
            ax.coastlines(resolution='110m', linewidth=0.5)
            ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
            lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
            lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
            margin = 0.05
            d_lon = (lon_max - lon_min) * margin
            d_lat = (lat_max - lat_min) * margin
            ax.set_extent([lon_min - d_lon, lon_max + d_lon, lat_min - d_lat, lat_max + d_lat],crs=ccrs.PlateCarree())
            ax.set_aspect('auto')

            data_masked = np.ma.masked_invalid(data)
            im = ax.pcolormesh(lon, lat, data_masked,transform=ccrs.PlateCarree(),cmap='RdYlBu_r',shading='auto')
            lon_center = (lon_min + lon_max) / 2
            lat_center = (lat_min + lat_max) / 2
            if res == 10 and 3 in MULTIRES:
                width_x3 = (lon_max - lon_min) * 3 / 10
                height_x3 = (lat_max - lat_min) * 3 / 10
                rect = Rectangle((lon_center - width_x3/2, lat_center - height_x3/2),  width_x3, height_x3, linewidth=2, edgecolor='lime', facecolor='none', transform=ccrs.PlateCarree(), zorder=10)
                ax.add_patch(rect)
            if res == 3 and 1 in MULTIRES:
                width_x1 = (lon_max - lon_min) * 1 / 3 
                height_x1 = (lat_max - lat_min) * 1 / 3
                rect = Rectangle((lon_center - width_x1/2, lat_center - height_x1/2), width_x1, height_x1, linewidth=2, edgecolor='yellow', facecolor='none', transform=ccrs.PlateCarree(), zorder=10)
                ax.add_patch(rect)
            
            ax.set_title(f"{var_name} - x{res}", fontsize=10)
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=30, shrink=0.8)
            cbar.set_label(var_name, fontsize=8)
    plt.suptitle(f"Multi-Résolution SST (t={time_idx})", fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure sauvegardée: {save_path}")
    
    return fig, axes

def main():
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats = yaml.safe_load(f)['norm_stats']
    sst_files = sorted(glob.glob(f"{DATA_DIR}/*.nc"))
    print(f" {len(sst_files)} fichiers trouvés")
    sst_files = sst_files[:15]
    patch_dims = {'time': 15, 'lat': 256, 'lon': 256}
    strides = {'time': 7, 'lat': 40, 'lon': 40}
    norm_stats_covs = {'sea_ice_fraction': {'type': 'minmax', 'min': 0.0, 'max': 1.0}}
    tmp_dir = tempfile.mkdtemp()
    dm = BaseDataModuleMultiRes(
        sst_paths=sst_files,
        multires=MULTIRES,
        xrds_kw={
            'patch_dims': patch_dims,
            'strides': strides,
            'subsel_patch': False 
        },
        tgt_vars=['slstr_av', 'aasti_av'],
        norm_stats=norm_stats,
        norm_stats_covs=norm_stats_covs,
        dl_kw={'batch_size': 1, 'num_workers': 0}
    )
    dm.subsel_path = tmp_dir
    print(f"DataModule créé avec résolutions: {MULTIRES}")
    dm.setup(stage='train')
    train_loader = dm.train_dataloader()
    

    print(f"   Recherche d'un patch avec au moins {MIN_DATA_COVERAGE*100:.0f}% de données en x1...")
    batch_dict = None
    for i, batch in enumerate(train_loader):
        x1_data = None
        if 'inp_x1' in batch and isinstance(batch['inp_x1'], dict):
            first_var = list(batch['inp_x1'].keys())[0]
            x1_data = batch['inp_x1'][first_var]
        elif hasattr(batch, 'inp_x1'):
            x1_dict = getattr(batch, 'inp_x1')
            if isinstance(x1_dict, dict):
                first_var = list(x1_dict.keys())[0]
                x1_data = x1_dict[first_var]
        
        if x1_data is not None:
            # Convertir en numpy si nécessaire
            if hasattr(x1_data, 'cpu'):
                x1_data = x1_data.cpu().numpy()
            
            # Calculer le pourcentage de données valides
            valid_ratio = np.sum(~np.isnan(x1_data)) / x1_data.size
            
            if valid_ratio >= MIN_DATA_COVERAGE:
                batch_dict = batch
                print(f"Patch trouvé (index {i}) avec {valid_ratio*100:.1f}% de données en x1")
                break
        
        if i > 100:
            print(f"Aucun patch trouvé après {i} tentatives, utilisation du dernier")
            batch_dict = batch
            break
    
    for key, item in batch_dict.items():
        if isinstance(item, dict):
            first_var = list(item.keys())[0]
            print(f"      {key}: {first_var} shape = {item[first_var].shape}")
    
    # 6. Visualisation
    print(f"\n6. Génération de la visualisation...")
    print(f"   Variables: {VARIABLES}")
    print(f"   Résolutions: {MULTIRES}")
    print(f"   Timestep: {TIME_INDEX}")
    
    fig, axes = plot_multires_comparison(
        batch_dict,
        VARIABLES,
        time_idx=TIME_INDEX,
        save_path=OUTPUT_FILE
    )
    plt.show()

if __name__ == "__main__":
    main()