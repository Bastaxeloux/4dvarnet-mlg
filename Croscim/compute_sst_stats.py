import numpy as np
import xarray as xr
from pathlib import Path
import yaml

data_dir = Path("/dmidata/users/malegu/data/netcdf_2024")
all_files = sorted(data_dir.glob("*_13vars.nc"))
print(f"Total fichiers disponibles: {len(all_files)}")
step = max(1, len(all_files) // 24)
selected_files = all_files[::step][:24]
print(f"Fichiers selectionnes: {len(selected_files)}")

# Variables à analyser
variables = {
    'aasti': ['av', 'std'],
    'avhrr': ['av', 'std'],
    'pmw': ['av', 'std'],
    'slstr': ['av', 'std']
}

covariates = ['sea_ice_fraction']

print("\n" + "=" * 60)
print("Calcul des statistiques")
print("=" * 60)

# Accumuler les valeurs valides pour chaque variable
stats = {}
for sat in variables:
    for var in variables[sat]:
        stats[f"{sat}_{var}"] = {'values': []}

for cov in covariates:
    stats[cov] = {'values': []}

# Charger et accumuler
for i, file_path in enumerate(selected_files):
    print(f"  [{i+1:2d}/{len(selected_files)}] {file_path.name}", end='')
    
    ds = xr.open_dataset(file_path)
    
    # Pour chaque variable satellite
    for sat in variables:
        for var in variables[sat]:
            var_key = f"{sat}_{var}"
            if var_key in ds:
                data = ds[var_key].values
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    # Échantillonner pour ne pas surcharger la mémoire
                    # Prendre max 100k valeurs par fichier
                    if len(valid_data) > 100000:
                        indices = np.random.choice(len(valid_data), 100000, replace=False)
                        valid_data = valid_data[indices]
                    stats[var_key]['values'].append(valid_data)
    
    # Pour les covariates
    for cov in covariates:
        if cov in ds:
            data = ds[cov].values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                if len(valid_data) > 100000:
                    indices = np.random.choice(len(valid_data), 100000, replace=False)
                    valid_data = valid_data[indices]
                stats[cov]['values'].append(valid_data)
    
    ds.close()
    print(" OK")

print("\n" + "=" * 60)
print("Statistiques finales")
print("=" * 60)


norm_stats = {}
for sat in variables:
    norm_stats[sat] = {}
    for var in variables[sat]:
        var_key = f"{sat}_{var}"
        if stats[var_key]['values']:
            all_values = np.concatenate(stats[var_key]['values'])
            mean_val = float(np.mean(all_values))
            std_val = float(np.std(all_values))
            
            norm_stats[sat][var] = {
                'mean': round(mean_val, 3),
                'std': round(std_val, 3),
                'type': 'zscore'
            }
            
            print(f"{var_key:20s}: mean={mean_val:8.3f}, std={std_val:8.3f}, n_samples={len(all_values):10d}")
        else:
            print(f"{var_key:20s}: NO DATA")
            norm_stats[sat][var] = {'mean': 0.0, 'std': 1.0, 'type': 'zscore'}

# Statistiques pour covariates (minmax normalization)
norm_stats_covs = {}
for cov in covariates:
    if stats[cov]['values']:
        all_values = np.concatenate(stats[cov]['values'])
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))
        
        norm_stats_covs[cov] = {
            'min': round(min_val, 3),
            'max': round(max_val, 3),
            'type': 'minmax'
        }
        
        # print(f"{cov:20s}: min={min_val:8.3f}, max={max_val:8.3f}, n_samples={len(all_values):10d}")
    else:
        # print(f"{cov:20s}: NO DATA")
        norm_stats_covs[cov] = {'min': 0.0, 'max': 1.0, 'type': 'minmax'}

yaml_content = {
    'norm_stats': norm_stats,
    'norm_stats_covs': norm_stats_covs
}

output_file = Path("contrib/SST/norm_stats.yaml")
with open(output_file, 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

print(f"\nStatistiques sauvegardees dans: {output_file}")

print("\nnorm_stats:")
for sat in norm_stats:
    print(f"  {sat}:")
    for var in norm_stats[sat]:
        stats_dict = norm_stats[sat][var]
        print(f"    {var}: {{mean: {stats_dict['mean']}, std: {stats_dict['std']}, type: {stats_dict['type']}}}")

print("\nnorm_stats_covs:")
for cov in norm_stats_covs:
    stats_dict = norm_stats_covs[cov]
    print(f"  {cov}: {{min: {stats_dict['min']}, max: {stats_dict['max']}, type: {stats_dict['type']}}}")
