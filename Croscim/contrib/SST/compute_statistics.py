import os
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
import random
import xarray as xr
import numpy as np
from glob import glob
from collections import defaultdict
import yaml
from tqdm import tqdm

# Nombre de fichiers à échantillonner
N_SAMPLES = 366

# Spécifie les types de normalisation attendus
norm_types = {
    "zscore": lambda arr: {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "type": "zscore"},
    "minmax": lambda arr: {"min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)), "type": "minmax"},
}

# Configuration : dictionnaire des variables satellites SST avec leur type de normalisation
VAR_GROUPS = {
    "aasti": {
        "av": "zscore",   # average SST
        "std": "zscore"   # standard deviation
    },
    "avhrr": {
        "av": "zscore",
        "std": "zscore"
    },
    "pmw": {
        "av": "zscore",
        "std": "zscore"
    },
    "slstr": {
        "av": "zscore",
        "std": "zscore"
    }
}

COVARIATES = {
    "sea_ice_fraction": "minmax"
}

# Variable spéciale : tgt_sst (fusion de slstr et aasti)
SPECIAL_VARS = {
    "tgt_sst": "zscore"
}


def compute_stats_from_files(file_list, variables, norm_types, compute_tgt_sst=False):
    # Initialisation accumulée pour chaque variable
    accum = {
        var: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf
        } 
        for var in variables
    }
    
    # Accumulateur pour tgt_sst si nécessaire
    if compute_tgt_sst:
        accum["tgt_sst"] = {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": np.inf,
            "max": -np.inf
        }

    # Barre de progression
    for f in tqdm(file_list, desc="Processing files", unit="file"):
        try:
            ds = xr.open_dataset(f)
            
            # Variables satellites standard
            for var in variables:
                if var in ds:
                    data = ds[var].values
                    arr = data.flatten()
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        accum[var]["count"] += arr.size
                        accum[var]["sum"] += arr.sum()
                        accum[var]["sum_sq"] += (arr ** 2).sum()
                        accum[var]["min"] = min(accum[var]["min"], arr.min())
                        accum[var]["max"] = max(accum[var]["max"], arr.max())
            
            # Calcul de tgt_sst (fusion slstr + aasti)
            if compute_tgt_sst and "slstr_av" in ds and "aasti_av" in ds:
                slstr = ds["slstr_av"].values
                aasti = ds["aasti_av"].values
                tgt_sst = np.where(~np.isnan(slstr), slstr, aasti)
                
                arr = tgt_sst.flatten()
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    accum["tgt_sst"]["count"] += arr.size
                    accum["tgt_sst"]["sum"] += arr.sum()
                    accum["tgt_sst"]["sum_sq"] += (arr ** 2).sum()
                    accum["tgt_sst"]["min"] = min(accum["tgt_sst"]["min"], arr.min())
                    accum["tgt_sst"]["max"] = max(accum["tgt_sst"]["max"], arr.max())
            
            ds.close()
        except Exception as e:
            tqdm.write(f"Skipping {f}: {e}")
            continue

    # Calcul des stats finales
    stats = {}
    all_vars = list(variables.keys()) + (["tgt_sst"] if compute_tgt_sst else [])
    
    for var in all_vars:
        agg = accum[var]
        if agg["count"] == 0:
            continue
        mean = agg["sum"] / agg["count"]
        var_val = (agg["sum_sq"] / agg["count"]) - mean**2
        std = np.sqrt(max(var_val, 0.0))

        norm_type = variables.get(var, "zscore") if var != "tgt_sst" else "zscore"
        if norm_type == "zscore":
            stats[var] = {"mean": float(mean), "std": float(std), "type": "zscore"}
        elif norm_type == "minmax":
            stats[var] = {"min": float(agg["min"]), "max": float(agg["max"]), "type": "minmax"}

    return stats


def normalize_group(path, variables, norm_types, N_SAMPLES=100, compute_tgt_sst=False):
    files = sorted(path)
    files = random.sample(files, min(N_SAMPLES, len(files)))
    return compute_stats_from_files(files, variables, norm_types, compute_tgt_sst)

def build_all_normalization_dicts(sst_dir):
    all_sat_vars = {}
    for sat, vars_dict in VAR_GROUPS.items():
        for var, norm_type in vars_dict.items():
            var_name = f"{sat}_{var}"
            all_sat_vars[var_name] = norm_type
    print(f"\nCalcul des stats sur {len(sst_dir)} fichiers...")
    stats = normalize_group(sst_dir, all_sat_vars, norm_types, N_SAMPLES, compute_tgt_sst=True)
    
    # Réorganiser les stats par satellite
    norm_stats = {}
    for sat, vars_dict in VAR_GROUPS.items():
        norm_stats[sat] = {}
        for var in vars_dict.keys():
            var_name = f"{sat}_{var}"
            if var_name in stats:
                norm_stats[sat][var] = stats[var_name]
    
    # Extraire tgt_sst
    if "tgt_sst" in stats:
        norm_stats["tgt_sst"] = stats["tgt_sst"]
    
    print("\nCalcul des stats COVARIATES (sea_ice_fraction)...")
    covs_stats = normalize_group(sst_dir, COVARIATES, norm_types, N_SAMPLES)

    return norm_stats, covs_stats


# Chemins des fichiers SST
sst_path = glob("/dmidata/users/malegu/data/netcdf_2024/*nc")

print(f"\nTrouvé {len(sst_path)} fichiers SST")

norm_stats, norm_stats_covs = build_all_normalization_dicts(sst_path)

print("\nStatistiques des satellites:")
for sat, vars_dict in norm_stats.items():
    if sat != "tgt_sst":
        print(f"\n  {sat.upper()}:")
        for var, stats in vars_dict.items():
            if stats["type"] == "zscore":
                print(f"    {var}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            else:
                print(f"    {var}: min={stats['min']:.3f}, max={stats['max']:.3f}")

if "tgt_sst" in norm_stats:
    print(f"\n  TGT_SST (fusion globale):")
    print(f"    mean={norm_stats['tgt_sst']['mean']:.3f}, std={norm_stats['tgt_sst']['std']:.3f}")

print("\nCovariates:")
for cov, stats in norm_stats_covs.items():
    if stats["type"] == "zscore":
        print(f"  {cov}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    else:
        print(f"  {cov}: min={stats['min']:.3f}, max={stats['max']:.3f}")

# Enregistrement .txt (format Python)
with open("contrib/SST/norm_stats.txt", "w") as f:
    f.write("norm_stats = ")
    f.write(repr(norm_stats))
    f.write("\n\n")
    f.write("norm_stats_covs = ")
    f.write(repr(norm_stats_covs))


with open("contrib/SST/norm_stats.yaml", "w") as f:
    yaml.dump({"norm_stats": norm_stats, "norm_stats_covs": norm_stats_covs}, 
              f, sort_keys=False, default_flow_style=False)

print("\nStats sauvegardées dans contrib/SST/norm_stats.yaml et norm_stats.txt")
