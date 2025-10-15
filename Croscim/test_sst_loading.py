import numpy as np
import xarray as xr
from contrib.SST.data import XrDataset
from contrib.SST.load_data import VAR_GROUPS, COVARIATES
import datetime
import glob
import os


print("=" * 60)
print("TEST CHARG SST")

SST_DATA_DIR = "/dmidata/users/malegu/data/netcdf_2024"

# Prendre les 15 premiers jours de 2024 (format: YYYYMMDDHH_13vars.nc)
sst_files = sorted(glob.glob(f"{SST_DATA_DIR}/202401*12_13vars.nc"))[:15]
times = [datetime.datetime.strptime(os.path.basename(f)[:10], "%Y%m%d%H") for f in sst_files]

patch_dims = {
    'time': len(times),
    'lat': 256,
    'lon': 256
}

strides = {
    'time': 1,
    'lat': 28,
    'lon': 28
}

print("\n")
print("=" * 60)
print(f"Config")
print(f"Répertoire: {SST_DATA_DIR}")
print(f"Patch dims: {patch_dims}")
print(f"Strides: {strides}")

print("\n")
print("=" * 60)
print(f"Chargement du masque")
first_file = xr.open_dataset(sst_files[0])
mask = first_file.surfmask.values  # (3600, 7200)
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Océan (0): {np.sum(mask == 0)} pixels soit {np.sum(mask == 0)/mask.size*100:.2f}%")
print(f"Terre (1): {np.sum(mask == 1)} pixels soit {np.sum(mask == 1)/mask.size*100:.2f}%")

print("\n")
print("=" * 60)
print(f"Initialisation du XrDataset SST")
try:
    dataset = XrDataset(
        sst_daily_paths=sst_files,
        tgt_vars=['slstr_av', 'aasti_av'],
        mask=mask,
        times=times,
        patch_dims=patch_dims,
        strides=strides,
        resize=1,  # Pas de coarsening pour le test (résolution native)
        res=5.0,  # 5km
        pad=False,
        stride_test=False,
        subsel_patch=False,
        load_data=False,
        verbose=True  # Mode debug
    )
    print(f" Dataset créé")
    print(f"Taille du dataset: {len(dataset)} patches. Dimensions: {dataset.da_dims}")
    print(f" Grid lat_1d: {dataset.lat_1d.shape}")
    print(f" Grid lon_1d: {dataset.lon_1d.shape}")
    print(f" Nombre de patches spatiaux: {dataset.ds_size}")

except Exception as e:
    print(f" Erreur lors de la création: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n")
print("=" * 60)
print(f"Chargement d'un patch de test")
try:
    sample = dataset[0]
    print(f"Patch chargé")
    
    print(f"\nVérification de la structure des données")
    print(f"Variables dans ce patch:")
    for key, val in sample.items():
        if isinstance(val, np.ndarray):
            print(f"   - {key:20s}: shape {val.shape}, dtype {val.dtype}")
    
    print(f"\nDimensions")
    
    # Canaux temporels (9 variables × nt timesteps)
    temporal_vars = []
    nt_expected = len(times)
    for sat in ['aasti', 'avhrr', 'pmw', 'slstr']:
        for var in ['av', 'std']:
            var_key = f"{sat}_{var}"
            if var_key in sample:
                temporal_vars.append(var_key)
                nt, nlat, nlon = sample[var_key].shape
                expected = (nt_expected, 256, 256)
                if (nt, nlat, nlon) == expected:
                    print(f" {var_key}: {sample[var_key].shape} (correct)")
                else:
                    print(f" {var_key}: {sample[var_key].shape} (attendu {expected})")
    
    # sea_ice_fraction
    if 'sea_ice_fraction' in sample:
        print(f" sea_ice_fraction: {sample['sea_ice_fraction'].shape}")
    
    # Canaux spatiaux (4: lat, lon, time, surfmask)
    spatial_vars = ['lat', 'lon', 'time', 'surfmask']
    for var in spatial_vars:
        if var in sample:
            expected_spatial = (240, 240)
            if sample[var].shape == expected_spatial:
                print(f" {var}: {sample[var].shape} (spatial, correct)")
            else:
                print(f" {var}: {sample[var].shape} (attendu {expected_spatial})")
    
    # Targets
    print(f"\nVérification des targets")
    for tgt in ['tgt_sst', 'tgt_slstr_av', 'tgt_aasti_av']:
        if tgt in sample:
            print(f"  {tgt}: {sample[tgt].shape}")
    
    print("\n")
    print("=" * 60)
    print(f"Comptage des canaux")
    n_temporal = len(temporal_vars) + 1  # +1 pour sea_ice_fraction
    n_spatial = len([v for v in spatial_vars if v in sample])
    total_channels = n_temporal * nt_expected + n_spatial  # 9 vars × nt timesteps + 4 spatial
    print(f"Variables temporelles: {n_temporal} x {nt_expected} timesteps = {n_temporal * nt_expected}")
    print(f"Variables spatiales: {n_spatial}")
    print(f"TOTAL: {total_channels} canaux")
    
    expected_total = 9 * nt_expected + 4
    if total_channels == expected_total:
        print(f" Nombre de canaux correct: {expected_total}")
    else:
        print(f" Nombre de canaux incorrect: {total_channels} (attendu {expected_total})")
    
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
