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
        verbose=False
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
print(f"Chargement de patches de test")

# Test patch 0 (zone polaire)
print("\nPatch 0 (zone polaire):")
try:
    sample = dataset[0]
    print(f"  Patch charge")
    
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
            expected_spatial = (256, 256)
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
    
    print("\n")
    print("=" * 60)
    print(f"Verification des valeurs")
    print("=" * 60)
    
    # Variables satellites (devrait contenir des températures SST)
    for group in VAR_GROUPS:
        for var in VAR_GROUPS[group]:
            key = f"{group}_{var}"
            if key in sample:
                data = sample[key]
                valid_mask = ~np.isnan(data)
                if valid_mask.any():
                    print(f"{key:20s}: min={np.nanmin(data):8.3f}, max={np.nanmax(data):8.3f}, "
                          f"mean={np.nanmean(data):8.3f}, nan={np.isnan(data).sum():7d}/{data.size}")
                else:
                    print(f"{key:20s}: ALL NaN")
    
    # Sea ice fraction (devrait être [0, 1])
    if 'sea_ice_fraction' in sample:
        data = sample['sea_ice_fraction']
        print(f"{'sea_ice_fraction':20s}: min={np.nanmin(data):8.3f}, max={np.nanmax(data):8.3f}, "
              f"mean={np.nanmean(data):8.3f}")
    
    print("\n")
    print("=" * 60)
    print(f"Canaux spatiaux")
    print("=" * 60)
    
    # Latitude (normalisé par 90, devrait être environ [-1, 1])
    if 'lat' in sample:
        data = sample['lat']
        print(f"{'lat':20s}: min={data.min():8.3f}, max={data.max():8.3f}, mean={data.mean():8.3f}")
        print(f"  Latitude reelle: {data.min()*90:.2f} deg a {data.max()*90:.2f} deg")
    
    # Longitude (normalisé par 180, devrait être environ [-1, 1])
    if 'lon' in sample:
        data = sample['lon']
        print(f"{'lon':20s}: min={data.min():8.3f}, max={data.max():8.3f}, mean={data.mean():8.3f}")
        print(f"  Longitude reelle: {data.min()*180:.2f} deg a {data.max()*180:.2f} deg")
    
    # Time (day of year / 366, devrait être [0, 1])
    if 'time' in sample:
        data = sample['time']
        unique_val = np.unique(data)[0] if len(np.unique(data)) == 1 else "varying"
        if isinstance(unique_val, (int, float)):
            print(f"{'time':20s}: unique value={unique_val:.6f} (day of year={int(unique_val*366)})")
        else:
            print(f"{'time':20s}: {unique_val}")
    
    # Surfmask (0=ocean, 1=land)
    if 'surfmask' in sample:
        data = sample['surfmask']
        n_ocean = (data == 0).sum()
        n_land = (data == 1).sum()
        print(f"{'surfmask':20s}: ocean={n_ocean}/{data.size} ({100*n_ocean/data.size:.1f}%), "
              f"land={n_land}/{data.size} ({100*n_land/data.size:.1f}%)")
    
    print("\n")
    print("=" * 60)
    print(f"Fusion slstr + aasti")
    print("=" * 60)
    
    if 'tgt_sst' in sample:
        tgt_sst = sample['tgt_sst']
        tgt_slstr = sample['tgt_slstr_av']
        tgt_aasti = sample['tgt_aasti_av']
        
        # Compter où chaque satellite contribue
        slstr_valid = ~np.isnan(tgt_slstr)
        aasti_valid = ~np.isnan(tgt_aasti)
        both_valid = slstr_valid & aasti_valid
        only_slstr = slstr_valid & ~aasti_valid
        only_aasti = aasti_valid & ~slstr_valid
        neither = ~slstr_valid & ~aasti_valid
        
        print(f"Pixels avec donnees:")
        print(f"  slstr seulement  : {only_slstr.sum():7d} ({100*only_slstr.sum()/tgt_sst.size:.1f}%)")
        print(f"  aasti seulement  : {only_aasti.sum():7d} ({100*only_aasti.sum()/tgt_sst.size:.1f}%)")
        print(f"  les deux         : {both_valid.sum():7d} ({100*both_valid.sum()/tgt_sst.size:.1f}%)")
        print(f"  aucun            : {neither.sum():7d} ({100*neither.sum()/tgt_sst.size:.1f}%)")
        
        # Vérifier la fusion (slstr priority)
        # Dans les zones où les deux sont valides, tgt_sst devrait être égal à tgt_slstr
        if both_valid.any():
            diff_in_both = tgt_sst[both_valid] - tgt_slstr[both_valid]
            max_diff = np.abs(diff_in_both).max()
            print(f"\nDans les zones avec slstr+aasti valides:")
            print(f"  Max difference (tgt_sst - slstr): {max_diff:.6f}")
            if max_diff < 1e-5:
                print(f"  Fusion correcte: slstr a priorite")
            else:
                print(f"  Fusion incorrecte: devrait utiliser slstr")
    
except Exception as e:
    print(f"Erreur lors du chargement patch 0: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test patch du milieu (zone equatoriale/temperee)
print("\n" + "=" * 60)
print("Patch du milieu (zone temperee):")
try:
    middle_idx = len(dataset) // 2
    sample2 = dataset[middle_idx]
    print(f"  Patch {middle_idx} charge")
    
    print(f"\n  Couverture des satellites:")
    for sat in ['aasti', 'avhrr', 'pmw', 'slstr']:
        var_key = f"{sat}_av"
        if var_key in sample2:
            valid = (~np.isnan(sample2[var_key])).sum()
            total = sample2[var_key].size
            print(f"    {sat:10s}: {valid:7d}/{total} ({100*valid/total:5.1f}%)")
    
    print("\n")
    print("=" * 60)
    print("Test termine")
    print("=" * 60)
    
except Exception as e:
    print(f"Erreur lors du chargement patch milieu: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
