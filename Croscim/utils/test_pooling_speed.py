import numpy as np
import time
import glob
import xarray as xr
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrib.SST.load_data import concatenate, VAR_GROUPS, COVARIATES


# Verfier GPU disponibles avec nvtop ou nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MOUNT_DIR = "/home/malegu/4D-MLG/Croscim/data/mounted/2024"

sst_files_x1 = sorted(glob.glob(f"{MOUNT_DIR}/*_x1.zarr"))[:15]
sst_files_x3 = sorted(glob.glob(f"{MOUNT_DIR}/*_x3.zarr"))[:15]
sst_files_x10 = sorted(glob.glob(f"{MOUNT_DIR}/*_x10.zarr"))[:15]

print(f"Fichiers Zarr x1 (fine):   {len(sst_files_x1)} fichiers")
print(f"Fichiers Zarr x3 (coarse): {len(sst_files_x3)} fichiers")
print(f"Fichiers Zarr x10 (coarse):{len(sst_files_x10)} fichiers")

if len(sst_files_x1) == 0:
    print("ERROR: No Zarr files found! Check mount.")
    sys.exit(1)


ds_x1_sample = xr.open_zarr(sst_files_x1[0])
lat_vals = ds_x1_sample.lat.values
lon_vals = ds_x1_sample.lon.values

# Prendre une région d'environ 2560 pixels (c'est ~256*10 pour le contexte x10)
lat_end_idx = min(2560, len(lat_vals)-1)
lon_end_idx = min(2560, len(lon_vals)-1)

lat_start, lat_end = lat_vals[0], lat_vals[lat_end_idx]
lon_start, lon_end = lon_vals[0], lon_vals[lon_end_idx]

ds_x1_sample.close()

slices = {
    "lon": slice(lon_start, lon_end),
    "lat": slice(lat_start, lat_end)
}
all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]

print(f"\nRégion: lat=[{lat_start:.2f}, {lat_end:.2f}], lon=[{lon_start:.2f}, {lon_end:.2f}]")
print(f"Variables à charger: {len(all_sst_vars + COVARIATES)}")
print(f"(Pour x1: ~{lat_end_idx}x{lon_end_idx} pixels, pour x3: ~{lat_end_idx//3}x{lon_end_idx//3}, pour x10: ~{lat_end_idx//10}x{lon_end_idx//10})")

print("\n" + "="*60)
print("Zarr x1 direct (haute résolution, sans pooling)")
print("="*60)
start = time.time()
ds_x1_direct = concatenate(
    sst_files_x1,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=1,
    domain_limits=None,
    verbose=True
)
time_x1_direct = time.time() - start

print("\n" + "="*60)
print("Zarr x3 direct (basse résolution, sans pooling)")
print("="*60)
start = time.time()
ds_x3_direct = concatenate(
    sst_files_x3,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=1,  # pas de pooling, c'est déjà x3
    domain_limits=None,
    verbose=True
)
time_x3_direct = time.time() - start

print("\n" + "="*60)
print("Zarr x10 direct (basse résolution, sans pooling)")
print("="*60)
start = time.time()
ds_x10_direct = concatenate(
    sst_files_x10,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=1,  # pas de pooling, c'est déjà x10
    domain_limits=None,
    verbose=True
)
time_x10_direct = time.time() - start

print("\n" + "="*60)
print("Zarr x1 + pooling GPU x10")
print("="*60)
start = time.time()
ds_x1_pool_gpu = concatenate(
    sst_files_x1,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=10,  # Pooling via GPU
    domain_limits=None,
    verbose=True,
    use_gpu=True
)
time_x1_pool_gpu = time.time() - start

print("\n" + "="*60)
print("Zarr x1 + pooling CPU x10")
print("="*60)
start = time.time()
ds_x1_pool_cpu = concatenate(
    sst_files_x1,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=10,  # Pooling via CPU
    domain_limits=None,
    verbose=True,
    use_gpu=False
)
time_x1_pool_cpu = time.time() - start

print("\n" + "="*60)
print("COMPARATIF (15 jours, région 2560x2560)")
print("="*60)
print(f"Zarr x1 direct:         {time_x1_direct:.2f}s")
print(f"Zarr x3 direct:         {time_x3_direct:.2f}s")
print(f"Zarr x10 direct:        {time_x10_direct:.2f}s")
print(f"Zarr x1 + Pool CPU x10: {time_x1_pool_cpu:.2f}s")
print(f"Zarr x1 + Pool GPU x10: {time_x1_pool_gpu:.2f}s")
print(f"\nRatio (Pool GPU vs x10 direct): {time_x1_pool_gpu / time_x10_direct:.2f}x")
print(f"Ratio (Pool CPU vs x10 direct): {time_x1_pool_cpu / time_x10_direct:.2f}x")
