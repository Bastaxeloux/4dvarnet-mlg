import numpy as np
import time
import glob
import xarray as xr
import os
from contrib.SST.load_data import concatenate, VAR_GROUPS, COVARIATES

# Verfier GPU disponibles avec nvtop ou nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = "/dmidata/users/malegu/data/netcdf_2024"
sst_files_zarr = sorted(glob.glob(f"{DATA_DIR}/*.zarr"))[:15]
sst_files_nc = sorted(glob.glob(f"{DATA_DIR}/*.nc"))[:15]

print(f"Fichiers Zarr: {len(sst_files_zarr)} fichiers")
print(f"Fichiers NetCDF: {len(sst_files_nc)} fichiers")

# Utiliser les fichiers zarr pour le test
sst_files = sst_files_zarr
print(f"\nTest avec ZARR (optimisé pour NFS)")

ds_sample = xr.open_zarr(sst_files[0])
lat_vals = ds_sample.lat.values
lon_vals = ds_sample.lon.values
ds_sample.close()
lat_start, lat_end = lat_vals[0], lat_vals[min(2560, len(lat_vals)-1)]
lon_start, lon_end = lon_vals[0], lon_vals[min(2560, len(lon_vals)-1)]

slices = {
    "lon": slice(lon_start, lon_end),
    "lat": slice(lat_start, lat_end)
}
all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]

print(f"\nRégion: lat=[{lat_start:.2f}, {lat_end:.2f}], lon=[{lon_start:.2f}, {lon_end:.2f}]")
print(f"Variables à charger: {len(all_sst_vars + COVARIATES)}")

print("\n" + "="*60)
print("NetCDF x10")
print("="*60)
start = time.time()
ds_nc = concatenate(
    sst_files_nc,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=10,
    domain_limits=None,
    verbose=True
)
time_nc = time.time() - start

print("\n" + "="*60)
print("ZARR x10 CPU")
print("="*60)
start = time.time()
ds_coarsened_cpu = concatenate(
    sst_files,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=10,
    domain_limits=None,
    verbose=True,
    use_gpu=False  # Force CPU
)
time_zarr_cpu = time.time() - start

print("\n" + "="*60)
print("ZARR x10 GPU")
print("="*60)
start = time.time()
ds_coarsened_gpu = concatenate(
    sst_files,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=10,
    domain_limits=None,
    verbose=True,
    use_gpu=True  # Force GPU
)
time_zarr_gpu = time.time() - start

print("\n" + "="*60)
print("ZARR x1 (sans pooling)")
print("="*60)
start = time.time()
ds_full = concatenate(
    sst_files,
    var_list=all_sst_vars + COVARIATES,
    slices=slices,
    type_coords="coords",
    resize=1,
    domain_limits=None,
    verbose=True  # Activer les timings détaillés
)
time_io_only = time.time() - start

print("\n" + "="*60)
print("COMPARATIF (15 jours)")
print("="*60)
print(f"NetCDF x10:    {time_nc:.2f}s")
print(f"Zarr x10 CPU:  {time_zarr_cpu:.2f}s")
print(f"Zarr x10 GPU:  {time_zarr_gpu:.2f}s")
print(f"Zarr x1 (sans pooling):  {time_io_only:.2f}s")
