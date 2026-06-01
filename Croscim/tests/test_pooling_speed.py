import numpy as np
import time
import glob
import xarray as xr
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Avoid import errors - define locally
VAR_GROUPS = {
    "aasti": ["av", "std"],
    "avhrr": ["av", "std"],
    "pmw": ["av", "std"],
    "slstr": ["av", "std"]
}
COVARIATES = ["sea_ice_fraction"]

# Import concatenate function only
try:
    from contrib.SST.load_data import concatenate
except ImportError:
    print("Warning: Could not import concatenate, will use manual xarray loading")
    def concatenate(paths, var_list, slices=None, **kwargs):
        """Simple fallback concatenate"""
        datasets = []
        for p in paths:
            ds = xr.open_zarr(p)
            if slices:
                ds = ds.sel(**slices)
            ds = ds[var_list]
            datasets.append(ds)
        result = xr.concat(datasets, dim='time')
        for ds in datasets:
            ds.close()
        return result


# Verfier GPU disponibles avec nvtop ou nvidia-smi
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
MOUNT_DIR = "/nwp/sst_malegu/data_2024"

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

# Prendre un patch 256×256 comme dans le vrai training
patch_size = 256
lat_end_idx = min(patch_size, len(lat_vals)-1)
lon_end_idx = min(patch_size, len(lon_vals)-1)

lat_start, lat_end = lat_vals[0], lat_vals[lat_end_idx]
lon_start, lon_end = lon_vals[0], lon_vals[lon_end_idx]

ds_x1_sample.close()

slices = {
    "lon": slice(lon_start, lon_end),
    "lat": slice(lat_start, lat_end)
}
all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]

print(f"\nPatch 256×256 (comme dans le training)")
print(f"Région: lat=[{lat_start:.2f}, {lat_end:.2f}], lon=[{lon_start:.2f}, {lon_end:.2f}]")
print(f"Variables à charger: {len(all_sst_vars + COVARIATES)}")
print(f"x1: {lat_end_idx}×{lon_end_idx}, x3: {lat_end_idx}×{lon_end_idx}, x10: {lat_end_idx}×{lon_end_idx} (precomputed=True)")

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

# # lets add the same test for netcdf

# netcdf_path = "/nwp/sst_malegu/data_2024"
# sst_files_nc = sorted(glob.glob(f"{netcdf_path}/*_x1.nc"))[:15]
# if len(sst_files_nc) == 0:
#     print("No NetCDF files found, skipping NetCDF pooling test.")
#     sys.exit(0)
    
# print("\n" + "="*60)
# print("NetCDF x1 direct (haute résolution, sans pooling)")
# print("="*60)
# start = time.time()
# ds_nc_x1_direct = concatenate(
#     sst_files_nc,
#     var_list=all_sst_vars + COVARIATES,
#     slices=slices,
#     type_coords="coords",
#     resize=1,
#     domain_limits=None,
#     verbose=True
# )
# time_nc_x1_direct = time.time() - start
    
# print("\n" + "="*60)
# print("NetCDF x1 + pooling GPU x10")
# print("="*60)
# start = time.time()
# ds_nc_x1_pool_gpu = concatenate(
#     sst_files_nc,
#     var_list=all_sst_vars + COVARIATES,
#     slices=slices,
#     type_coords="coords",
#     resize=10,  # Pooling via GPU
#     domain_limits=None,
#     verbose=True,
#     use_gpu=True
# )
# time_nc_x1_pool_gpu = time.time() - start   

# print("\n" + "="*60)
# print("NetCDF x1 + pooling CPU x10")
# print("="*60)
# start = time.time()
# ds_nc_x1_pool_cpu = concatenate(
#     sst_files_nc,
#     var_list=all_sst_vars + COVARIATES,
#     slices=slices,
#     type_coords="coords",
#     resize=10,  # Pooling via CPU
#     domain_limits=None,
#     verbose=True,
#     use_gpu=False
# )
# time_nc_x1_pool_cpu = time.time() - start   

print("\n" + "="*60)
print("RÉSUMÉ DES TEMPS DE CHARGEMENT")
print("="*60)
print(f"\nZarr x1 direct (sans pooling)       : {time_x1_direct:.2f} s")
# print(f"NetCDF x1 direct (sans pooling)     : {time_nc_x1_direct:.2f} s")
print("\n" + "-"*60)
print(f"\nZarr x3 direct (sans pooling)       : {time_x3_direct:.2f} s")
print(f"Zarr x10 direct (sans pooling)      : {time_x10_direct:.2f} s")
print("\n" + "-"*60)
print("Be carefull because I think these pooling test doesnt include the reading time from disk since it has already been done")
print(f"\nZarr x1 + pooling GPU x10           : {time_x1_pool_gpu:.2f} s")
print(f"Zarr x1 + pooling CPU x10           : {time_x1_pool_cpu:.2f} s")
# print("\n" + "-"*60)
# print(f"\nNetCDF x1 + pooling GPU x10         : {time_nc_x1_pool_gpu:.2f} s")
# print(f"NetCDF x1 + pooling CPU x10         : {time_nc_x1_pool_cpu:.2f} s")