import xarray as xr
zarr_path = '/dmidata/users/malegu/data/netcdf_2024/2024010112_x1.zarr'
ds = xr.open_zarr(zarr_path)
print(f"  lat: {ds.lat.shape} -> chunks: {ds.lat.chunks}")
print(f"  lon: {ds.lon.shape} -> chunks: {ds.lon.chunks}")
print(f"  time: {ds.time.shape} -> chunks: {ds.time.chunks}")
for var in ['aasti_av', 'avhrr_av', 'analysed_st', 'surfmask']:
    if var in ds:
        print(f"  {var}: {ds[var].shape} -> chunks: {ds[var].chunks}")