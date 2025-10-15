import xarray as xr
import numpy as np

file_path = "/dmidata/users/malegu/data/netcdf_2024/2024010112_13vars.nc"

print("=" * 60)
print("Verification fichier NetCDF")
print("=" * 60)

ds = xr.open_dataset(file_path)

print("\nVariables:")
for var in ds.data_vars:
    data = ds[var].values
    print(f"\n{var}:")
    print(f"  shape: {data.shape}")
    print(f"  dtype: {data.dtype}")
    print(f"  min: {np.nanmin(data):.3f}")
    print(f"  max: {np.nanmax(data):.3f}")
    print(f"  mean: {np.nanmean(data):.3f}")
    print(f"  NaN count: {np.isnan(data).sum()} / {data.size} ({100*np.isnan(data).sum()/data.size:.1f}%)")
    print(f"  Valid count: {(~np.isnan(data)).sum()} / {data.size} ({100*(~np.isnan(data)).sum()/data.size:.1f}%)")

print("\nCoordonnees:")
for coord in ds.coords:
    if coord not in ds.dims:
        print(f"  {coord}: {ds[coord].values}")

# Vérifier un petit patch
print("\n" + "=" * 60)
print("Verification patch [0:256, 0:256]")
print("=" * 60)

for var in ['aasti_av', 'avhrr_av', 'pmw_av', 'slstr_av']:
    if var in ds:
        patch = ds[var].isel(lat=slice(0, 256), lon=slice(0, 256)).values
        valid = (~np.isnan(patch)).sum()
        print(f"{var:15s}: valid pixels = {valid}/{patch.size} ({100*valid/patch.size:.1f}%)")
