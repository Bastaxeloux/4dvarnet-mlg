from glob import glob
import datetime
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import as_strided
import time as time_module
import os
import re

# Don't import cupy at module level to avoid issues in DataLoader workers
# We'll do lazy import in _fast_pool_gpu instead
HAS_CUPY = True  # Assume available, will check at runtime

# SST Satellites: 4 satellites with average and standard deviation
VAR_GROUPS = {
    "aasti": ["av", "std"],   # Sparse, mainly at poles
    "avhrr": ["av", "std"],   # Good coverage
    "pmw": ["av", "std"],     # Very covering but less precise, smooth
    "slstr": ["av", "std"]    # Sparse, no data at poles
}

# Covariates: only sea ice fraction for now
COVARIATES = ["sea_ice_fraction"]

def organize_by_resolution(file_paths):
    """
    Transform a list of file paths into a dict indexed by resolution.
    Detects resolution from filename pattern: *_x1.zarr, *_x3.zarr, *_x10.zarr
    Example:
        Input:  ['/path/2024010112_x1.zarr', '/path/2024010212_x1.zarr',
                 '/path/2024010112_x3.zarr', '/path/2024010212_x3.zarr']
        Output: {1: [...x1 paths...], 3: [...x3 paths...]}
    """    
    file_paths = np.atleast_1d(file_paths)
    resolution_dict = {}
    has_resolution_suffix = False
    for path in file_paths:
        match = re.search(r'_x(\d+)\.(zarr|nc)$', os.path.basename(path))
        if match:
            has_resolution_suffix = True
            res = int(match.group(1))
            if res not in resolution_dict:
                resolution_dict[res] = []
            resolution_dict[res].append(path)
    if not has_resolution_suffix:
        return file_paths
    for res in resolution_dict:
        resolution_dict[res] = np.sort(resolution_dict[res])
    return resolution_dict

def denormalize_minmax(norm_data, min_val, max_val):
    return norm_data * (max_val - min_val) + min_val

def summarize_lonlat(lon, lat):
    for name, arr in zip(['lon', 'lat'], [lon, lat]):
        arr = np.asarray(arr)
        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        center = arr[arr.shape[0] // 2, arr.shape[1] // 2] if arr.ndim == 2 else arr[len(arr) // 2]
        print(f"{name.upper()}: min={vmin:.4f}, max={vmax:.4f}, center={center:.4f}")

def fast_pool(var, fy, fx, mode="mean", use_gpu=True):
    arr = var.values
    # Disable GPU in DataLoader workers (they can't initialize CUDA)
    # Only use GPU in main process
    import multiprocessing
    in_worker = multiprocessing.current_process().name != 'MainProcess'
    
    if use_gpu and HAS_CUPY and not in_worker:
        return _fast_pool_gpu(arr, fy, fx, mode)
    else:
        return _fast_pool_cpu(arr, fy, fx, mode)

def _fast_pool_cpu(arr, fy, fx, mode="mean"):
    """CPU implementation using numpy."""
    *leading, ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]
        *leading, ny, nx = arr.shape

    out_ny, out_nx = ny // fy, nx // fx

    if mode == "mode":
        # Vectorized mode pooling for categorical data (surfmask: 0-4)
        shape = (*leading, out_ny, fy, out_nx, fx)
        blocks = arr.reshape(shape)
        axes = list(range(len(leading))) + [len(leading), len(leading)+2, len(leading)+1, len(leading)+3]
        blocks = blocks.transpose(axes).reshape(*leading, out_ny, out_nx, fy * fx)

        blocks_int = np.round(blocks).astype(np.int32)
        blocks_int = np.clip(blocks_int, 0, 4)

        counts = np.zeros((*leading, out_ny, out_nx, 5), dtype=np.int32)
        for val in range(5):
            counts[..., val] = np.sum(blocks_int == val, axis=-1)

        return np.argmax(counts, axis=-1).astype(arr.dtype)
    else:
        shape = (*leading, out_ny, fy, out_nx, fx)
        strides = (*arr.strides[:-2], arr.strides[-2]*fy, arr.strides[-2], arr.strides[-1]*fx, arr.strides[-1])
        blocks = as_strided(arr, shape=shape, strides=strides)
        return np.nanmean(blocks, axis=(-1, -3))

def _fast_pool_gpu(arr, fy, fx, mode="mean"):
    """ GPU implementation using cupy - lazy import to avoid worker issues."""
    try:
        import cupy as cp
    except ImportError:
        return _fast_pool_cpu(arr, fy, fx, mode)

    if mode == "mode":
        # Mode pooling on CPU (small data anyway)
        return _fast_pool_cpu(arr, fy, fx, mode)

    *leading, ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]
    arr_gpu = cp.asarray(arr)
    shape = (*leading, ny // fy, fy, nx // fx, fx)
    arr_reshaped = arr_gpu.reshape(shape)
    result = cp.nanmean(arr_reshaped, axis=(-1, -3))
    return cp.asnumpy(result)

def fast_coarsen_xr(ds, factor_y=2, factor_x=2, mode="mean", use_gpu=True):
    """
    SST-specific coarsening: works with (time, lat, lon) dimensions.
    Pools data variables and coarsens lat/lon coordinates (1D).
    """
    out = {}
    for var in ds.data_vars:
        if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
            out[var] = (ds[var].dims, fast_pool(ds[var], factor_y, factor_x, mode=mode, use_gpu=use_gpu))
        else:
            out[var] = ds[var]

    new_coords = {}
    
    # Coarsen 1D lat/lon coordinates
    for coord_name, factor in [('lat', factor_y), ('lon', factor_x)]:
        if coord_name in ds.coords:
            coord_vals = ds.coords[coord_name].values
            coord_vals = coord_vals[: len(coord_vals) - (len(coord_vals) % factor)]
            coord_new = coord_vals.reshape(-1, factor).mean(axis=1)
            new_coords[coord_name] = coord_new
    
    # Keep other coordinates as-is
    for c in ds.coords:
        if c not in ['lat', 'lon']:
            new_coords[c] = ds.coords[c]

    return xr.Dataset(out, coords=new_coords)

def fast_coarsen_xr_array(da, factor_y=2, factor_x=2, dims=('yc', 'xc'), mode="mean"):
    """
    Coarsen a DataArray along two spatial dimensions (e.g., 'yc', 'xc') using pooling.

    Parameters:
    - da: xarray.DataArray
    - factor_y: int, coarsening factor along the first dimension (e.g., 'yc')
    - factor_x: int, coarsening factor along the second dimension (e.g., 'xc')
    - dims: tuple of two str, names of the dimensions to coarsen (e.g., ('yc', 'xc'))
    - mode: str, reduction method ('mean', 'sum', 'max', etc.)

    Returns:
    - xarray.DataArray with coarsened data and updated coordinates
    """

    # Apply pooling
    pooled = fast_pool(da, factor_y, factor_x, mode=mode)

    # Handle coordinates
    new_coords = {}
    for d, factor in zip(dims, [factor_y, factor_x]):
        coord = da.coords[d].values
        coord = coord[: len(coord) - (len(coord) % factor)]
        coord_new = coord.reshape(-1, factor).mean(axis=1)
        new_coords[d] = coord_new

    # Preserve other coordinates
    for c in da.coords:
        if c not in dims and c not in ['lat', 'lon']:
            new_coords[c] = da.coords[c]

    # Optionally coarsen lat/lon
    for var in ['lon', 'lat']:
        if var in da.coords:
            pooled_coord = fast_pool(da.coords[var], factor_y, factor_x)
            new_coords[var] = (dims, pooled_coord)

    return xr.DataArray(
        pooled,
        dims=da.dims,
        coords=new_coords,
        attrs=da.attrs
    )

def load_data(sensor=None, base_dir=None, pattern=None):
    # if pattern is not None:
    #     return sorted(glob(pattern))
    # if sensor is None or base_dir is None:
    #     raise ValueError("Either provide 'pattern' or both 'sensor' and 'base_dir'")
    # return sorted(glob(f"{base_dir}/**/*{sensor}*.nc", recursive=True))
    raise NotImplementedError("load_data() is not Implemented for SST")


def _materialize_array(data, path, var, slices):
    """Compute one lazy array, retrying one transient Blosc read failure."""
    if not hasattr(data, 'compute'):
        return data

    for attempt in range(2):
        try:
            return data.compute()
        except RuntimeError as exc:
            is_blosc_error = "blosc decompression" in str(exc).lower()
            if not is_blosc_error or attempt == 1:
                raise RuntimeError(
                    f"Failed to read variable {var!r} from {path!r} "
                    f"with slices={slices!r}: {exc}"
                ) from exc
            print(
                f"[ZARR READ RETRY] variable={var} path={path} slices={slices}",
                flush=True,
            )

def concatenate(paths, var_list, slices=None, type_coords="index", resize=1, domain_limits=None, verbose=False, use_gpu=True):
    """
    SST-specific: concatenate multiple NetCDF/Zarr files along time dimension.
    Supports spatial slicing and coarsening via resize factor.
    Auto-detects .zarr format.
    """
    t_total = time_module.time()
    t_open = t_select = t_coarsen = t_to_numpy = 0.0
    
    # Premier fichier - détection automatique zarr/netcdf
    t0 = time_module.time()
    if paths[0].endswith('.zarr'):
        ds = xr.open_zarr(paths[0])
    else:
        ds = xr.open_dataset(paths[0])
    t_open += time_module.time() - t0
    
    if domain_limits is not None:
        ds = ds.sel(**domain_limits)
    times = [ds.time[0].data]
    ds = ds[var_list]
    
    t0 = time_module.time()
    if slices is not None:
        if type_coords == "index":
            ds = ds.isel(**slices)
        else:
            ds = ds.sel(**slices)
    t_select += time_module.time() - t0
    
    t0 = time_module.time()
    if resize != 1:
        ds = fast_coarsen_xr(ds, factor_x=resize, factor_y=resize, use_gpu=use_gpu)
    t_coarsen += time_module.time() - t0
    
    t0 = time_module.time()
    ds_vars = {}
    for var in var_list:
        if var in ds:
            data = ds[var].data
            # Convert Dask arrays to numpy
            data = _materialize_array(data, paths[0], var, slices)
            ds_vars[var] = np.squeeze(data)
    t_to_numpy += time_module.time() - t0
    
    coords = ds.coords
    ds.close()
    data_vars = {var: [ds_vars[var]] for var in ds_vars}
    
    # Fichiers suivants
    for path in paths[1:]:
        t0 = time_module.time()
        if path.endswith('.zarr'):
            ds = xr.open_zarr(path)
        else:
            ds = xr.open_dataset(path)
        t_open += time_module.time() - t0
        
        if domain_limits is not None:
            ds = ds.sel(**domain_limits)
        times.append(ds.time[0].data)
        ds = ds[var_list]
        
        t0 = time_module.time()
        if slices is not None:
            if type_coords == "index":
                ds = ds.isel(**slices)
            else:
                ds = ds.sel(**slices)
        t_select += time_module.time() - t0
        
        t0 = time_module.time()
        if resize != 1:
            ds = fast_coarsen_xr(ds, factor_x=resize, factor_y=resize, use_gpu=use_gpu)
        t_coarsen += time_module.time() - t0
        
        t0 = time_module.time()
        for var in var_list:
            if var in ds:
                data = ds[var].data
                # Convert Dask arrays to numpy
                data = _materialize_array(data, path, var, slices)
                data_vars[var].append(np.squeeze(data))
        t_to_numpy += time_module.time() - t0
        
        ds.close()

    # Stack along time
    t0 = time_module.time()
    for var in data_vars:
        data_vars[var] = np.stack(data_vars[var], axis=0)
    t_stack = time_module.time() - t0

    # Create final Dataset (SST uses lat/lon)
    t0 = time_module.time()
    concat = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), data_vars[var]) for var in data_vars},
        coords=dict(
            time=times,
            lat=coords["lat"],
            lon=coords["lon"]
        )
    )
    t_create_ds = time_module.time() - t0
    
    t_total = time_module.time() - t_total
    
    if verbose:
        print(f"\nTimings concatenate() [resize={resize}, {len(paths)} files]:")
        print(f"   Open files:     {t_open:.3f}s ({t_open/t_total*100:.1f}%)")
        print(f"   Select slices:  {t_select:.3f}s ({t_select/t_total*100:.1f}%)")
        print(f"   Coarsen:        {t_coarsen:.3f}s ({t_coarsen/t_total*100:.1f}%)")
        print(f"   To numpy:       {t_to_numpy:.3f}s ({t_to_numpy/t_total*100:.1f}%)")
        print(f"   Stack:          {t_stack:.3f}s ({t_stack/t_total*100:.1f}%)")
        print(f"   Create Dataset: {t_create_ds:.3f}s ({t_create_ds/t_total*100:.1f}%)")
        print(f"   TOTAL:          {t_total:.3f}s")

    return concat

def load_mfdata(asip_paths, cimr_paths, cristal_paths,
                covariates_paths, covariates,
                times, slices=None, type_coords="index",resize=1):
    # def select_paths_from_dates(files, times, fmt="%Y%m%d"):
    #     if isinstance(times, list):
    #         dates = []
    #         for t in times:
    #             start = datetime.datetime.strptime(t.start, "%Y-%m-%d")
    #             end = datetime.datetime.strptime(t.stop, "%Y-%m-%d")
    #             dates.extend([(start + datetime.timedelta(days=x)).strftime(fmt) for x in range((end-start).days)])
    #     else:
    #         start = datetime.datetime.strptime(times.start, "%Y-%m-%d")
    #         end = datetime.datetime.strptime(times.stop, "%Y-%m-%d")
    #         dates = [(start + datetime.timedelta(days=x)).strftime(fmt) for x in range((end-start).days)]
    #     return np.sort([f for f in files if any(s in f for s in dates)])

    # sel_asip = select_paths_from_dates(asip_paths, times)
    # sel_cimr = select_paths_from_dates(cimr_paths, times, fmt="%Y-%m-%d")
    # sel_cristal = select_paths_from_dates(cristal_paths, times, fmt="%Y-%m-%d")
    # sel_covariates = select_paths_from_dates(covariates_paths, times, fmt="%Y-%m-%d")

    # asip = concatenate(sel_asip, VAR_GROUPS["asip"], slices, type_coords, resize=resize)
    # cimr = concatenate(sel_cimr, VAR_GROUPS["cimr"], None, type_coords)
    # cristal = concatenate(sel_cristal, VAR_GROUPS["cristal"], None, type_coords)
    # covs = concatenate(sel_covariates, covariates, None, type_coords)

    # return asip, cimr, cristal, covs
    raise NotImplementedError("load_mfdata() is a CROSCIM function. Not Implemented for SST")
