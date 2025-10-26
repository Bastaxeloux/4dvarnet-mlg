#!/usr/bin/env python3
"""
Precompute multi-resolution SST data for faster training.

Generates x3 and x10 from x1 (native resolution).
Uses xarray.to_zarr() - clean and simple, no warnings.

Usage:
    python precompute_resolutions.py /path/to/file_x1.nc
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import as_strided

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def fast_pool_cpu(arr, fy, fx):
    """Fast CPU pooling using stride tricks."""
    *leading, ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]
    shape = (*leading, ny // fy, fy, nx // fx, fx)
    strides = (*arr.strides[:-2], arr.strides[-2]*fy, arr.strides[-2], arr.strides[-1]*fx, arr.strides[-1])
    blocks = as_strided(arr, shape=shape, strides=strides)
    return np.nanmean(blocks, axis=(-1, -3))


def fast_pool_gpu(arr, fy, fx):
    """Fast GPU pooling using CuPy."""
    *leading, ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]
    arr_gpu = cp.asarray(arr)
    shape = (*leading, ny // fy, fy, nx // fx, fx)
    arr_reshaped = arr_gpu.reshape(shape)
    result = cp.nanmean(arr_reshaped, axis=(-1, -3))
    return cp.asnumpy(result)


def fast_pool(var, fy, fx, use_gpu=True):
    """Pool array by averaging over blocks."""
    arr = var.values if hasattr(var, 'values') else var
    if use_gpu and HAS_CUPY:
        return fast_pool_gpu(arr, fy, fx)
    else:
        return fast_pool_cpu(arr, fy, fx)


def pool_dataset(ds, factor, use_gpu=True, verbose=False):
    """
    Pool all variables in a dataset by a given factor.
    Returns a new xarray Dataset with pooled data and coordinates.
    """
    lat_vals = ds.coords['lat'].values
    lon_vals = ds.coords['lon'].values

    # Pool coordinates
    lat_pooled = lat_vals[: len(lat_vals) - (len(lat_vals) % factor)].reshape(-1, factor).mean(axis=1)
    lon_pooled = lon_vals[: len(lon_vals) - (len(lon_vals) % factor)].reshape(-1, factor).mean(axis=1)

    # Pool data variables
    data_vars = {}
    for var in ds.data_vars:
        if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
            if verbose:
                print(f"  Pooling {var}...", end=' ', flush=True)
            t0 = time.time()
            pooled = fast_pool(ds[var], fy=factor, fx=factor, use_gpu=use_gpu)
            data_vars[var] = (ds[var].dims, pooled)
            if verbose:
                print(f"{time.time()-t0:.2f}s")
        else:
            data_vars[var] = ds[var]

    coords = {
        'lat': lat_pooled,
        'lon': lon_pooled,
        'time': ds.coords['time']
    }

    return xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)


def precompute_resolutions(input_path, output_dir=None, use_gpu=True, verbose=True):
    """
    Precompute x3 and x10 from x1 (native resolution).
    Clean implementation using xarray.to_zarr() - no warnings.
    """
    t_start = time.time()
    input_path = Path(input_path)
    basename = input_path.stem.replace('_x1', '')
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        'x3': output_dir / f"{basename}_x3.zarr",
        'x10': output_dir / f"{basename}_x10.zarr"
    }

    if verbose:
        print(f"Processing: {input_path.name}")
        print(f"  x3  => {output_paths['x3'].name}")
        print(f"  x10 => {output_paths['x10'].name}")

    t0 = time.time()
    if str(input_path).endswith('.zarr'):
        ds = xr.open_zarr(input_path)
    else:
        ds = xr.open_dataset(input_path)

    if verbose:
        print(f"\nLoaded in {time.time()-t0:.2f}s ({ds.dims})")
    for path in output_paths.values():
        if path.exists():
            import shutil
            shutil.rmtree(path)

    if verbose:
        print("\n[1/2] x3 (pooling by 3)...")
    t0 = time.time()
    ds_x3 = pool_dataset(ds, factor=3, use_gpu=use_gpu, verbose=verbose)
    encoding = {
        var: {'chunks': (256, 256)}
        for var in ds_x3.data_vars if 'lat' in ds_x3[var].dims
    }
    ds_x3.to_zarr(output_paths['x3'], mode='w', encoding=encoding, consolidated=True)
    if verbose:
        size_mb = sum(f.stat().st_size for f in output_paths['x3'].rglob('*') if f.is_file()) / 1e6
        print(f"x3 done in {time.time()-t0:.2f}s ({size_mb:.1f} MB)")

    # --- Resolution x10 ---
    if verbose:
        print("\n[2/2] x10 (pooling by 10)...")
    t0 = time.time()
    ds_x10 = pool_dataset(ds, factor=10, use_gpu=use_gpu, verbose=verbose)
    encoding = {
        var: {'chunks': (256, 256)}
        for var in ds_x10.data_vars if 'lat' in ds_x10[var].dims
    }
    ds_x10.to_zarr(output_paths['x10'], mode='w', encoding=encoding, consolidated=True)
    if verbose:
        size_mb = sum(f.stat().st_size for f in output_paths['x10'].rglob('*') if f.is_file()) / 1e6
        print(f"x10 done in {time.time()-t0:.2f}s ({size_mb:.1f} MB)")

    ds.close()

    total_time = time.time() - t_start
    total_size = sum(
        sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        for path in output_paths.values()
    ) / 1e9
    if verbose:
        print(f"Complete in {total_time:.2f}s ({total_size:.2f} GB)")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Precompute x3 and x10 from x1")
    parser.add_argument('input', type=str, help='Input NetCDF or Zarr file (x1)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for pooling (requires CuPy)')
    parser.add_argument('-q', '--quiet', action='store_true', default=False,
                        help='Quiet mode')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.use_gpu and not HAS_CUPY:
        print("Warning: --use-gpu specified but CuPy not available. Using CPU.", file=sys.stderr)
        args.use_gpu = False

    try:
        precompute_resolutions(
            input_path=args.input,
            output_dir=args.output,
            use_gpu=args.use_gpu,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
