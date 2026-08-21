#!/usr/bin/env python3
"""
Precompute multi-resolution SST data for faster training.

Generates x3 and x10 from x1 (native resolution).
Uses xarray.to_zarr() - clean and simple, no warnings.

Usage:
    python precompute_resolutions.py /path/to/file_x1.nc
"""

import argparse
import os
import shutil
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


def remove_path(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def write_zarr_atomic(ds, target, encoding):
    target = Path(target)
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    remove_path(temp)
    try:
        ds.to_zarr(temp, mode='w', encoding=encoding, consolidated=True)
        remove_path(target)
        temp.rename(target)
    except BaseException:
        remove_path(temp)
        raise


def fast_pool_cpu(arr, fy, fx, method='mean'):
    """Fast CPU pooling using stride tricks.

    Args:
        arr: Input array
        fy, fx: Pooling factors
        method: 'mean' for continuous data, 'mode' for categorical data (e.g., masks)
    """
    *leading, ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[..., :ny - (ny % fy), :nx - (nx % fx)]
        *leading, ny, nx = arr.shape

    out_ny, out_nx = ny // fy, nx // fx

    if method == 'mode':
        # Vectorized mode pooling for categorical data (surfmask: 0-4)
        # Reshape to (*leading, out_ny, fy, out_nx, fx) then to (*leading, out_ny, out_nx, fy*fx)
        shape = (*leading, out_ny, fy, out_nx, fx)
        blocks = arr.reshape(shape)
        # Transpose to group spatial blocks together
        axes = list(range(len(leading))) + [len(leading), len(leading)+2, len(leading)+1, len(leading)+3]
        blocks = blocks.transpose(axes).reshape(*leading, out_ny, out_nx, fy * fx)

        # Round to integers and clamp to valid range
        blocks_int = np.round(blocks).astype(np.int32)
        blocks_int = np.clip(blocks_int, 0, 4)

        # Count occurrences of each value (0-4) - vectorized
        counts = np.zeros((*leading, out_ny, out_nx, 5), dtype=np.int32)
        for val in range(5):
            counts[..., val] = np.sum(blocks_int == val, axis=-1)

        # Mode = value with max count
        result = np.argmax(counts, axis=-1).astype(arr.dtype)
        return result
    else:
        shape = (*leading, out_ny, fy, out_nx, fx)
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


def fast_pool(var, fy, fx, use_gpu=True, method='mean'):
    """Pool array by averaging (for continuous) or mode (for categorical) over blocks."""
    arr = var.values if hasattr(var, 'values') else var
    if method == 'mode':
        # Mode pooling always on CPU (categorical data is small anyway)
        return fast_pool_cpu(arr, fy, fx, method='mode')
    elif use_gpu and HAS_CUPY:
        return fast_pool_gpu(arr, fy, fx)
    else:
        return fast_pool_cpu(arr, fy, fx, method='mean')


def pool_dataset(ds, factor, use_gpu=True, verbose=False):
    """
    Pool all variables in a dataset by a given factor.
    Returns a new xarray Dataset with pooled data and coordinates.

    Uses:
    - Mode pooling for categorical variables (surfmask)
    - Mean pooling for continuous variables (SST, etc.)
    """
    # List of categorical variables that should use mode pooling
    CATEGORICAL_VARS = {'surfmask', 'mask', 'land_mask', 'sea_mask'}

    lat_vals = ds.coords['lat'].values
    lon_vals = ds.coords['lon'].values

    # Pool coordinates
    lat_pooled = lat_vals[: len(lat_vals) - (len(lat_vals) % factor)].reshape(-1, factor).mean(axis=1)
    lon_pooled = lon_vals[: len(lon_vals) - (len(lon_vals) % factor)].reshape(-1, factor).mean(axis=1)

    # Pool data variables
    data_vars = {}
    for var in ds.data_vars:
        if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
            # Determine pooling method based on variable name
            if var.lower() in CATEGORICAL_VARS:
                method = 'mode'
                method_str = '(mode)'
            else:
                method = 'mean'
                method_str = '(mean)'

            if verbose:
                print(f"  Pooling {var} {method_str}...", end=' ', flush=True)
            t0 = time.time()
            pooled = fast_pool(ds[var], fy=factor, fx=factor, use_gpu=use_gpu, method=method)
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


def precompute_resolutions(input_path, output_dir=None, use_gpu=True, verbose=True, save_format='netcdf', chunk_size=768, compression_level=4):
    """
    Precompute x3 and x10 from x1 (native resolution).

    Args:
        save_format: 'netcdf', 'zarr' ou 'both' (défaut: 'netcdf')
        chunk_size: taille des chunks spatiaux
        compression_level: niveau compression NetCDF 1-9 (défaut: 4)
    """
    t_start = time.time()
    input_path = Path(input_path)
    basename = input_path.stem.replace('_x1', '')
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Chemins de sortie selon le format
    output_paths = {}
    if save_format in ('netcdf', 'both'):
        output_paths['x3_nc'] = output_dir / f"{basename}_x3.nc"
        output_paths['x10_nc'] = output_dir / f"{basename}_x10.nc"
    if save_format in ('zarr', 'both'):
        output_paths['x3_zarr'] = output_dir / f"{basename}_x3.zarr"
        output_paths['x10_zarr'] = output_dir / f"{basename}_x10.zarr"

    if verbose:
        print(f"Processing: {input_path.name}")
        for key, path in output_paths.items():
            print(f"  {key} => {path.name}")

    t0 = time.time()
    if str(input_path).endswith('.zarr'):
        ds = xr.open_zarr(input_path)
    else:
        ds = xr.open_dataset(input_path)

    if verbose:
        print(f"\nLoaded in {time.time()-t0:.2f}s ({ds.dims})")

    # Nettoyer les fichiers existants
    for path in output_paths.values():
        remove_path(path)

    # --- Resolution x3 ---
    if verbose:
        print("\n[1/2] x3 (pooling by 3)...")
    t0 = time.time()
    ds_x3 = pool_dataset(ds, factor=3, use_gpu=use_gpu, verbose=verbose)

    # Adapter chunk_size aux dimensions du dataset (pour NetCDF et Zarr)
    ny_x3, nx_x3 = ds_x3.dims['lat'], ds_x3.dims['lon']
    chunk_lat_x3 = min(chunk_size, ny_x3)
    chunk_lon_x3 = min(chunk_size, nx_x3)

    # Encoder pour NetCDF avec chunks optimisés
    if save_format in ('netcdf', 'both'):
        encoding_nc = {}
        for var in ds_x3.data_vars:
            if 'lat' in ds_x3[var].dims and 'lon' in ds_x3[var].dims:
                encoding_nc[var] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': (chunk_lat_x3, chunk_lon_x3)
                }
            else:
                encoding_nc[var] = {'zlib': True, 'complevel': compression_level}
        ds_x3.to_netcdf(output_paths['x3_nc'], format='NETCDF4', encoding=encoding_nc)
        if verbose:
            size_mb = output_paths['x3_nc'].stat().st_size / 1e6
            print(f"x3 NetCDF done in {time.time()-t0:.2f}s ({size_mb:.1f} MB)")

    if save_format in ('zarr', 'both'):
        encoding_zarr = {var: {'chunks': (chunk_lat_x3, chunk_lon_x3)} for var in ds_x3.data_vars if 'lat' in ds_x3[var].dims}
        write_zarr_atomic(ds_x3, output_paths['x3_zarr'], encoding_zarr)

    # --- Resolution x10 ---
    if verbose:
        print("\n[2/2] x10 (pooling by 10)...")
    t0 = time.time()
    ds_x10 = pool_dataset(ds, factor=10, use_gpu=use_gpu, verbose=verbose)

    # Adapter chunk_size aux dimensions du dataset (pour NetCDF et Zarr)
    ny_x10, nx_x10 = ds_x10.dims['lat'], ds_x10.dims['lon']
    chunk_lat_x10 = min(chunk_size, ny_x10)
    chunk_lon_x10 = min(chunk_size, nx_x10)

    if save_format in ('netcdf', 'both'):
        encoding_nc = {}
        for var in ds_x10.data_vars:
            if 'lat' in ds_x10[var].dims and 'lon' in ds_x10[var].dims:
                encoding_nc[var] = {
                    'zlib': True,
                    'complevel': compression_level,
                    'shuffle': True,
                    'chunksizes': (chunk_lat_x10, chunk_lon_x10)
                }
            else:
                encoding_nc[var] = {'zlib': True, 'complevel': compression_level}
        ds_x10.to_netcdf(output_paths['x10_nc'], format='NETCDF4', encoding=encoding_nc)
        if verbose:
            size_mb = output_paths['x10_nc'].stat().st_size / 1e6
            print(f"x10 NetCDF done in {time.time()-t0:.2f}s ({size_mb:.1f} MB)")

    if save_format in ('zarr', 'both'):
        encoding_zarr = {var: {'chunks': (chunk_lat_x10, chunk_lon_x10)} for var in ds_x10.data_vars if 'lat' in ds_x10[var].dims}
        write_zarr_atomic(ds_x10, output_paths['x10_zarr'], encoding_zarr)

    ds.close()

    total_time = time.time() - t_start
    total_size = sum(path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) for path in output_paths.values()) / 1e9
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
    parser.add_argument('--format', type=str, default='zarr', choices=['netcdf', 'zarr', 'both'],
                        help='Output format (default: zarr)')

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
            verbose=not args.quiet,
            save_format=args.format
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
