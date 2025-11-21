#!/usr/bin/env python3
"""
Test pour diagnostiquer l'usage mémoire lors du chargement de patches
"""
import xarray as xr
import numpy as np
import psutil
import os
import sys

def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_zarr_loading():
    """Test memory usage when loading zarr patches"""

    zarr_path = "/nwp/sst_malegu/data_2024/2024010112_x1.zarr"

    print(f"Testing zarr loading from: {zarr_path}")
    print(f"Initial memory: {get_memory_mb():.1f} MB\n")

    # Step 1: Open dataset (should be lazy)
    mem_before_open = get_memory_mb()
    ds = xr.open_zarr(zarr_path)
    mem_after_open = get_memory_mb()
    print(f"1. After xr.open_zarr():")
    print(f"   Memory delta: {mem_after_open - mem_before_open:.1f} MB")
    print(f"   Dataset size: {ds.dims}")
    print(f"   Data type: {type(ds['aasti_av'].data)}")  # Should be dask.array
    print()

    # Step 2: Select a small spatial patch (256x256)
    mem_before_select = get_memory_mb()
    ds_patch = ds.isel(lat=slice(0, 256), lon=slice(0, 256))
    mem_after_select = get_memory_mb()
    print(f"2. After .isel(lat=slice(0,256), lon=slice(0,256)):")
    print(f"   Memory delta: {mem_after_select - mem_before_select:.1f} MB")
    print(f"   Patch size: {ds_patch.dims}")
    print(f"   Data type: {type(ds_patch['aasti_av'].data)}")
    print()

    # Step 3: Compute one variable
    mem_before_compute = get_memory_mb()
    var_data = ds_patch['aasti_av'].data
    if hasattr(var_data, 'compute'):
        var_data = var_data.compute()
    mem_after_compute = get_memory_mb()

    expected_mb = var_data.nbytes / 1024 / 1024
    actual_mb = mem_after_compute - mem_before_compute

    print(f"3. After .compute() on 'aasti_av':")
    print(f"   Variable shape: {var_data.shape}")
    print(f"   Expected size: {expected_mb:.1f} MB")
    print(f"   Actual memory delta: {actual_mb:.1f} MB")
    print(f"   Ratio (actual/expected): {actual_mb/expected_mb:.2f}x")
    print()

    # Step 4: Compute ALL variables in patch
    mem_before_all = get_memory_mb()
    all_vars = {}
    for var in ['aasti_av', 'aasti_std', 'avhrr_av', 'avhrr_std',
                'pmw_av', 'pmw_std', 'slstr_av', 'slstr_std', 'sea_ice_fraction']:
        if var in ds_patch:
            data = ds_patch[var].data
            if hasattr(data, 'compute'):
                data = data.compute()
            all_vars[var] = data

    mem_after_all = get_memory_mb()

    total_expected_mb = sum(v.nbytes for v in all_vars.values()) / 1024 / 1024
    total_actual_mb = mem_after_all - mem_before_all

    print(f"4. After computing ALL 9 variables:")
    print(f"   Expected size: {total_expected_mb:.1f} MB")
    print(f"   Actual memory delta: {total_actual_mb:.1f} MB")
    print(f"   Ratio (actual/expected): {total_actual_mb/total_expected_mb:.2f}x")
    print()

    # Step 5: Close dataset
    ds.close()
    mem_after_close = get_memory_mb()
    print(f"5. After ds.close():")
    print(f"   Memory after close: {mem_after_close:.1f} MB")
    print(f"   Memory still held: {mem_after_close - mem_before_open:.1f} MB")
    print()

    # Step 6: Delete variables and force GC
    del all_vars
    del ds_patch
    del ds
    import gc
    gc.collect()
    mem_after_gc = get_memory_mb()
    print(f"6. After del + gc.collect():")
    print(f"   Memory after GC: {mem_after_gc:.1f} MB")
    print(f"   Memory leaked: {mem_after_gc - mem_before_open:.1f} MB")
    print()

def test_concatenate_15_days():
    """Test memory when loading 15 days like in real training"""
    import glob

    # Get 15 files from January 2024
    all_files = sorted(glob.glob("/nwp/sst_malegu/data_2024/2024*.zarr"))
    paths = [f for f in all_files if f.endswith("_x1.zarr")][:15]

    print(f"\n{'='*70}")
    print(f"TEST 2: Loading 15 timesteps (realistic training case)")
    print(f"{'='*70}")
    print(f"Testing concatenate with {len(paths)} files")
    print(f"Initial memory: {get_memory_mb():.1f} MB\n")

    # Simulate what concatenate() does
    mem_before = get_memory_mb()

    all_data = []
    slices = {"lat": slice(0, 256), "lon": slice(0, 256)}
    var_list = ['aasti_av', 'aasti_std', 'avhrr_av', 'avhrr_std',
                'pmw_av', 'pmw_std', 'slstr_av', 'slstr_std', 'sea_ice_fraction']

    for i, path in enumerate(paths):
        ds = xr.open_zarr(path)
        ds = ds[var_list]
        ds = ds.isel(**slices)

        # Compute all variables
        data_dict = {}
        for var in var_list:
            if var in ds:
                data = ds[var].data
                if hasattr(data, 'compute'):
                    data = data.compute()
                data_dict[var] = np.squeeze(data)

        all_data.append(data_dict)
        ds.close()

        if (i + 1) % 5 == 0:
            mem_current = get_memory_mb()
            print(f"After {i+1} files: {mem_current:.1f} MB (delta: {mem_current - mem_before:.1f} MB)")

    mem_after_load = get_memory_mb()

    # Stack along time dimension
    stacked = {}
    for var in var_list:
        stacked[var] = np.stack([d[var] for d in all_data], axis=0)

    mem_after_stack = get_memory_mb()

    # Calculate expected size
    expected_mb = sum(v.nbytes for v in stacked.values()) / 1024 / 1024

    print(f"\nFinal results:")
    print(f"  Shape per variable: {stacked['aasti_av'].shape}")
    print(f"  Expected data size: {expected_mb:.1f} MB")
    print(f"  Actual memory used: {mem_after_stack - mem_before:.1f} MB")
    print(f"  Overhead ratio: {(mem_after_stack - mem_before) / expected_mb:.2f}x")
    print()

    # Cleanup
    del all_data
    del stacked
    import gc
    gc.collect()
    mem_after_gc = get_memory_mb()
    print(f"After cleanup: {mem_after_gc:.1f} MB")
    print(f"Memory leaked: {mem_after_gc - mem_before:.1f} MB")

if __name__ == "__main__":
    test_zarr_loading()
    test_concatenate_15_days()
