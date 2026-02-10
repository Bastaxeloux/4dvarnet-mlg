#!/usr/bin/env python3
"""
Simulate what happens in a persistent worker loading multiple samples
"""
import xarray as xr
import numpy as np
import psutil
import os
import glob

def simple_concatenate(paths, var_list, slices=None):
    """Simplified version of concatenate() for testing"""
    all_data = []

    for path in paths:
        ds = xr.open_zarr(path)
        ds = ds[var_list]
        if slices is not None:
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

    # Stack along time dimension
    stacked = {}
    for var in var_list:
        stacked[var] = np.stack([d[var] for d in all_data], axis=0)

    # Create xarray Dataset
    times = list(range(len(paths)))
    result = xr.Dataset(
        data_vars={var: (("time", "lat", "lon"), stacked[var]) for var in stacked},
        coords=dict(time=times, lat=range(stacked[var_list[0]].shape[1]), lon=range(stacked[var_list[0]].shape[2]))
    )

    return result

def get_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def simulate_worker_loading():
    """Simulate a worker loading multiple samples like in training"""

    # Get all x1 files
    all_files = sorted(glob.glob("/nwp/sst_malegu/data_2024/2024*.zarr"))
    x1_paths = [f for f in all_files if f.endswith("_x1.zarr")]

    print(f"Simulating persistent worker loading samples")
    print(f"Initial memory: {get_memory_mb():.1f} MB\n")

    var_list = ['aasti_av', 'aasti_std', 'avhrr_av', 'avhrr_std',
                'pmw_av', 'pmw_std', 'slstr_av', 'slstr_std', 'sea_ice_fraction']

    # Simulate loading 20 samples (like a worker would do)
    for sample_idx in range(20):
        mem_before_sample = get_memory_mb()

        # Choose 15 consecutive files (sliding window)
        start_idx = sample_idx % (len(x1_paths) - 15)
        paths_for_sample = x1_paths[start_idx:start_idx + 15]

        # Load x1 resolution (256x256 patch)
        slices_x1 = {"lat": slice(100, 356), "lon": slice(100, 356)}
        result_x1 = simple_concatenate(
            paths_for_sample,
            var_list=var_list,
            slices=slices_x1
        )

        # Simulate some processing (like the model would do)
        for var in var_list:
            if var in result_x1:
                _ = result_x1[var].values.mean()

        # Now load x3 resolution for the same sample
        slices_x3 = {"lat": slice(33, 118), "lon": slice(33, 118)}

        # Get x3 paths
        x3_paths_for_sample = [p.replace("_x1.zarr", "_x3.zarr") for p in paths_for_sample]
        result_x3 = concatenate(
            x3_paths_for_sample,
            var_list=var_list,
            slices=slices_x3,
            type_coords="index",
            resize=1,
            verbose=False
        )

        # Load x10 resolution
        slices_x10 = {"lat": slice(10, 35), "lon": slice(10, 35)}
        x10_paths_for_sample = [p.replace("_x1.zarr", "_x10.zarr") for p in paths_for_sample]
        result_x10 = concatenate(
            x10_paths_for_sample,
            var_list=var_list,
            slices=slices_x10,
            type_coords="index",
            resize=1,
            verbose=False
        )

        mem_after_sample = get_memory_mb()
        mem_delta = mem_after_sample - mem_before_sample

        # Try to free memory
        del result_x1
        del result_x3
        del result_x10

        mem_after_del = get_memory_mb()
        mem_freed = mem_after_sample - mem_after_del

        if sample_idx % 5 == 0:
            print(f"Sample {sample_idx+1:2d}: "
                  f"Before={mem_before_sample:6.1f}MB | "
                  f"After={mem_after_sample:6.1f}MB | "
                  f"Delta={mem_delta:5.1f}MB | "
                  f"Freed={mem_freed:5.1f}MB | "
                  f"Net={mem_after_del - mem_before_sample:5.1f}MB")

    print(f"\nFinal memory: {get_memory_mb():.1f} MB")
    print(f"Total leaked: {get_memory_mb() - 137:.1f} MB (from ~137 MB baseline)")

if __name__ == "__main__":
    simulate_worker_loading()
