# Tests:
# 1. organize_by_resolution() function
# 2. XrDataset with dict format (multi-resolution files)
# 3. XrDatasetMultiResTrain with precomputed=True

import sys
sys.path.insert(0, '.')
import glob
import numpy as np
import pandas as pd
import time
from contrib.SST.load_data import organize_by_resolution
from contrib.SST.data import XrDataset
from contrib.SST.data_multires import XrDatasetMultiResTrain
import traceback


def test_organize_by_resolution():
    print("TEST 1: organize_by_resolution()")
    files = sorted(glob.glob("/home/malegu/4D-MLG/Croscim/data/mounted/2024/*_x*.zarr"))[:12]
    print(f"Input: {len(files)} Zarr files")
    for f in files[:3]:
        print(f"  {f.split('/')[-1]}")
    print(f"  ...")

    result = organize_by_resolution(files)
    
    assert isinstance(result, dict), "Result should be a dict"
    assert sorted(result.keys()) == [1, 3, 10], "Should have resolutions 1, 3, 10"
    
    print(f"Ok : Output: dict with {len(result)} resolutions")
    for res in sorted(result.keys()):
        print(f"  x{res}: {len(result[res])} files")
    
    return result


def test_xrdataset_with_dict():
    print("\nTEST 2: XrDataset with dict format")
    files = sorted(glob.glob("/home/malegu/4D-MLG/Croscim/data/mounted/2024/*_x*.zarr"))[:15]
    files_dict = organize_by_resolution(files)
    
    times_list = []
    for f in sorted(files_dict[1]):
        basename = f.split('/')[-1]
        date_str = basename[:10]
        times_list.append(pd.to_datetime(date_str, format="%Y%m%d%H"))
    times = np.array(times_list)
    
    t0 = time.time()
    ds = XrDataset(
        sst_daily_paths=files_dict,  # Pass as dict !
        tgt_vars=["slstr_av"],
        mask=None,
        times=times,
        patch_dims={'time': 3, 'lat': 256, 'lon': 256},
        strides={'time': 1, 'lat': 1, 'lon': 1},
        resize=1,
        verbose=False
    )
    t_init = time.time() - t0
    
    assert ds.is_multiresolution, "Should detect multi-resolution"
    assert ds.sst_daily_paths_by_resolution is not None, "Should store dict"
    
    print(f"Ok : XrDataset created in {t_init:.3f}s")
    print(f"  is_multiresolution: {ds.is_multiresolution}")
    print(f"  resize: {ds.resize}")
    print(f"  Data dims: {ds.da_dims}")
    
    return ds, files_dict


def test_xrdataset_multires_precomputed():
    print("\nTEST 3: XrDatasetMultiResTrain with precomputed=True")
    files = sorted(glob.glob("/home/malegu/4D-MLG/Croscim/data/mounted/2024/*_x*.zarr"))[:30]
    files_dict = organize_by_resolution(files)
    
    # Load time values from x1 files
    times_list = []
    for f in sorted(files_dict[1]):
        basename = f.split('/')[-1]
        date_str = basename[:10]
        times_list.append(pd.to_datetime(date_str, format="%Y%m%d%H"))
    times = np.array(times_list)

    t0 = time.time()
    ds = XrDatasetMultiResTrain(
        multires=[10, 3, 1],
        precomputed=True,  # Use pre-calculated resolutions (FAST!)
        sst_daily_paths=files_dict,
        tgt_vars=["slstr_av"],
        mask=None,
        times=times,
        patch_dims={'time': 3, 'lat': 256, 'lon': 256},
        strides={'time': 1, 'lat': 64, 'lon': 64},
        resize=1,
        verbose=False
    )
    t_init = time.time() - t0
    
    print(f"OK : XrDatasetMultiResTrain created in {t_init:.3f}s")
    print(f"  precomputed: {ds.precomputed}")
    print(f"  is_multiresolution: {ds.is_multiresolution}")
    print(f"  multires: {ds.multires}")
    print(f"  Dataset length: {len(ds)}")
    
    if len(ds) > 0:
        print(f"\n  Testing item extraction...")
        t0 = time.time()
        sample = ds[0]
        t_item = time.time() - t0
        
        print(f"  Got item in {t_item:.3f}s")
        print(f"  Keys in sample: {sorted(sample.keys())[:5]}...")
        has_multires = any(k in sample for k in ['patch_x1', 'patch_x3', 'patch_x10'])
        print(f"  Has multi-resolution patches: {has_multires}")
    return ds


def test_xrdataset_multires_pooled():
    """Test XrDatasetMultiResTrain with precomputed=False (pooling mode)."""
    print("\nTEST 4: XrDatasetMultiResTrain with precomputed=False")
    files = sorted(glob.glob("/home/malegu/4D-MLG/Croscim/data/mounted/2024/*_x1.zarr"))[:9]
    
    # Load time values
    times_list = []
    for f in sorted(files):
        basename = f.split('/')[-1]
        date_str = basename[:10]
        times_list.append(pd.to_datetime(date_str, format="%Y%m%d%H"))
    times = np.array(times_list)
    
    print(f"Creating XrDatasetMultiResTrain with:")
    print(f"  multires: [10, 3, 1]")
    print(f"  precomputed: False (pool x1 on-the-fly)")
    print(f"  Files: {len(files)} x1 files")
    print(f"  Times: {len(times)} time steps")
    
    # Create dataset with precomputed=False
    t0 = time.time()
    ds = XrDatasetMultiResTrain(
        multires=[10, 3, 1],
        precomputed=False,  # Pool on-the-fly (SLOWER but works for inference)
        sst_daily_paths=files,  # Pass as list, not dict
        tgt_vars=["slstr_av"],
        mask=None,
        times=times,
        patch_dims={'time': 3, 'lat': 256, 'lon': 256},
        strides={'time': 1, 'lat': 64, 'lon': 64},
        resize=1,
        verbose=False
    )
    t_init = time.time() - t0

    print(f"Ok : XrDatasetMultiResTrain created in {t_init:.3f}s")
    print(f"  precomputed: {ds.precomputed}")
    print(f"  is_multiresolution: {ds.is_multiresolution}")
    print(f"  multires: {ds.multires}")
    print(f"  Dataset length: {len(ds)}")
    
    return ds


if __name__ == "__main__":
    try:
        result1 = test_organize_by_resolution()
        ds2, files_dict = test_xrdataset_with_dict()
        ds3 = test_xrdataset_multires_precomputed()
        ds4 = test_xrdataset_multires_pooled()
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
