#!/usr/bin/env python3
"""
Test script to compare Zarr Directory Store vs Zarr Zip Store performance.

Zarr Zip Store stores all chunks in a single .zip file, which should:
1. Reduce file open overhead (1 file instead of ~650)
2. Maintain random access capability
3. Support concurrent reads better than SquashFS
"""
import numpy as np
import time
import xarray as xr
import zarr
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration
TEST_DATA_DIR = Path("/nwp/sst_malegu/data_2024")
OUTPUT_DIR = Path("/nwp/sst_malegu/tmp/zarr_zip_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find a sample zarr file
sample_files = sorted(TEST_DATA_DIR.glob("*_x1.zarr"))[:1]
if not sample_files:
    print("ERROR: No Zarr files found in", TEST_DATA_DIR)
    sys.exit(1)

sample_zarr = sample_files[0]
print(f"Using sample file: {sample_zarr}")

# Define output paths
dir_store_path = OUTPUT_DIR / "test_directory.zarr"
zip_store_path = OUTPUT_DIR / "test_zipstore.zip"

print("\n" + "="*60)
print("STEP 1: Load original data from directory store")
print("="*60)
start = time.time()
ds_original = xr.open_zarr(sample_zarr)
time_load_original = time.time() - start
print(f"Time to open original: {time_load_original:.3f}s")
print(f"Shape: {dict(ds_original.dims)}")
print(f"Variables: {list(ds_original.data_vars)}")

# Count files in original directory store
n_files_original = sum(1 for _ in sample_zarr.rglob('*') if _.is_file())
print(f"Number of files in directory store: {n_files_original}")

print("\n" + "="*60)
print("STEP 2: Create Zarr Zip Store version")
print("="*60)

# Remove old test files
if dir_store_path.exists():
    import shutil
    shutil.rmtree(dir_store_path)
if zip_store_path.exists():
    zip_store_path.unlink()

# Load data into memory to rechunk properly
print("Loading data into memory to prepare for rechunking...")
ds_loaded = ds_original.load()
ds_original.close()

# Create directory store copy (for fair comparison)
start = time.time()
encoding = {var: {'chunks': (512, 512)} for var in ds_loaded.data_vars if 'lat' in ds_loaded[var].dims}
ds_loaded.to_zarr(dir_store_path, mode='w', encoding=encoding, zarr_format=2)
time_write_dir = time.time() - start
print(f"Time to write directory store: {time_write_dir:.3f}s")

# Create Zip Store version
start = time.time()
store = zarr.storage.ZipStore(str(zip_store_path), mode='w')
ds_loaded.to_zarr(store, mode='w', encoding=encoding, zarr_format=2)
store.close()
time_write_zip = time.time() - start
print(f"Time to write Zip Store: {time_write_zip:.3f}s")

ds_loaded.close()

# Check file sizes
dir_size_mb = sum(f.stat().st_size for f in dir_store_path.rglob('*') if f.is_file()) / 1024**2
zip_size_mb = zip_store_path.stat().st_size / 1024**2
n_files_dir = sum(1 for _ in dir_store_path.rglob('*') if _.is_file())

print(f"\nDirectory store: {dir_size_mb:.1f} MB ({n_files_dir} files)")
print(f"Zip Store:       {zip_size_mb:.1f} MB (1 file)")

print("\n" + "="*60)
print("STEP 3: Benchmark read performance (single read)")
print("="*60)

# Test 1: Read from directory store
start = time.time()
ds_dir = xr.open_zarr(dir_store_path)
data_dir = ds_dir['slstr_av'].values  # Force load into memory
time_read_dir_single = time.time() - start
ds_dir.close()
print(f"Directory store read: {time_read_dir_single:.3f}s")

# Test 2: Read from Zip Store
start = time.time()
store = zarr.storage.ZipStore(str(zip_store_path), mode='r')
ds_zip = xr.open_zarr(store)
data_zip = ds_zip['slstr_av'].values  # Force load into memory
time_read_zip_single = time.time() - start
ds_zip.close()
store.close()
print(f"Zip Store read:       {time_read_zip_single:.3f}s")

# Verify data is identical
if np.allclose(data_dir, data_zip, equal_nan=True):
    print("Data verification: IDENTICAL")
else:
    print("Data verification: DIFFERENT (ERROR!)")

print("\n" + "="*60)
print("STEP 4: Benchmark multiple file reads (simulate batch loading)")
print("="*60)

# For this we need multiple files - let's use the first 15 days
sample_files_batch = sorted(TEST_DATA_DIR.glob("*_x1.zarr"))[:15]
print(f"Testing with {len(sample_files_batch)} files")

# Test directory store batch read
start = time.time()
datasets_dir = []
for f in sample_files_batch:
    ds = xr.open_zarr(f)
    # Simulate actually reading some data (not just opening metadata)
    _ = ds['slstr_av'].values
    datasets_dir.append(ds)
time_batch_dir = time.time() - start
for ds in datasets_dir:
    ds.close()
print(f"Directory store batch ({len(sample_files_batch)} files): {time_batch_dir:.3f}s")
print(f"  Average per file: {time_batch_dir/len(sample_files_batch):.3f}s")

# Create Zip Store versions of all files
print("\nCreating Zip Store versions for batch test...")
zip_files = []
for i, src_zarr in enumerate(sample_files_batch):
    zip_path = OUTPUT_DIR / f"batch_{i:03d}.zip"
    if not zip_path.exists():
        ds = xr.open_zarr(src_zarr)
        ds_mem = ds.load()  # Load into memory to avoid chunking issues
        ds.close()
        store = zarr.storage.ZipStore(str(zip_path), mode='w')
        encoding = {var: {'chunks': (512, 512)} for var in ds_mem.data_vars if 'lat' in ds_mem[var].dims}
        ds_mem.to_zarr(store, mode='w', encoding=encoding, zarr_format=2)
        store.close()
        ds_mem.close()
    zip_files.append(zip_path)
    print(f"  {i+1}/{len(sample_files_batch)}", end='\r')
print()

# Test Zip Store batch read
start = time.time()
datasets_zip = []
for zip_path in zip_files:
    store = zarr.storage.ZipStore(str(zip_path), mode='r')
    ds = xr.open_zarr(store)
    # Simulate actually reading some data
    _ = ds['slstr_av'].values
    datasets_zip.append((ds, store))
time_batch_zip = time.time() - start
for ds, store in datasets_zip:
    ds.close()
    store.close()
print(f"Zip Store batch ({len(zip_files)} files):       {time_batch_zip:.3f}s")
print(f"  Average per file: {time_batch_zip/len(zip_files):.3f}s")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Storage efficiency:")
print(f"  Directory store: {n_files_dir} files, {dir_size_mb:.1f} MB")
print(f"  Zip Store:       1 file, {zip_size_mb:.1f} MB")
print(f"\nSingle file read:")
print(f"  Directory store: {time_read_dir_single:.3f}s")
print(f"  Zip Store:       {time_read_zip_single:.3f}s")
print(f"  Speedup:         {time_read_dir_single/time_read_zip_single:.2f}x")
print(f"\nBatch read ({len(sample_files_batch)} files):")
print(f"  Directory store: {time_batch_dir:.3f}s ({time_batch_dir/len(sample_files_batch):.3f}s/file)")
print(f"  Zip Store:       {time_batch_zip:.3f}s ({time_batch_zip/len(zip_files):.3f}s/file)")
print(f"  Speedup:         {time_batch_dir/time_batch_zip:.2f}x")

# Cleanup
print("\nCleaning up test files...")
import shutil
if dir_store_path.exists():
    shutil.rmtree(dir_store_path)
for zf in zip_files:
    if zf.exists():
        zf.unlink()
if zip_store_path.exists():
    zip_store_path.unlink()
print("Done!")
