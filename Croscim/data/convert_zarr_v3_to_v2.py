"""
Convert zarr v3 directory to zarr v2 format.
Creates a backup copy (data_YYYY_v3) before converting.

Usage:
    python convert_zarr_v3_to_v2.py data_2024 --dry-run
    python convert_zarr_v3_to_v2.py data_2024
"""

import argparse
import sys
import shutil
from pathlib import Path
import zarr
from tqdm import tqdm


def convert_zarr_v3_to_v2(zarr_path):
    """Convert a single zarr v3 file to v2 format in-place."""
    try:
        temp_path = Path(str(zarr_path) + '_temp_v2')
        source = zarr.open_group(str(zarr_path), mode='r')
        target = zarr.open_group(str(temp_path), mode='w', zarr_format=2)
        zarr.copy_all(source, target, log=None)
        shutil.rmtree(zarr_path)
        temp_path.rename(zarr_path)
        return True
    except Exception as e:
        print(f"\n  ERROR: {zarr_path.name}: {e}")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        return False


def process_directory(data_dir, dry_run=False):
    """Process all zarr files in a data_YYYY directory."""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    zarr_files = sorted(data_dir.glob('*_x*.zarr'))
    if not zarr_files:
        print(f"No zarr files found in {data_dir}")
        return
    
    print(f"Found {len(zarr_files)} zarr files in {data_dir.name}/")
    
    print("\nExamples:")
    for f in zarr_files[:3]:
        print(f"  - {f.name}")
    if len(zarr_files) > 3:
        print(f"  ... and {len(zarr_files) - 3} more")
    
    if dry_run:
        print(f"\n--dry-run mode: no changes made")
        print(f"Run without --dry-run to convert {len(zarr_files)} files")
        return
    
    # Create backup directory
    backup_dir = Path(str(data_dir) + '_v3')
    
    if backup_dir.exists():
        print(f"\nERROR: Backup directory already exists: {backup_dir.name}")
        print("Remove it or rename the data directory.")
        return
    
    print(f"\nStep 1: Creating backup {backup_dir.name}/")
    shutil.copytree(data_dir, backup_dir)
    print(f"✓ Backup created")
    
    # Convert files
    print(f"\nStep 2: Converting {len(zarr_files)} files to zarr v2...")
    
    success = 0
    failed = []
    
    for zarr_path in tqdm(zarr_files, desc="Converting"):
        if convert_zarr_v3_to_v2(zarr_path):
            success += 1
        else:
            failed.append(zarr_path.name)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DONE: {success}/{len(zarr_files)} files converted")
    if failed:
        print(f"Failed: {len(failed)}")
        for name in failed:
            print(f"  - {name}")
    print(f"\nOriginal data backed up in: {backup_dir.name}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert zarr v3 to v2 for a single data_YYYY directory",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', type=str, help='data_YYYY directory to convert')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    args = parser.parse_args()
    process_directory(args.data_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
