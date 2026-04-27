"""
Fix surfmask in existing x3/x10 zarr files.
The surfmask was incorrectly computed using mean pooling instead of mode pooling,
resulting in continuous values (0.0, 0.01, ...) instead of categorical (0, 1, 2, 3).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import zarr
from scipy import stats
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm


def mode_pool_2d(arr, fy, fx):
    """Pool 2D array using mode (most frequent value). Vectorized for speed."""
    ny, nx = arr.shape
    if ny % fy != 0 or nx % fx != 0:
        arr = arr[:ny - (ny % fy), :nx - (nx % fx)]
        ny, nx = arr.shape

    out_ny, out_nx = ny // fy, nx // fx

    # Reshape to (out_ny, fy, out_nx, fx) then to (out_ny, out_nx, fy*fx)
    blocks = arr.reshape(out_ny, fy, out_nx, fx).transpose(0, 2, 1, 3).reshape(out_ny, out_nx, fy * fx)

    # Round to integers (surfmask values are 0, 1, 2, 3, 4)
    blocks_int = np.round(blocks).astype(np.int32)
    blocks_int = np.clip(blocks_int, 0, 4)  # Clamp to valid range

    # Count occurrences of each value (0-4) for each block - vectorized
    # Shape: (out_ny, out_nx, 5) - counts for values 0,1,2,3,4
    counts = np.zeros((out_ny, out_nx, 5), dtype=np.int32)
    for val in range(5):
        counts[:, :, val] = np.sum(blocks_int == val, axis=2)

    # Mode = value with max count
    result = np.argmax(counts, axis=2).astype(np.float32)

    return result


def check_needs_fix(target_path):
    """Check if a zarr file needs surfmask fix. Returns True if needs fix."""
    try:
        store = zarr.open(str(target_path), mode='r')
        if 'surfmask' not in store:
            return False
        mask = np.array(store['surfmask'][:])
        unique_vals = np.unique(mask)
        # Already categorical if only integers 0-4
        return not all(v in [0, 1, 2, 3, 4] for v in unique_vals)
    except Exception:
        return False


def fix_surfmask_in_zarr(x1_path, target_path, factor):
    """Fix surfmask in target zarr by recomputing from x1 using mode pooling."""
    store_x1 = zarr.open(str(x1_path), mode='r')
    surfmask_x1 = np.array(store_x1['surfmask'][:])

    new_surfmask = mode_pool_2d(surfmask_x1, factor, factor)

    store_target = zarr.open(str(target_path), mode='r+')
    store_target['surfmask'][:] = new_surfmask
    return True


def process_directory(data_dir, dry_run=False):
    """Process all x3 and x10 files in a directory."""
    data_dir = Path(data_dir)

    # Find all x1 files
    x1_files = sorted(data_dir.glob('*_x1.zarr'))
    if not x1_files:
        print(f"No x1.zarr files found in {data_dir}")
        return

    print(f"Scanning {len(x1_files)} dates in {data_dir}...")

    # First pass: identify files that need fixing
    files_to_fix = []
    for x1_path in tqdm(x1_files, desc="Checking"):
        basename = x1_path.name.replace('_x1.zarr', '')

        x3_path = data_dir / f"{basename}_x3.zarr"
        if x3_path.exists() and check_needs_fix(x3_path):
            files_to_fix.append((x1_path, x3_path, 3))

        x10_path = data_dir / f"{basename}_x10.zarr"
        if x10_path.exists() and check_needs_fix(x10_path):
            files_to_fix.append((x1_path, x10_path, 10))

    # Report
    if not files_to_fix:
        print("\nAll surfmasks are already correct.")
        return

    for _, target_path, factor in files_to_fix:
        print(f"  - {target_path.name}")
    
    print(f"\nTotal: ")
    print(f"  {len(files_to_fix)} fichiers a corriger:")
    
    if dry_run:
        print(f"Relancer sans --dry-run pour appliquer les corrections.")
        return

    # Fix files
    print(f"\nCorrection en cours...")
    for x1_path, target_path, factor in tqdm(files_to_fix, desc="Fixing"):
        fix_surfmask_in_zarr(x1_path, target_path, factor)

    print(f"\nTermine! {len(files_to_fix)} fichiers corriges.")


def main():
    parser = argparse.ArgumentParser(description="Fix surfmask in x3/x10 zarr files")
    parser.add_argument('data_dir', type=str, help='Directory containing zarr files')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be done')

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"Error: Directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    process_directory(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
