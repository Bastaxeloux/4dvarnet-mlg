"""
Quick script to visualize the surfmask on the full globe
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load one file
data_dir = Path('/dmidata/users/malegu/data/netcdf_2024')
files = sorted(data_dir.glob('*_13vars.nc'))
print(f"Loading: {files[0]}")
ds = xr.open_dataset(files[0])
surfmask = ds['surfmask'].values
unique_values = np.unique(surfmask)
total_pixels = surfmask.size
for val in unique_values:
    count = (surfmask == val).sum()
    print(f"  Value {int(val)}: {count:>9,} ({count/total_pixels*100:>5.1f}%)")

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(surfmask, cmap='viridis', origin='lower', interpolation='nearest')
ax.set_title('Surfmask', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
cbar = plt.colorbar(im, ax=ax, label='Surfmask value', ticks=unique_values)
cbar.ax.set_yticklabels([f'{int(v)}' for v in unique_values])

# Add text info with all values
info_text = f"Total pixels: {total_pixels:,}\n\n"
for val in unique_values:
    count = (surfmask == val).sum()
    info_text += f"Value {int(val)}: {count:>9,} ({count/total_pixels*100:>5.1f}%)\n"
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
        va='top', ha='left', fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
output_file = 'figs/SST/00_surfmask.png'
Path('figs/SST').mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# Minimal ASCII compare: keep tiny and simple
if __name__ == '__main__':
    import sys
    import warnings

    # Usage: python check_surfmask.py /path/to/ascii.asc
    if len(sys.argv) <= 1:
        print("No ASCII path provided. To compare run: python check_surfmask.py /path/to/original_surfmask.asc")
        sys.exit(0)

    ascii_path = sys.argv[1]
    ascii_arr = None
    ascii_nodata = None
    try:
        ascii_arr = np.loadtxt(ascii_path)
    except Exception:
        # Simple fallback: skip first 3 lines (common small header) and retry
        try:
            ascii_arr = np.loadtxt(ascii_path, skiprows=3)
            print("Loaded ASCII by skipping first 3 lines.")
        except Exception as e:
            warnings.warn(f"Could not read ASCII file {ascii_path}: {e}")
            sys.exit(1)

    print(f"ASCII loaded: {ascii_path}; shape={ascii_arr.shape}")

    # Print some basic ASCII stats
    unique_a = np.unique(ascii_arr)
    total_a = ascii_arr.size
    print("ASCII stats:")
    for v in unique_a:
        cnt = int((ascii_arr == v).sum())
        print(f"  Value {int(v)}: {cnt:>9,} ({cnt/total_a*100:>5.1f}%)")

    # Save a simple image of the ASCII surfmask
    out_dir = Path('figs/SST')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(ascii_arr, cmap='viridis', origin='lower', interpolation='nearest')
    ax.set_title('Surfmask (ASCII)', fontsize=14)
    plt.tight_layout()
    out_path = out_dir / '00_surfmask_ascii.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

