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
