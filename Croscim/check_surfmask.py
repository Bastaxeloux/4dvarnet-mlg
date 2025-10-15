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
print(f"\nDataset variables: {list(ds.data_vars)}")
print(f"Dataset coords: {list(ds.coords)}")

# Get surfmask
surfmask = ds['surfmask'].values
print(f"\nSurfmask shape: {surfmask.shape}")
unique_values = np.unique(surfmask)
print(f"Surfmask unique values: {unique_values}")
print(f"Surfmask dtype: {surfmask.dtype}")

# Count pixels for all values
total_pixels = surfmask.size
print(f"\nPixel counts:")
for val in unique_values:
    count = (surfmask == val).sum()
    print(f"  Value {int(val)}: {count:>9,} ({count/total_pixels*100:>5.1f}%)")

# Plot
fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(surfmask, cmap='viridis', origin='lower', interpolation='nearest')
ax.set_title('Surfmask Global - 5 valeurs distinctes', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude index')
ax.set_ylabel('Latitude index')

# Add colorbar with explicit labels for all 5 values
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
output_file = 'figs/SST/check_surfmask.png'
Path('figs/SST').mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_file}")
plt.close()

# Also check one satellite data to see ocean coverage
print("\n" + "="*80)
print("Checking SLSTR coverage for comparison:")
slstr_av = ds['slstr_av'].values
valid_slstr = ~np.isnan(slstr_av)
print(f"Valid SLSTR pixels: {valid_slstr.sum()} ({valid_slstr.sum()/total_pixels*100:.1f}%)")

# Analysis - which value is ocean?
print("\nNote: Earth is ~71% ocean, ~29% land")
print("\nHypothèses possibles:")
for val in unique_values:
    count = (surfmask == val).sum()
    pct = count/total_pixels*100
    print(f"  Si valeur {int(val)} = OCEAN : {pct:.1f}% de la surface")
