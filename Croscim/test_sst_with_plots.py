import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import glob
from contrib.SST.data import XrDataset, BaseDataModule
from contrib.SST.load_data import VAR_GROUPS, COVARIATES
import yaml

# Configuration
DATA_DIR = "/dmidata/users/malegu/data/netcdf_2024"
OUTPUT_DIR = "figs/SST"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load norm_stats
with open('contrib/SST/norm_stats.yaml', 'r') as f:
    norm_stats_file = yaml.safe_load(f)
    norm_stats = norm_stats_file['norm_stats']  # Extract the nested dict

sst_files = sorted(glob.glob(f"{DATA_DIR}/*.nc"))
print(f"\nFound {len(sst_files)} SST files")
times = np.arange(15)

patch_dims = {'time': 15, 'lat': 768, 'lon': 768}
strides = {'time': 1, 'lat': 768, 'lon': 768}

dataset = XrDataset(
    sst_daily_paths=sst_files[:30],  # Use first 30 days
    tgt_vars=['slstr_av', 'aasti_av'],
    mask=None,  # Will use surfmask from data
    times=times,
    patch_dims=patch_dims,
    strides=strides,
    postpro_fn=None,  # No preprocessing for raw data test
    resize=1,
    res=5.0,
    verbose=False
)

n_patches = len(dataset)
print(f"Dataset created with patches: {n_patches}")
print(f"Grid shape: lat={len(dataset.lat_1d)}, lon={len(dataset.lon_1d)}")

sample = dataset[0]


print("\nSATELLITE COVERAGE ")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Sample Patch (t=7)', fontsize=16, fontweight='bold')
t_idx = 7
for i, sat in enumerate(['aasti', 'avhrr', 'pmw', 'slstr']):
    var_key = f"{sat}_av"
    if var_key in sample:
        data = sample[var_key][t_idx]
        
        # Average value
        ax = axes[0, i]
        im = ax.imshow(data, cmap='RdYlBu_r', origin='lower')
        ax.set_title(f'{sat.upper()} - Average SST', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude index')
        ax.set_ylabel('Latitude index')
        plt.colorbar(im, ax=ax, label='SST (°C)')
        
        # Valid data mask
        ax = axes[1, i]
        valid_mask = ~np.isnan(data)
        coverage = valid_mask.sum() / valid_mask.size * 100
        im = ax.imshow(valid_mask, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'{sat.upper()} - Coverage: {coverage:.1f}%', fontsize=12)
        ax.set_xlabel('Longitude index')
        ax.set_ylabel('Latitude index')
        plt.colorbar(im, ax=ax, label='Valid data')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_satellite_coverage.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/01_satellite_coverage.png")
plt.close()


print("\nTARGET FUSION (SLSTR + AASTI)")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Target SST Fusion Strategy', fontsize=16, fontweight='bold')
t_idx = 7
# SLSTR data
ax = axes[0, 0]
slstr_data = sample['slstr_av'][t_idx]
im = ax.imshow(slstr_data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('SLSTR SST (priority)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
# AASTI data
ax = axes[0, 1]
aasti_data = sample['aasti_av'][t_idx]
im = ax.imshow(aasti_data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('AASTI SST (fallback for poles)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
# Fused target
ax = axes[1, 0]
tgt_sst = sample['tgt_sst'][t_idx]
im = ax.imshow(tgt_sst, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('Fused Target SST (slstr where valid, else aasti)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
# Coverage map
ax = axes[1, 1]
slstr_valid = ~np.isnan(slstr_data)
aasti_valid = ~np.isnan(aasti_data)
tgt_valid = ~np.isnan(tgt_sst)

coverage_map = np.zeros_like(tgt_sst)
coverage_map[slstr_valid] = 1  # SLSTR: blue
coverage_map[~slstr_valid & aasti_valid] = 2  # AASTI only: red
coverage_map[~tgt_valid] = 0  # No data: white

im = ax.imshow(coverage_map, cmap='RdBu_r', origin='lower', vmin=0, vmax=2)
ax.set_title('Data Source Map', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['No data', 'SLSTR', 'AASTI'])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_target_fusion.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/02_target_fusion.png")
plt.close()

# Test 5: Inpainting simulation (simplified version)

print("\nINPAINTING")

# Simple inpainting function (local to test script)
def simulate_inpainting(data):
    """Simulate 50% removal with random rectangles (simplified version for visualization)"""
    obs_mask = ~np.isnan(data)
    inpaint_mask = np.zeros_like(data, dtype=np.float32)
    masked_data = data.copy()
    
    dtime, dyc, dxc = data.shape
    for t in range(dtime):
        if np.sum(obs_mask[t]) > .02 * dyc * dxc:
            obs_obj = .5 * np.sum(obs_mask[t])
            initial_valid = obs_mask[t].copy()
            current_mask = obs_mask[t].copy()
            
            while np.sum(current_mask) >= obs_obj:
                half_h = np.random.randint(2, 10)
                half_w = np.random.randint(2, 10)
                yc = np.random.randint(0, dyc)
                xc = np.random.randint(0, dxc)
                current_mask[max(0,yc-half_h):min(dyc,yc+half_h+1),
                            max(0,xc-half_w):min(dxc,xc+half_w+1)] = 0
            
            # Mark removed pixels
            inpaint_mask[t] = (initial_valid & ~current_mask).astype(np.float32)
            masked_data[t] = np.where(current_mask, data[t], np.nan)
    
    return masked_data, inpaint_mask

# Apply inpainting to SLSTR
slstr_inpainted, inpaint_mask = simulate_inpainting(sample['slstr_av'])

# Visualize inpainting
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Inpainting : 50% Removal', fontsize=16, fontweight='bold')

t_idx = 7

# Original SLSTR
ax = axes[0, 0]
original_slstr = sample['slstr_av'][t_idx]
im = ax.imshow(original_slstr, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('Original SLSTR SST', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
valid_orig = ~np.isnan(original_slstr)
ax.text(0.02, 0.98, f'Valid pixels: {valid_orig.sum()}', 
        transform=ax.transAxes, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# After inpainting
ax = axes[0, 1]
inpainted_slstr_t = slstr_inpainted[t_idx]
im = ax.imshow(inpainted_slstr_t, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('After Inpainting', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
valid_inpaint = ~np.isnan(inpainted_slstr_t)
ax.text(0.02, 0.98, f'Valid pixels: {valid_inpaint.sum()}', 
        transform=ax.transAxes, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Inpaint mask
ax = axes[0, 2]
mask_t = inpaint_mask[t_idx]
im = ax.imshow(mask_t, cmap='Reds', origin='lower', vmin=0, vmax=1)
ax.set_title('Inpaint Mask (1=removed, 0=kept)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Mask value')
ax.text(0.02, 0.98, f'Removed pixels: {mask_t.sum():.0f}', 
        transform=ax.transAxes, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Difference (removed regions)
ax = axes[1, 0]
removed_regions = valid_orig.astype(float) - valid_inpaint.astype(float)
im = ax.imshow(removed_regions, cmap='Reds', origin='lower', vmin=0, vmax=1)
ax.set_title('Removed Regions', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Removed')

# Rectangular patterns
ax = axes[1, 1]
# Show edge detection to highlight rectangles
from scipy.ndimage import sobel
edges_x = sobel(mask_t, axis=1)
edges_y = sobel(mask_t, axis=0)
edges = np.sqrt(edges_x**2 + edges_y**2)
im = ax.imshow(edges, cmap='hot', origin='lower')
ax.set_title('Rectangle Edges (2-10 pixels)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Edge strength')

# Statistics
ax = axes[1, 2]
ax.axis('off')
stats_text = f"""
INPAINTING STATISTICS

Original valid pixels: {valid_orig.sum()}
After inpainting: {valid_inpaint.sum()}
Removed pixels: {mask_t.sum():.0f}

Removal rate: {(1 - valid_inpaint.sum()/valid_orig.sum())*100:.1f}%
"""
ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_inpainting_mask.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/03_inpainting_mask.png")
plt.close()




print("\nSPATIAL METADATA CHANNELS")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Spatial & Temporal Metadata Channels', fontsize=16, fontweight='bold')

# Latitude
ax = axes[0, 0]
im = ax.imshow(sample['lat'], cmap='viridis', origin='lower')
ax.set_title('Latitude Channel (normalized)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized lat')

# Longitude
ax = axes[0, 1]
im = ax.imshow(sample['lon'], cmap='viridis', origin='lower')
ax.set_title('Longitude Channel (normalized)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized lon')

# Time (day of year)
ax = axes[1, 0]
im = ax.imshow(sample['time'], cmap='twilight', origin='lower')
ax.set_title('Time Channel (day of year / 366)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized time')

# Surface mask
ax = axes[1, 1]
im = ax.imshow(sample['surfmask'], cmap='binary', origin='lower', vmin=0, vmax=1)
ax.set_title('Surface Mask (0=ocean, 1=land)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Mask value')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_spatial_channels.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/04_spatial_channels.png")
plt.close()


print("\nTEMPORAL EVOLUTION")

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('SST Temporal Evolution', fontsize=16, fontweight='bold')

for t in range(min(15, patch_dims['time'])):
    row = t // 5
    col = t % 5
    ax = axes[row, col]
    
    data = sample['slstr_av'][t]
    im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
    ax.set_title(f't={t}', fontsize=10)
    ax.axis('off')
    
    if col == 4:
        plt.colorbar(im, ax=ax, label='SST (°C)', fraction=0.046)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_temporal_evolution.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/05_temporal_evolution.png")
plt.close()