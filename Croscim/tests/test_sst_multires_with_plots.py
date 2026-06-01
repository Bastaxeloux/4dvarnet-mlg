import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import glob
import time
import pandas as pd

from contrib.SST.data_multires import XrDatasetMultiResTrain
from contrib.SST.load_data import organize_by_resolution, VAR_GROUPS, COVARIATES
import yaml


def get_data(patch, var_key, t_idx):
    """Get data from patch (which can be dict or TrainingItem)"""
    if isinstance(patch, dict) and var_key in patch:
        return patch[var_key][t_idx]
    elif hasattr(patch, var_key):
        return getattr(patch, var_key)[t_idx]
    return None

def get_geo_bounds(patch):
    """
    Extract geographic bounds from a patch.
    Returns (lat_min, lat_max, lon_min, lon_max) in actual geographic coordinates.
    """
    if isinstance(patch, dict):
        lat_norm = patch.get('lat', None)
        lon_norm = patch.get('lon', None)
    else:
        lat_norm = getattr(patch, 'lat', None)
        lon_norm = getattr(patch, 'lon', None)
    if lat_norm is None or lon_norm is None:
        return None
    # Dénormaliser
    lat_real = lat_norm * 90.0
    lon_real = lon_norm * 180.0
    
    lat_min = float(np.nanmin(lat_real))
    lat_max = float(np.nanmax(lat_real))
    lon_min = float(np.nanmin(lon_real))
    lon_max = float(np.nanmax(lon_real))
    return (lat_min, lat_max, lon_min, lon_max)

def get_pixel_bounds_in_patch(patch_from, geo_bounds_to_find):
    """
    Find pixel indices in patch_from that correspond to geographic bounds.
    Converts geographic coordinates to pixel indices.
    """
    if geo_bounds_to_find is None:
        return None
    if isinstance(patch_from, dict):
        lat_norm = patch_from.get('lat', None)
        lon_norm = patch_from.get('lon', None)
    else:
        lat_norm = getattr(patch_from, 'lat', None)
        lon_norm = getattr(patch_from, 'lon', None)
    if lat_norm is None or lon_norm is None:
        return None
    lat_real = lat_norm * 90.0
    lon_real = lon_norm * 180.0
    
    lat_min, lat_max, lon_min, lon_max = geo_bounds_to_find
    
    # Find pixels within the geographic bounds
    mask = ((lat_real >= lat_min) & (lat_real <= lat_max) & 
            (lon_real >= lon_min) & (lon_real <= lon_max))
    
    if np.sum(mask) == 0:
        return None
    
    pixels = np.where(mask)
    return {
        'lat_min': int(np.min(pixels[0])),
        'lat_max': int(np.max(pixels[0])),
        'lon_min': int(np.min(pixels[1])),
        'lon_max': int(np.max(pixels[1]))
    }

# Configuration
OUTPUT_DIR = "figs/SST_multires"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with open('contrib/SST/norm_stats.yaml', 'r') as f:
    norm_stats_file = yaml.safe_load(f)
    norm_stats = norm_stats_file['norm_stats']

print("\n" + "="*80)
print("MULTI-RESOLUTION DATALOADER TEST WITH PLOTS")
print("="*80)


print("\n[1/5] Setting up data...")
mount_dir = "/nwp/sst_malegu/data_2024"
files = sorted(glob.glob(f"{mount_dir}/*_x*.zarr"))[:45]  # Use 45 files for 15 timesteps
files_dict = organize_by_resolution(files)
times_list = []
for f in sorted(files_dict[1]):
    basename = f.split('/')[-1]
    date_str = basename[:10]
    times_list.append(pd.to_datetime(date_str, format="%Y%m%d%H"))
times = np.array(times_list)

print(f"Ok : Files organized by resolution:")
print(f"  x1: {len(files_dict[1])} files")
print(f"  x3: {len(files_dict[3])} files")
print(f"  x10: {len(files_dict[10])} files")
print(f"Time range: {len(times)} time steps")





print("\n[2/5] Creating XrDatasetMultiResTrain with precomputed=True...")

t0 = time.time()
dataset_precomp = XrDatasetMultiResTrain(
    multires=[10, 3, 1],
    precomputed=True,
    sst_daily_paths=files_dict,
    tgt_vars=['slstr_av', 'aasti_av'],
    mask=None,
    times=times,
    patch_dims={'time': 15, 'lat': 256, 'lon': 256},
    strides={'time': 1, 'lat': 128, 'lon': 128},
    resize=1,
    res=5.0,
    verbose=False
)
t_init_precomp = time.time() - t0

print(f"Dataset created in {t_init_precomp:.3f}s")
print(f"  precomputed: {dataset_precomp.precomputed}")
print(f"  Dataset length: {len(dataset_precomp)}")

# Use a fixed index for reproducible nested patch visualization
# Use middle of dataset to get patches closer to center of globe (not at poles/corners)
sample_idx = len(dataset_precomp) // 2 if len(dataset_precomp) > 0 else None
if sample_idx is None:
    print(f"  ERROR: Dataset is empty! len={len(dataset_precomp)}")
    sys.exit(1)

print(f"  Using patch at index {sample_idx} (middle of dataset for geographic center)")

t0 = time.time()
sample_precomp = dataset_precomp[sample_idx]
t_sample_precomp = time.time() - t0

print(f"  Sample extracted in {t_sample_precomp:.3f}s")
print(f"  Multi-res patches: {sorted([k for k in sample_precomp.keys() if k.startswith('patch')])}")




print("\n[3/5] Creating XrDatasetMultiResTrain with precomputed=False...")

files_x1_only = sorted(glob.glob(f"{mount_dir}/*_x1.zarr"))[:45]  # Use 45 files for 15 timesteps
t0 = time.time()
dataset_pooled = XrDatasetMultiResTrain(
    multires=[10, 3, 1],
    precomputed=False,
    sst_daily_paths=files_x1_only,
    tgt_vars=['slstr_av', 'aasti_av'],
    mask=None,
    times=times,
    patch_dims={'time': 15, 'lat': 256, 'lon': 256},
    strides={'time': 1, 'lat': 128, 'lon': 128},
    resize=1,
    res=5.0,
    verbose=False
)
t_init_pooled = time.time() - t0

print(f"Dataset created in {t_init_pooled:.3f}s")
print(f"  precomputed: {dataset_pooled.precomputed}")
print(f"  Dataset length: {len(dataset_pooled)}")

t0 = time.time()
sample_pooled = dataset_pooled[sample_idx]  # Use same index as precomputed
t_sample_pooled = time.time() - t0
print(f"  Sample extracted in {t_sample_pooled:.3f}s (index {sample_idx})")




print("\n[3.5/5] Data coverage diagnostics...")
print("PRECOMPUTED MODE:")
print(f"  Sample keys: {sample_precomp.keys()}")
print(f"  Sample type: {type(sample_precomp)}")

for res, factor in [(1, 'x1'), (3, 'x3'), (10, 'x10')]:
    patch_key = f'patch_{factor}'
    if patch_key in sample_precomp:
        patch = sample_precomp[patch_key]
        if isinstance(patch, dict):
            if 'slstr_av' in patch:
                data = patch['slstr_av'][2]  # t=2
                valid = np.sum(~np.isnan(data))
                coverage = 100 * valid / data.size
                print(f"  {factor}: {data.shape} - {valid}/{data.size} valid ({coverage:.1f}%)")
            else:
                print(f"  {factor}: dict but no slstr_av - keys: {list(patch.keys())[:5]}")
        elif hasattr(patch, 'slstr_av'):
            data = getattr(patch, 'slstr_av')[2]  # t=2
            valid = np.sum(~np.isnan(data))
            coverage = 100 * valid / data.size
            print(f"  {factor}: {data.shape} - {valid}/{data.size} valid ({coverage:.1f}%)")
        else:
            print(f"  {factor}: object exists but no slstr_av attribute")

print("POOLED MODE:")
for res, factor in [(1, 'x1'), (3, 'x3'), (10, 'x10')]:
    patch_key = f'patch_{factor}'
    if patch_key in sample_pooled:
        patch = sample_pooled[patch_key]
        if isinstance(patch, dict):
            if 'slstr_av' in patch:
                data = patch['slstr_av'][2]
                valid = np.sum(~np.isnan(data))
                coverage = 100 * valid / data.size
                print(f"  {factor}: {data.shape} - {valid}/{data.size} valid ({coverage:.1f}%)")
            else:
                print(f"  {factor}: dict but no slstr_av")
        elif hasattr(patch, 'slstr_av'):
            data = getattr(patch, 'slstr_av')[2]
            valid = np.sum(~np.isnan(data))
            coverage = 100 * valid / data.size
            print(f"  {factor}: {data.shape} - {valid}/{data.size} valid ({coverage:.1f}%)")



print("\n[4/5] Generating visualization plots...")

def get_patch_bounds(patch, res_label):
    """Extract lat/lon bounds from a patch's coordinates"""
    if isinstance(patch, dict):
        if 'lat' in patch and 'lon' in patch:
            lat = patch['lat'].squeeze()
            lon = patch['lon'].squeeze()
            # X1 patch has normalized coordinates [-1, 1], need to denormalize
            # This is because x1 comes from XrDataset which returns normalized coords
            # while x3/x10 come from extract_enlarged_patch_from_datasets which returns real coords
            lat = lat * 90.0
            lon = lon * 180.0
            lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
            lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
            return {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max}
    elif hasattr(patch, 'lat') and hasattr(patch, 'lon'):
        lat = patch.lat.squeeze()
        lon = patch.lon.squeeze()
        # x3/x10 patches already have real coordinates
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        return {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max}
    return None

def pixel_bounds_from_coords(patch, coord_bounds):
    """Convert lat/lon bounds to pixel coordinates within the patch"""
    if coord_bounds is None:
        return None
    
    if isinstance(patch, dict):
        if 'lat' not in patch or 'lon' not in patch:
            return None
        lat_2d = patch['lat'].squeeze()
        lon_2d = patch['lon'].squeeze()
    elif hasattr(patch, 'lat') and hasattr(patch, 'lon'):
        lat_2d = patch.lat.squeeze()
        lon_2d = patch.lon.squeeze()
    else:
        return None
    
    # Find pixels within the coordinate bounds
    mask = ((lat_2d >= coord_bounds['lat_min']) & (lat_2d <= coord_bounds['lat_max']) &
            (lon_2d >= coord_bounds['lon_min']) & (lon_2d <= coord_bounds['lon_max']))
    
    if np.sum(mask) == 0:
        return None
    
    pixels = np.where(mask)
    return {
        'pixel_lat_min': int(np.min(pixels[0])),
        'pixel_lat_max': int(np.max(pixels[0])),
        'pixel_lon_min': int(np.min(pixels[1])),
        'pixel_lon_max': int(np.max(pixels[1]))
    }

def draw_rectangle_on_ax(ax, pixel_bounds, label='', color='red', linewidth=2):
    """Draw a rectangle on matplotlib axis using pixel coordinates"""
    if pixel_bounds is None:
        return
    
    rect_lon = [pixel_bounds['pixel_lon_min'], pixel_bounds['pixel_lon_max'],
                pixel_bounds['pixel_lon_max'], pixel_bounds['pixel_lon_min'],
                pixel_bounds['pixel_lon_min']]
    rect_lat = [pixel_bounds['pixel_lat_min'], pixel_bounds['pixel_lat_min'],
                pixel_bounds['pixel_lat_max'], pixel_bounds['pixel_lat_max'],
                pixel_bounds['pixel_lat_min']]
    
    ax.plot(rect_lon, rect_lat, color=color, linewidth=linewidth, label=label)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Multi-Resolution Patches with Nested Resolution Overlays', fontsize=16, fontweight='bold')
t_idx = 2 
var_key = 'slstr_av'

# Get bounds of all patches for later use
bounds = {}
for res, factor in [(1, 'x1'), (3, 'x3'), (10, 'x10')]:
    patch_key = f'patch_{factor}'
    if patch_key in sample_precomp:
        patch = sample_precomp[patch_key]
        b = get_patch_bounds(patch, factor)
        bounds[res] = b

# Plot each resolution
for i, (res, factor) in enumerate([(1, 'x1'), (3, 'x3'), (10, 'x10')]):
    patch_key = f'patch_{factor}'
    if patch_key in sample_precomp:
        patch = sample_precomp[patch_key]
        # Handle both dict and TrainingItem
        if isinstance(patch, dict):
            if var_key in patch:
                data = patch[var_key][t_idx]
            else:
                continue
        elif hasattr(patch, var_key):
            data = getattr(patch, var_key)[t_idx]
        else:
            continue
        ax = axes[0, i]
        im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
        ax.set_title(f'SLSTR {factor}\nShape: {data.shape}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='SST (°C)')
        
        # Draw rectangles for nested resolutions
        legend_drawn = False
        
        if factor == 'x3' and bounds.get(1):  # Draw x1 bounds on x3 plot
            pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
            if pixel_bounds is not None:
                draw_rectangle_on_ax(ax, pixel_bounds, label='x1 patch', color='cyan', linewidth=2)
                legend_drawn = True
        
        if factor == 'x10' and bounds.get(3):  # Draw x3 bounds on x10 plot
            pixel_bounds = pixel_bounds_from_coords(patch, bounds[3])
            if pixel_bounds is not None:
                draw_rectangle_on_ax(ax, pixel_bounds, label='x3 patch', color='lime', linewidth=2)
                legend_drawn = True
        
        if factor == 'x10' and bounds.get(1):  # Draw x1 bounds on x10 plot (second rect)
            pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
            if pixel_bounds is not None:
                draw_rectangle_on_ax(ax, pixel_bounds, label='x1 patch', color='cyan', linewidth=2)
                legend_drawn = True
        
        if legend_drawn:
            ax.legend(loc='upper right', fontsize=9)
        
        ax = axes[1, i]
        valid_mask = ~np.isnan(data)
        coverage = valid_mask.sum() / valid_mask.size * 100
        im = ax.imshow(valid_mask, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'{factor} Coverage: {coverage:.1f}%', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Valid data')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_multires_alignment_precomputed.png", dpi=150, bbox_inches='tight')
print(f" Saved: 01_multires_alignment_precomputed.png")
plt.close()



# NOUVELLE FIGURE: Comparaison tgt_sst (fusion) vs slstr_av vs aasti_av
print(f"\n[4b/5] Generating FUSION integrity check plot...")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Fusion Integrity Check: tgt_sst vs slstr_av vs aasti_av (x10 resolution)', 
             fontsize=16, fontweight='bold')

# Variables à comparer
var_names = ['tgt_sst', 'slstr_av', 'aasti_av']
var_labels = ['FUSION (tgt_sst)', 'SLSTR', 'AASTI']

patch_key = 'patch_x10'
if patch_key in sample_precomp:
    patch = sample_precomp[patch_key]
    
    for i, (var_name, var_label) in enumerate(zip(var_names, var_labels)):
        # Extraire les données
        if isinstance(patch, dict):
            if var_name in patch:
                data = patch[var_name][t_idx]
            else:
                # Variable non disponible
                axes[0, i].text(0.5, 0.5, f'{var_name}\nNOT AVAILABLE', 
                              ha='center', va='center', fontsize=14)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                axes[2, i].axis('off')
                continue
        elif hasattr(patch, var_name):
            data = getattr(patch, var_name)[t_idx]
        else:
            axes[0, i].text(0.5, 0.5, f'{var_name}\nNOT AVAILABLE', 
                          ha='center', va='center', fontsize=14)
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            axes[2, i].axis('off')
            continue
        
        # Ligne 1: Données SST
        ax = axes[0, i]
        im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-2, vmax=32)
        valid_count = np.sum(~np.isnan(data))
        total_count = data.size
        coverage_pct = 100 * valid_count / total_count
        ax.set_title(f'{var_label}\nCoverage: {coverage_pct:.1f}%', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='SST (°C)')
        
        # Ligne 2: Masque de validité
        ax = axes[1, i]
        valid_mask = ~np.isnan(data)
        im = ax.imshow(valid_mask, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Valid pixels: {valid_count}/{total_count}', fontsize=10)
        plt.colorbar(im, ax=ax, label='Valid')
        
        # Ligne 3: Statistiques
        ax = axes[2, i]
        ax.axis('off')
        if valid_count > 0:
            stats_text = f"""
Statistics:
  Min:  {np.nanmin(data):.2f} °C
  Max:  {np.nanmax(data):.2f} °C
  Mean: {np.nanmean(data):.2f} °C
  Std:  {np.nanstd(data):.2f} °C
  
  Valid: {valid_count}/{total_count}
  Coverage: {coverage_pct:.1f}%
            """
        else:
            stats_text = "NO VALID DATA"
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
               verticalalignment='center')
    
    # Vérifier si tgt_sst == aasti_av (bug potentiel)
    if isinstance(patch, dict):
        has_tgt = 'tgt_sst' in patch
        has_aasti = 'aasti_av' in patch
        if has_tgt and has_aasti:
            tgt_data = patch['tgt_sst']
            aasti_data = patch['aasti_av']
            are_identical = np.allclose(tgt_data, aasti_data, equal_nan=True)
            
            # Ajouter un warning si identiques
            if are_identical:
                fig.text(0.5, 0.02, 
                        '⚠️  WARNING: tgt_sst is IDENTICAL to aasti_av - Fusion may be broken!',
                        ha='center', fontsize=14, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                print(f"  ⚠️  WARNING: tgt_sst is IDENTICAL to aasti_av!")
            else:
                diff = np.abs(tgt_data - aasti_data)
                n_diff = np.sum(diff > 0.01)
                pct_diff = 100 * n_diff / diff.size
                fig.text(0.5, 0.02,
                        f'✅ OK: tgt_sst differs from aasti_av ({pct_diff:.1f}% of pixels differ)',
                        ha='center', fontsize=12, color='green', fontweight='bold')
                print(f"  ✅ OK: tgt_sst differs from aasti_av ({pct_diff:.1f}% of pixels)")

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/01b_fusion_integrity_check.png", dpi=150, bbox_inches='tight')
print(f"Saved: 01b_fusion_integrity_check.png")
plt.close()





fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Precomputed vs Pooled Mode (x10 resolution)', 
             fontsize=16, fontweight='bold')

var_key = 'slstr_av'
patch_key_precomp = 'patch_x10'
patch_key_pooled = 'patch_x10'

# Precomputed
if patch_key_precomp in sample_precomp:
    patch_p = sample_precomp[patch_key_precomp]
    if hasattr(patch_p, var_key):
        data_p = getattr(patch_p, var_key)[t_idx]
        ax = axes[0, 0]
        im = ax.imshow(data_p, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
        ax.set_title('Precomputed (direct x10 file)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='SST (°C)')
        
        ax = axes[1, 0]
        valid = ~np.isnan(data_p)
        im = ax.imshow(valid, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Precomputed Coverage: {valid.sum()/valid.size*100:.1f}%', fontsize=12)
        plt.colorbar(im, ax=ax, label='Valid data')

# Pooled
if patch_key_pooled in sample_pooled:
    patch_l = sample_pooled[patch_key_pooled]
    if hasattr(patch_l, var_key):
        data_l = getattr(patch_l, var_key)[t_idx]
        ax = axes[0, 1]
        im = ax.imshow(data_l, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
        ax.set_title('Pooled (x1 pooled x10)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='SST (°C)')
        
        ax = axes[1, 1]
        valid = ~np.isnan(data_l)
        im = ax.imshow(valid, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Pooled Coverage: {valid.sum()/valid.size*100:.1f}%', fontsize=12)
        plt.colorbar(im, ax=ax, label='Valid data')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_precomputed_vs_pooled.png", dpi=150, bbox_inches='tight')
print(f"Saved: 02_precomputed_vs_pooled.png")
plt.close()





fig, ax = plt.subplots(1, 1, figsize=(10, 6))

modes = ['Precomputed\n(x3/x10 direct)', 'Pooled\n(x1 pooled)']
init_times = [t_init_precomp, t_init_pooled]
sample_times = [t_sample_precomp, t_sample_pooled]

x = np.arange(len(modes))
width = 0.35

bars1 = ax.bar(x - width/2, init_times, width, label='Init time', color='skyblue')
bars2 = ax.bar(x + width/2, sample_times, width, label='Sample extract time', color='coral')

ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Performance: Precomputed vs Pooled Mode', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(modes)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_performance_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved: 03_performance_comparison.png")
plt.close()





fig, axes = plt.subplots(3, 5, figsize=(20, 12))
fig.suptitle('Temporal Evolution - SLSTR at x1, x3, x10 (Precomputed Mode)', 
             fontsize=16, fontweight='bold')

var_key = 'slstr_av'
for t in range(min(5, 5)):
    # x1
    if 'patch_x1' in sample_precomp:
        patch = sample_precomp['patch_x1']
        if isinstance(patch, dict):
            if var_key in patch:
                data = patch[var_key][t]
                ax = axes[0, t]
                im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
                ax.set_title(f't={t} (x1)', fontsize=10)
                ax.axis('off')
        elif hasattr(patch, var_key):
            data = getattr(patch, var_key)[t]
            ax = axes[0, t]
            im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
            ax.set_title(f't={t} (x1)', fontsize=10)
            ax.axis('off')
    
    # x3
    if 'patch_x3' in sample_precomp:
        patch = sample_precomp['patch_x3']
        if isinstance(patch, dict):
            if var_key in patch:
                data = patch[var_key][t]
                ax = axes[1, t]
                im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
                ax.set_title(f't={t} (x3)', fontsize=10)
                # Draw x1 bounds on x3 plot
                if bounds.get(1):
                    pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                    draw_rectangle_on_ax(ax, pixel_bounds, label='x1', color='cyan', linewidth=1.5)
                ax.axis('off')
        elif hasattr(patch, var_key):
            data = getattr(patch, var_key)[t]
            ax = axes[1, t]
            im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
            ax.set_title(f't={t} (x3)', fontsize=10)
            # Draw x1 bounds on x3 plot
            if bounds.get(1):
                pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                draw_rectangle_on_ax(ax, pixel_bounds, label='x1', color='cyan', linewidth=1.5)
            ax.axis('off')
    
    # x10
    if 'patch_x10' in sample_precomp:
        patch = sample_precomp['patch_x10']
        if isinstance(patch, dict):
            if var_key in patch:
                data = patch[var_key][t]
                ax = axes[2, t]
                im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
                ax.set_title(f't={t} (x10)', fontsize=10)
                # Draw x3 and x1 bounds on x10 plot
                if bounds.get(3):
                    pixel_bounds = pixel_bounds_from_coords(patch, bounds[3])
                    draw_rectangle_on_ax(ax, pixel_bounds, label='x3', color='lime', linewidth=1.5)
                if bounds.get(1):
                    pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                    draw_rectangle_on_ax(ax, pixel_bounds, label='x1', color='cyan', linewidth=1.5)
                ax.axis('off')
        elif hasattr(patch, var_key):
            data = getattr(patch, var_key)[t]
            ax = axes[2, t]
            im = ax.imshow(data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
            ax.set_title(f't={t} (x10)', fontsize=10)
            # Draw x3 and x1 bounds on x10 plot
            if bounds.get(3):
                pixel_bounds = pixel_bounds_from_coords(patch, bounds[3])
                draw_rectangle_on_ax(ax, pixel_bounds, label='x3', color='lime', linewidth=1.5)
            if bounds.get(1):
                pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                draw_rectangle_on_ax(ax, pixel_bounds, label='x1', color='cyan', linewidth=1.5)
            ax.axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_temporal_evolution_multires.png", dpi=150, bbox_inches='tight')
print(f"Saved: 04_temporal_evolution_multires.png")
plt.close()






fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Satellite Coverage by Resolution (Precomputed Mode, t=2)', 
             fontsize=16, fontweight='bold')

t_idx = 2
satellites = ['aasti', 'avhrr', 'pmw', 'slstr']
resolutions = [('patch_x1', 'x1'), ('patch_x3', 'x3'), ('patch_x10', 'x10')]

for res_idx, (patch_key, res_label) in enumerate(resolutions):
    if patch_key not in sample_precomp:
        continue
    patch = sample_precomp[patch_key]
    
    for sat_idx, sat in enumerate(satellites):
        ax = axes[res_idx, sat_idx]
        var_key = f"{sat}_av"
        
        # Handle both dict and TrainingItem
        if isinstance(patch, dict):
            if var_key in patch:
                data = patch[var_key][t_idx]
                valid_mask = ~np.isnan(data)
                coverage = valid_mask.sum() / valid_mask.size * 100
                
                im = ax.imshow(valid_mask, cmap='Greys', origin='lower', vmin=0, vmax=1)
                ax.set_title(f'{sat.upper()} ({res_label})\nCov: {coverage:.1f}%', 
                            fontsize=10)
                # Draw nested patch rectangles on x3 and x10 rows
                if res_label == 'x3' and bounds.get(1):
                    pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                    draw_rectangle_on_ax(ax, pixel_bounds, color='cyan', linewidth=1.5)
                elif res_label == 'x10':
                    if bounds.get(3):
                        pixel_bounds = pixel_bounds_from_coords(patch, bounds[3])
                        draw_rectangle_on_ax(ax, pixel_bounds, color='lime', linewidth=1.5)
                    if bounds.get(1):
                        pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                        draw_rectangle_on_ax(ax, pixel_bounds, color='cyan', linewidth=1.5)
                ax.axis('off')
        elif hasattr(patch, var_key):
            data = getattr(patch, var_key)[t_idx]
            valid_mask = ~np.isnan(data)
            coverage = valid_mask.sum() / valid_mask.size * 100
            
            im = ax.imshow(valid_mask, cmap='Greys', origin='lower', vmin=0, vmax=1)
            ax.set_title(f'{sat.upper()} ({res_label})\nCov: {coverage:.1f}%', 
                        fontsize=10)
            # Draw nested patch rectangles on x3 and x10 rows
            if res_label == 'x3' and bounds.get(1):
                pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                draw_rectangle_on_ax(ax, pixel_bounds, color='cyan', linewidth=1.5)
            elif res_label == 'x10':
                if bounds.get(3):
                    pixel_bounds = pixel_bounds_from_coords(patch, bounds[3])
                    draw_rectangle_on_ax(ax, pixel_bounds, color='lime', linewidth=1.5)
                if bounds.get(1):
                    pixel_bounds = pixel_bounds_from_coords(patch, bounds[1])
                    draw_rectangle_on_ax(ax, pixel_bounds, color='cyan', linewidth=1.5)
            ax.axis('off')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_satellite_coverage_multires.png", dpi=150, bbox_inches='tight')
print(f"Saved: 05_satellite_coverage_multires.png")
plt.close()
