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
DATA_DIR = "/home/malegu/4D-MLG/Croscim/data/mounted/2024"
OUTPUT_DIR = "figs/SST"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("TEST ET VISUALISATION")

# Load norm_stats
with open('contrib/SST/norm_stats.yaml', 'r') as f:
    norm_stats_file = yaml.safe_load(f)
    norm_stats = norm_stats_file['norm_stats']
sst_files = sorted(glob.glob(f"{DATA_DIR}/*_x1.zarr"))
print(f"\nFound {len(sst_files)} SST files in {DATA_DIR}")
times = np.arange(15)

patch_dims = {'time': 15, 'lat': 768, 'lon': 768}
strides = {'time': 1, 'lat': 768, 'lon': 768}

dataset = XrDataset(
    sst_daily_paths=sst_files[:30], 
    tgt_vars=['slstr_av', 'aasti_av'],
    mask=None,
    times=times,
    patch_dims=patch_dims,
    strides=strides,
    postpro_fn=None,
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
plt.suptitle('Patch at t=7 of size 768x768', fontsize=16, fontweight='bold')
t_idx = 7
for i, sat in enumerate(['aasti', 'avhrr', 'pmw', 'slstr']):
    var_key = f"{sat}_av"
    if var_key in sample:
        data = sample[var_key][t_idx]
        
        # Average value
        ax = axes[0, i]
        im = ax.imshow(data, cmap='RdYlBu_r', origin='lower')
        ax.set_title(f'{sat.upper()} - Average', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='SST (°C)')
        
        # Valid data mask
        ax = axes[1, i]
        valid_mask = ~np.isnan(data)
        coverage = valid_mask.sum() / valid_mask.size * 100
        im = ax.imshow(valid_mask, cmap='Greys', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'{sat.upper()} - Coverage: {coverage:.1f}%', fontsize=12)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Valid data')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_satellite_coverage.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/01_satellite_coverage.png")
plt.close()


print("\nTARGET FUSION (SLSTR + AASTI)")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Target Fusion', fontsize=16, fontweight='bold')
t_idx = 7
# SLSTR data
ax = axes[0, 0]
slstr_data = sample['slstr_av'][t_idx]
im = ax.imshow(slstr_data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('SLSTR', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
# AASTI data
ax = axes[0, 1]
aasti_data = sample['aasti_av'][t_idx]
im = ax.imshow(aasti_data, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('AASTI', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
# Fused target
ax = axes[1, 0]
tgt_sst = sample['tgt_sst'][t_idx]
im = ax.imshow(tgt_sst, cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('Fusion (slstr where valid, else aasti)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='SST (°C)')
# Coverage map
ax = axes[1, 1]
slstr_valid = ~np.isnan(slstr_data)
aasti_valid = ~np.isnan(aasti_data)
tgt_valid = ~np.isnan(tgt_sst)

# Créer une carte avec NaN pour "No data" (apparaîtra blanc)
coverage_map = np.full_like(tgt_sst, np.nan)
coverage_map[slstr_valid] = 1  # SLSTR: bleu
coverage_map[~slstr_valid & aasti_valid] = 2  # AASTI : rouge
from matplotlib.colors import ListedColormap
colors = ['#3498db', '#e74c3c']
cmap = ListedColormap(colors)

im = ax.imshow(coverage_map, cmap=cmap, origin='lower', vmin=1, vmax=2)
ax.set_title('Fusion Map', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, ticks=[1.25, 1.75])
cbar.ax.set_yticklabels(['SLSTR', 'AASTI'])

# Ajouter stats dans un coin
slstr_pct = slstr_valid.sum() / slstr_valid.size * 100
aasti_only_pct = (~slstr_valid & aasti_valid).sum() / aasti_valid.size * 100
no_data_pct = (~tgt_valid).sum() / tgt_valid.size * 100
ax.text(0.02, 0.98, f'SLSTR: {slstr_pct:.1f}%\nAASTI: {aasti_only_pct:.1f}%\nNo data: {no_data_pct:.1f}%',
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_target_fusion.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/02_target_fusion.png")
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
ax.set_title(f"Day {(sample['time'].argmax()+1)%366}", fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized time')

# Surface mask, its not binary it has multiple values
ax = axes[1, 1]
im = ax.imshow(sample['surfmask'], cmap='viridis', origin='lower')
ax.set_title('Surface Mask (0=Land, 1=Ocean, 2=Interface Water/ice, 3=Ice)', fontsize=12, fontweight='bold')
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

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_temporal_evolution.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/05_temporal_evolution.png")
plt.close()





print("\nVRAI TRAININGITEM")
norm_stats_covs = {
    'sea_ice_fraction': {'type': 'minmax', 'min': 0.0, 'max': 1.0}
}

# Créer une instance de BaseDataModule
dm = BaseDataModule(
    sst_paths=sst_files,
    covariates_paths=[], covariates=COVARIATES,
    tgt_vars=['slstr_av', 'aasti_av'],
    mask_path='dummy.nc',
    domain_name='test', domains={},
    xrds_kw={}, dl_kw={},
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs
)
dm.tgt_vars = ['slstr_av', 'aasti_av']
dataset.postpro_fn = dm.post_fn(rand_obs=True)  
training_item = dataset[0]
print(f"TrainingItem chargé (type: {type(training_item).__name__})")

# Quelques vérifications essentielles
aasti_av_inpainted = training_item.aasti_av
slstr_av_inpainted = training_item.slstr_av
inpaint_mask_real = training_item.inpaint_mask

print(f"Inpainting appliqué: {(inpaint_mask_real == 1).sum() / inpaint_mask_real.size * 100:.1f}% pixels supprimés")
print(f"Normalisation: valeurs dans [{np.nanmin(training_item.tgt_sst):.2f}, {np.nanmax(training_item.tgt_sst):.2f}]")



###############################################################
# FIGURE 6 : Comparaison avant/après normalisation + inpainting
################################################################

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
fig.suptitle('TrainingItem Real: Normalisation + Inpainting', fontsize=16, fontweight='bold')

t_idx = 7

# Row 1: Original vs Normalized/Inpainted
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(sample['slstr_av'][t_idx], cmap='RdYlBu_r', origin='lower', vmin=-5, vmax=30)
ax.set_title('SLSTR Original (°C)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(sample['aasti_av'][t_idx], cmap='RdYlBu_r', origin='lower', vmin=-25, vmax=10)
ax.set_title('AASTI Original (°C)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[0, 2])
tgt_sst_norm = training_item.tgt_sst[t_idx]
im = ax.imshow(tgt_sst_norm, cmap='RdYlBu_r', origin='lower')
ax.set_title('Target SST Normalisé (fusion)', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)

# Row 2: After processing
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(slstr_av_inpainted[t_idx], cmap='RdYlBu_r', origin='lower')
ax.set_title('SLSTR après Inpaint et Normalisation', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(aasti_av_inpainted[t_idx], cmap='RdYlBu_r', origin='lower')
ax.set_title('AASTI après Inpaint et Normalisation', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[1, 2])
# Couverture temporelle : combien de timesteps ont des données valides
temporal_coverage = np.sum(~np.isnan(training_item.tgt_sst), axis=0)  # (nlat, nlon)
im = ax.imshow(temporal_coverage, cmap='viridis', origin='lower', vmin=0, vmax=patch_dims['time'])
ax.set_title(f'Couverture temporelle (sur {patch_dims["time"]} timesteps)', fontsize=11, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Nombre de timesteps valides')

plt.savefig(f"{OUTPUT_DIR}/06_trainingitem_validation.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/06_trainingitem_validation.png")
plt.close()




#################################################################
# Figure 7: Vérification de la cohérence spatiale de l'inpainting
#################################################################
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
t_idx = 7

# Échelle commune pour SLSTR et AASTI (pour comparaison directe)
vmin_common, vmax_common = -25, 30

# Colonne 1: SLSTR
ax = axes[0, 0]
im = ax.imshow(sample['slstr_av'][t_idx], cmap='RdYlBu_r', origin='lower', vmin=vmin_common, vmax=vmax_common)
ax.set_title('SLSTR Original (°C)', fontsize=12)
plt.colorbar(im, ax=ax, label='SST (°C)')

ax = axes[1, 0]
slstr_denorm = slstr_av_inpainted[t_idx] * norm_stats['slstr']['av']['std'] + norm_stats['slstr']['av']['mean']
im = ax.imshow(slstr_denorm, cmap='RdYlBu_r', origin='lower', vmin=vmin_common, vmax=vmax_common)
ax.set_title('SLSTR Inpainté (dénorm)', fontsize=12)
plt.colorbar(im, ax=ax, label='SST (°C)')

# Colonne 2: AASTI
ax = axes[0, 1]
im = ax.imshow(sample['aasti_av'][t_idx], cmap='RdYlBu_r', origin='lower', vmin=vmin_common, vmax=vmax_common)
ax.set_title('AASTI Original (°C)', fontsize=12)
plt.colorbar(im, ax=ax, label='SST (°C)')

ax = axes[1, 1]
aasti_denorm = aasti_av_inpainted[t_idx] * norm_stats['aasti']['av']['std'] + norm_stats['aasti']['av']['mean']
im = ax.imshow(aasti_denorm, cmap='RdYlBu_r', origin='lower', vmin=vmin_common, vmax=vmax_common)
ax.set_title('AASTI Inpainté (dénorm)', fontsize=12)
plt.colorbar(im, ax=ax, label='SST (°C)')

# Colonne 3: Masques
ax = axes[0, 2]
slstr_removed = ~np.isnan(sample['slstr_av'][t_idx]) & np.isnan(slstr_denorm)
im = ax.imshow(slstr_removed, cmap='Reds', origin='lower', vmin=0, vmax=1)
ax.set_title(f'SLSTR: pixels supprimés ({slstr_removed.sum()})', fontsize=12)
plt.colorbar(im, ax=ax)

ax = axes[1, 2]
aasti_removed = ~np.isnan(sample['aasti_av'][t_idx]) & np.isnan(aasti_denorm)
im = ax.imshow(aasti_removed, cmap='Reds', origin='lower', vmin=0, vmax=1)
ax.set_title(f'AASTI: pixels supprimés ({aasti_removed.sum()})', fontsize=12)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_inpainting_coherence.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/07_inpainting_coherence.png")
plt.close()





# Créer un dossier séparé
ATTR_DIR = f"{OUTPUT_DIR}/trainingitem_attributes"
Path(ATTR_DIR).mkdir(parents=True, exist_ok=True)
t_idx = 7
all_attrs = training_item._fields
print(f"\nPlotting {len(all_attrs)} attributs à t={t_idx}...")
for attr_name in all_attrs:
    attr_data = getattr(training_item, attr_name)
    if not isinstance(attr_data, np.ndarray):
        continue
    if attr_data.ndim == 3:
        data_2d = attr_data[t_idx]
    elif attr_data.ndim == 2:
        data_2d = attr_data
    else:
        continue
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if 'mask' in attr_name.lower(): cmap = 'viridis'
    elif 'inpaint' in attr_name.lower(): cmap = 'Reds'
    else: cmap = 'RdYlBu_r'
    im = ax.imshow(data_2d, cmap=cmap, origin='lower')
    ax.set_title(f'{attr_name} (t={t_idx})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, label=attr_name)
    
    # Stats dans le titre
    valid_data = data_2d[~np.isnan(data_2d)] if np.any(np.isnan(data_2d)) else data_2d
    if len(valid_data) > 0:
        ax.text(0.02, 0.98, f'min={np.min(valid_data):.2f}, max={np.max(valid_data):.2f}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{ATTR_DIR}/{attr_name}.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"OK {attr_name}.png")

print(f"\nTous les attributs sauvegardés dans: {ATTR_DIR}/")
print("="*80)
plt.close()