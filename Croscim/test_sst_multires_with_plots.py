import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.append('.')
import glob
from contrib.SST.data_multires import BaseDataModuleMultiRes, plot_multires
import yaml
import tempfile

# ============================================================
# Configuration et chargement des données
# ============================================================
DATA_DIR = "/dmidata/users/malegu/data/netcdf_2024"
OUTPUT_DIR = "figs/SST_multires"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PATCH_INDICES = {'slstr_av': 0, 'aasti_av': 0, 'tgt_sst': 0}
T_IDX = 7  # Timestep central par défaut

with open('contrib/SST/norm_stats.yaml', 'r') as f:
    norm_stats = yaml.safe_load(f)['norm_stats']

sst_files = sorted(glob.glob(f"{DATA_DIR}/*.nc"))
print(f"\nFound {len(sst_files)} SST files")
print(f"Using first 15 files (15 days)")

patch_dims = {'time': 15, 'lat': 256, 'lon': 256}
strides = {'time': 7, 'lat': 40, 'lon': 40}
multires = [10, 3, 1]

norm_stats_covs = {'sea_ice_fraction': {'type': 'minmax', 'min': 0.0, 'max': 1.0}}
tmp_patch_dir = tempfile.mkdtemp()

print(f"\nCréation DataModule multi-résolution {multires}...")
dm = BaseDataModuleMultiRes(
    sst_paths=sst_files[:15],
    multires=multires,
    xrds_kw={'patch_dims': patch_dims, 'strides': strides, 'subsel_patch': False},
    tgt_vars=['slstr_av', 'aasti_av'],
    norm_stats=norm_stats,
    norm_stats_covs=norm_stats_covs,
    dl_kw={'batch_size': 1, 'num_workers': 0}
)
dm.subsel_path = tmp_patch_dir
dm.setup(stage='train')
train_loader = dm.train_dataloader()

# Charger plusieurs patches pour avoir le choix
print("\nChargement de patches...")
all_batches = []
for i, batch_dict in enumerate(train_loader):
    if i >= max(PATCH_INDICES.values()) + 1:
        break
    all_batches.append(batch_dict)

    patch_x1 = batch_dict['patch_x1']
    if hasattr(patch_x1, 'lat'):
        lat = patch_x1.lat
        if lat.ndim > 0:
            lat_center = (lat.min().item() + lat.max().item()) / 2
        else:
            lat_center = lat.item()
        print(f"  Patch {i}: lat center = {lat_center:6.1f}°")

print(f"\nNombre de patches chargés: {len(all_batches)}")
resolution_colors = {1: 'red', 3: 'green', 10: 'blue'}

print(f"\nPlot (patch index={PATCH_INDICES['slstr_av']}, t_idx={T_IDX})")
batch_main = all_batches[PATCH_INDICES['slstr_av']]
plot_multires(
    batch_dict=batch_main,
    var=['slstr_av', 'aasti_av', 'tgt_sst'],
    t_idx=T_IDX,
    resolution_colors=resolution_colors,
    title='SST Multi-Resolution Comparison',
    save_dir=OUTPUT_DIR
)

