# Data Preprocessing

Scripts for converting raw satellite SST data into the multi-resolution Zarr format used for training.

## Pipeline overview

The raw data comes as daily squashfs archives from the CLS SST processing chain. Each archive contains 13+ variables (4 satellite sensors x {av, std}, sea_ice_fraction, surfmask, analysed_st, etc.).

The preprocessing pipeline converts these into Zarr stores at 3 resolutions:

```
squashfs archive  →  extract  →  verify  →  convert to x1 Zarr  →  compute x3 and x10
```

The output for one year looks like:
```
data_2024/
├── 2024010112_x1.zarr    # 5 km native resolution (3600 x 7200)
├── 2024010112_x3.zarr    # 15 km coarsened (1200 x 2400)
├── 2024010112_x10.zarr   # 50 km coarsened (360 x 720)
├── 2024010200_x1.zarr
├── ...
```

## Batch processing

| Script | Machine | What it does |
|---|---|---|
| `process_all_years.sh` | DMI local (`/dmidata/`) | Runs the full pipeline for all years (2010-2025) sequentially. Uses 10 CPU cores. |
| `process_all_years_gefion.sh` | Gefion (`/dcai/`) | Same pipeline for Gefion cluster. Uses 48 CPU cores. |

## Processing scripts

| Script | What it does |
|---|---|
| `converter.py` | Converts extracted daily files into x1 Zarr format. Handles variable mapping, chunking (768x768), and metadata. |
| `compute_res_yearly.py` | Generates x3 and x10 resolution Zarr files from x1 by spatial pooling. |
| `compute_res_daily.py` | Same as above but processes one day at a time. |
| `fix_surfmask.py` | Fixes surface mask inconsistencies in Zarr files. |
| `ajout_ascii_manquant.py` | Fills in missing ASCII reference data in extracted archives. |
| `rename_to_x1.py` | Renames original Zarr files to follow the `_x1.zarr` naming convention. |
| `verif_fichiers.py` | Verifies that all expected files are present and readable for a given year. |
| `zarr_viewer.py` | Interactive Zarr file inspector — prints variables, shapes, chunks, and basic statistics. |

## Data locations

| Machine | Path | Contents |
|---|---|---|
| DMI local | `/nwp/sst_malegu/data_YYYY/` | Preprocessed Zarr files (2010-2025) |
| Gefion | `/dcai/projects/cu_0026/data_sst/data_YYYY/` | Same data on Gefion storage |
