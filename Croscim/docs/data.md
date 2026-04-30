# Data

This document describes the active SST data contract.

## Data Roots

Known roots:

| Machine | Path | Contents |
|---|---|---|
| DMI/Ohm | `/nwp/sst_malegu/data_YYYY/` | Daily SST Zarr files |
| Gefion | `/dcai/projects/cu_0026/data_sst/data_YYYY/` | Same layout on HPC storage |
| Gefion | `/dcai/projects/cu_0026/data_sst/sqfs/` | Durable SQFS archive storage |

`/dcai/projects/cu_0026/xfer/guimae/inbox` is only the transfer inbox. Move
archives to `data_sst/sqfs/` before preprocessing and do not point processing
jobs at `xfer`.

Daily files are expected to follow:

```text
YYYYMMDD12_x1.zarr
YYYYMMDD12_x3.zarr
YYYYMMDD12_x10.zarr
```

Approximate grids:

| Resolution | Scale | Shape |
|---|---:|---:|
| x1 | 5 km | 3600 x 7200 |
| x3 | 15 km | 1200 x 2400 |
| x10 | 50 km | 360 x 720 |

Training patches are 256 x 256 pixels over a 15-day temporal window.

## Gefion Preprocessing

Run preprocessing on a compute node, not on a login node:

```bash
sbatch data/process_year_gefion.slurm 2022
```

The job reads one archive from `data_sst/sqfs/`, extracts it temporarily under
`data_sst/squash_YYYY_extract`, writes Zarr files to `data_sst/data_YYYY`, and
removes the extraction after a complete year succeeds.

## Variables

Satellite groups are defined in `contrib/SST/load_data.py`:

| Group | Variables | Notes |
|---|---|---|
| `slstr` | `av`, `std` | Sentinel-3 SLSTR, high accuracy, sparse at poles |
| `aasti` | `av`, `std` | AATSR/Envisat heritage, useful in polar regions |
| `avhrr` | `av`, `std` | Good global coverage |
| `pmw` | `av`, `std` | Near-global, smoother passive microwave source |

Covariates:

- `sea_ice_fraction`

Other important fields:

- `surfmask`
- `lat`, `lon`
- `lat_geo`, `lon_geo`
- `time`
- `time_indices`

## Target Fusion

`tgt_sst` is built from SLSTR and AASTI:

- if `sea_ice_fraction >= 0.70`, use AASTI
- otherwise, use SLSTR

During self-supervised learning:

- `tgt_sst_full` is the complete fused target before artificial masking.
- `tgt_sst` is the masked fused input seen by the solver.
- `inpaint_mask` marks pixels removed by artificial masking.

This distinction is important. Losses should evaluate against `tgt_sst_full`
when available, while the solver input starts from masked `tgt_sst`.

## Normalization

Normalization stats live in config YAML and in `contrib/SST/norm_stats.yaml`.
Satellite variables use z-score normalization. `sea_ice_fraction` uses min-max
normalization.

`contrib/SST/norm_stats.txt` is not the source of truth; treat it as a static
reference or historical export.

## Patch Filtering

Training patch validation rejects low-value samples using `is_valid_patch()`:

- valid data ratio must be at least 8 percent
- variance must be at least 0.05
- ocean/ice ratio must be at least 5 percent

The multi-resolution training dataset retries rejected samples up to the current
retry limit before keeping a patch with a warning.

Patch filtering is disabled in test mode because test reconstruction should
cover the full target domain, including sparse or mostly empty patches.

## Multi-Resolution Nesting

Training extraction starts from the finest resolution:

1. Extract x1 patch from the normal index grid.
2. Find an x3 patch that geographically contains x1.
3. Find an x10 patch that geographically contains x3.

The geometry logic is in:

- `contrib/SST/data_multires.py`
- `src/utils.py`

## Coordinate Pitfalls

- `lat` and `lon` are normalized channels used by the model.
- `lat_geo` and `lon_geo` are real geographic coordinates in degrees.
- Interpolation between resolutions must use `lat_geo` and `lon_geo`, not the
  normalized channels.

## Dask and Zarr

xarray plus Dask plus PyTorch multiprocessing can deadlock. The active data
path avoids xarray inside hot `__getitem__()` loading where possible and uses
pure Zarr reads. Dask scheduler is set to `synchronous` in datamodule setup and
worker initialization.
