# Data

This document describes the active SST data contract.

## Data Roots

Known roots:

| Machine | Path | Contents |
|---|---|---|
| DMI/Ohm | `/nwp/sst_malegu/data_YYYY/` | Daily SST Zarr files |
| Gefion | `/dcai/projects/cu_0026/data_sst/data_YYYY/` | Same layout on HPC storage |
| Gefion | `/dcai/projects/cu_0026/data_sst/sqfs/` | Durable SQFS archive storage |
| Jean Zay | `$STORE/croscim/sqfs/` | Durable SQFS archive storage |
| Jean Zay | `$SCRATCH/croscim/sqfs/` | SQFS staging visible from compute nodes |
| Jean Zay | `$SCRATCH/croscim/data_sst/data_YYYY/` | Daily SST Zarr files |

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

The script defaults to `defq`. Do not force `-p cpuq` unless the Python module
stack has been changed: `SciPy-bundle/2023.07` can load a NumPy build requiring
`AVX512_SPR`, which failed on at least one `cpuq` node during 2022
preprocessing.

The job reads one archive from `data_sst/sqfs/`, extracts it temporarily under
`/dcai/projects/cu_0026/guimae/croscim/tmp/squash_YYYY_extract`, writes Zarr
files to `data_sst/data_YYYY`, and removes the extraction after a complete year
succeeds.

The currently validated complete range for expanded training is 2017–2024:
2922 daily x1/x3/x10 stores. Years 2014–2015 have no SLSTR input, while 2016
has incomplete SLSTR coverage and is excluded from the current experiment.
File counts alone are not sufficient validation because the missing-ASCII
fallback can still create one output per day; inspect preprocessing logs and
source availability.

## Jean Zay Preprocessing

The active Jean Zay pipeline reserves one `cpu_p1` node with 40 physical cores
and initially runs 32 preprocessing workers to leave RAM headroom for global
ASCII conversion. Memory is assigned automatically by Jean Zay; its SLURM
configuration rejects explicit `--mem` options. Archives remain in `$STORE`;
the `archive` partition stages them into `$SCRATCH/croscim/sqfs` because
`$STORE` is not mounted on `cpu_p1` nodes. Temporary extraction and Zarr output
also use `$SCRATCH` because a processed year contains many small files.

From the Croscim root, load and verify the environment:

```bash
source scripts/jeanzay/env.sh
python -c "import numpy,xarray,zarr,netCDF4,dask; print(zarr.__version__)"
```

Submit the complete publication range with one command:

```bash
bash data/submit_all_years_jeanzay.sh
squeue -u "$USER"
tail -F "$WORK"/croscim/logs/stage_sqfs_*.out
```

This creates two jobs. The first copies and validates all eight SQFS archives on
the `archive` partition. The second waits through an `afterok` dependency, then
processes 2017 through 2024 sequentially in one `cpu_p1` allocation. Follow the
second job with:

```bash
tail -F "$WORK"/croscim/logs/preprocess_all_JOB_ID.out
```

The job writes a run manifest under `$WORK/croscim/manifests` and retains an
extraction when processing fails. Set `KEEP_EXTRACT=1` only when an otherwise
successful extraction must be retained for inspection.

After the first complete year, inspect `MaxRSS` with `sacct`. Only increase to
40 workers if the observed peak leaves sufficient margin below the 192 GB node
memory:

```bash
NB_CORES=40 sbatch data/process_year_jeanzay.slurm YEAR
```

For the publication protocol, use 2017--2022 for training statistics and
training, full 2023 for model selection, and keep full 2024 sealed for the
final evaluation.

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

Normalization stats live in config YAML. The helper
`contrib/SST/compute_statistics.py` generates `contrib/SST/norm_stats.yaml` and
`contrib/SST/norm_stats.txt`; those generated files should be refreshed from
data, not hand-edited.

For residual x3/x1 training, mean-temperature fields (`slstr_av`, `aasti_av`,
`avhrr_av`, `pmw_av`, `tgt_sst`, `tgt_sst_full`) should use the shared
`sst_common` z-score scale computed from the raw fused `tgt_sst`. This keeps
residual subtractions physically consistent in the x10 -> x3 -> x1 cascade.
Satellite `_std` fields keep their own z-score stats. `sea_ice_fraction` uses
min-max normalization.

The `sst_common` values generated from 1000 sampled x1 files across 2017–2024
and copied into `multires_gefion_resunet.yaml` are:

```yaml
mean: 7.78095617993726
std: 20.58217902545633
type: zscore
```

If the training date range or source data changes substantially, regenerate
stats with `compute_statistics.py` and copy the generated values into the active
configs again.

## Patch Filtering

Training patch validation rejects low-value samples using `is_valid_patch()`:

- active configs set valid data ratio to at least 2 percent
- variance must be at least 0.05
- ocean/ice ratio must be at least 5 percent

The multi-resolution training dataset retries rejected samples up to the current
retry limit before keeping a patch with a warning.

Validation uses a fixed cached set under `val_set_dir`. The current builder
scans a fixed candidate budget, ranks the best candidates for visualization,
and samples the loss subset from candidates passing the low validation filter.
On Gefion, the cache is built by DDP rank 0 and then loaded by the other ranks.

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
