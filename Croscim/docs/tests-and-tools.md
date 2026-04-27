# Tests and Tools

Tests and utilities are useful but not uniform. Many assume access to the DMI
data root `/nwp/sst_malegu`.

## Tests

Run tests individually:

```bash
python3 tests/<script>.py
```

Some tests are Hydra-aware:

```bash
python3 tests/test_dynamic_prior.py xp=SST/multires_lite
```

There is no pytest configuration. Treat these as standalone validation and
debug scripts.

## Test Categories

| Category | Scripts |
|---|---|
| Model and architecture | `test_dynamic_prior.py`, `test_multires_fix.py` |
| Patch geometry | `test_encompassing_patches.py`, `test_nested_patches.py`, `test_single_day_mode.py`, `test_specific_train_sample.py`, `test_multiresolution_loader.py` |
| Performance and memory | `test_chunk_sizes.py`, `test_pooling_speed.py`, `test_memory_load.py`, `test_worker_memory.py` |
| End-to-end plotting | `test_sst_with_plots.py`, `test_sst_multires_with_plots.py` |
| I/O and interpolation | `test_interpolation.py`, `test_zarr_zip_store.py` |

## Test Caveats

- Most tests require real Zarr data under `/nwp/sst_malegu`.
- Plot tests can write under `figs/`.
- Performance tests can create temporary Zarr outputs.
- Some scripts still contain hard-coded local paths.
- Use `python3` on machines where `python` is not available.

## Data Preprocessing

The data pipeline converts raw satellite archives to x1 Zarr and then computes
x3/x10 stores.

Main batch scripts:

```bash
data/process_all_years.sh
data/process_all_years_gefion.sh
```

Important single-step scripts:

```bash
python3 data/converter.py YEAR --parallel N --zarr-output-dir DIR
python3 data/compute_res_yearly.py YEAR --parallel N --save-format zarr --output-dir DIR
python3 data/compute_res_daily.py INPUT_X1 -o DIR --format zarr
python3 data/fix_surfmask.py DATA_DIR --dry-run
python3 data/zarr_viewer.py FILE.zarr
```

Check each script before running: some contain source-path, output-path, or
cleanup assumptions.

## Tools

Utility scripts under `tools/`:

| Script | Purpose |
|---|---|
| `analyze_timings.py` | Parse training timing logs |
| `plot_zarr.py` | Visualize variables from a daily Zarr store |
| `plot_multires.py` | Compare x1/x3/x10 patches |
| `check_files.sh` | Check expected files for a year |
| `check_surfmask.py` | Inspect surface mask consistency |
| `check_zarr_chunks.py` | Inspect Zarr chunking |

Several tools still contain hard-coded paths. Prefer passing explicit paths
when supported and inspect the script before using it for production checks.

## Notebooks

`archive/Notebooks/` and root `Notebooks/` contain legacy exploratory notebooks.
They are not maintained runbooks and may contain stale paths or embedded
outputs.
