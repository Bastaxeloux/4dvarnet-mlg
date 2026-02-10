# Tests

Standalone validation and debugging scripts. These are **not** a pytest suite — run them individually with `python tests/<script>.py`.

Most scripts require the conda environment to be activated and some expect access to the data at `/nwp/sst_malegu/`.

## Scripts

### Model & architecture tests
| Script | What it validates |
|---|---|
| `test_dynamic_prior.py` | The dynamic prior cost Phi(state) evolves correctly during optimization steps |
| `test_multires_fix.py` | Multi-resolution alignment after DAW cropping and interpolation |

### Data & patch tests
| Script | What it validates |
|---|---|
| `test_encompassing_patches.py` | Multi-res patches correctly nest geographically (x1 inside x3 inside x10) |
| `test_nested_patches.py` | Geometry of nested patches at different resolutions |
| `test_single_day_mode.py` | Single-day test inference mode works correctly |
| `test_specific_train_sample.py` | Loads and inspects a specific training sample for debugging |
| `test_multiresolution_loader.py` | The multi-res DataLoader produces correct batch structure |

### Performance & memory tests
| Script | What it validates |
|---|---|
| `test_chunk_sizes.py` | Impact of Zarr chunk sizes on I/O performance |
| `test_pooling_speed.py` | Benchmarks pooling operations (CPU vs GPU, different methods) |
| `test_memory_load.py` | Profiles memory usage during data loading |
| `test_worker_memory.py` | Profiles DataLoader worker memory consumption |

### End-to-end tests
| Script | What it validates |
|---|---|
| `test_sst_with_plots.py` | Full SST pipeline test with visualization output |
| `test_sst_multires_with_plots.py` | Full multi-res pipeline test with visualization output |

### I/O tests
| Script | What it validates |
|---|---|
| `test_interpolation.py` | Spatial interpolation accuracy between resolutions |
| `test_zarr_zip_store.py` | Zarr zip store backend as alternative to directory store |
