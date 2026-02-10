# Tools

Standalone utility scripts for analysis, visualization, and data integrity checks.

## Scripts

| Script | What it does |
|---|---|
| `analyze_timings.py` | Parses training logs to analyze step duration and throughput |
| `plot_zarr.py` | Visualizes variables from Zarr daily files (all 4 sensors, surfmask, etc.) |
| `plot_multires.py` | Plots multi-resolution patch comparisons (x1 vs x3 vs x10) |
| `check_files.sh` | Verifies that all expected data files are present for a given year |
| `check_surfmask.py` | Checks surface mask consistency across Zarr files |
| `check_zarr_chunks.py` | Inspects Zarr chunking structure and reports chunk sizes |
