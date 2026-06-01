# tools

Standalone inspection, plotting, and analysis utilities.

Scripts:

- `analyze_timings.py`: parse timing logs.
- `plot_zarr.py`: visualize variables in daily Zarr files.
- `plot_multires.py`: compare x1/x3/x10 patches.
- `check_files.sh`: check expected files.
- `check_surfmask.py`: inspect surface masks.
- `check_zarr_chunks.py`: inspect chunking.

Several tools still include hard-coded paths. Inspect scripts before using them
as production checks.

See [../docs/tests-and-tools.md](../docs/tests-and-tools.md).
