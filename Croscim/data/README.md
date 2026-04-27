# data

Preprocessing scripts for converting raw satellite SST data to the Zarr layout
used by training.

Pipeline:

```text
raw daily archive -> extract/verify -> x1 Zarr -> x3/x10 Zarr
```

Important scripts:

- `converter.py`: convert extracted daily data to x1 Zarr.
- `compute_res_yearly.py`: build x3/x10 stores for a year.
- `compute_res_daily.py`: build x3/x10 for one daily x1 file.
- `process_all_years.sh`: local batch pipeline.
- `process_all_years_gefion.sh`: Gefion batch pipeline.
- `fix_surfmask.py`: inspect/fix surface mask issues.
- `verif_fichiers.py`: check expected files.
- `zarr_viewer.py`: inspect Zarr variables, shapes, chunks, stats.

Some scripts have machine-specific paths and cleanup assumptions. See
[../docs/tests-and-tools.md](../docs/tests-and-tools.md) and
[../docs/data.md](../docs/data.md).
