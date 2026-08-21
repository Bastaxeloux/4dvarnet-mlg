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
- `process_year_gefion.slurm`: Gefion SLURM pipeline for one SQFS year.
- `process_year_jeanzay.slurm`: Jean Zay CPU pipeline for one SQFS year.
- `submit_years_jeanzay.sh`: submit one or more independent Jean Zay years.
- `process_all_years_gefion.sh`: older Gefion batch pipeline; inspect before
  use because it has historical path assumptions.
- `fix_surfmask.py`: inspect/fix surface mask issues.
- `verif_fichiers.py`: check expected files.
- `zarr_viewer.py`: inspect Zarr variables, shapes, chunks, stats.

On Gefion, SQFS archives should live in
`/dcai/projects/cu_0026/data_sst/sqfs`. The `xfer/guimae/inbox` directory is
only for transfer staging.

The currently validated training range is 2017–2024. Years 2014–2015 lack
SLSTR and 2016 has incomplete SLSTR coverage, so generated daily Zarr counts do
not make those years scientifically valid.

Some scripts have machine-specific paths and cleanup assumptions. See
[../docs/tests-and-tools.md](../docs/tests-and-tools.md) and
[../docs/data.md](../docs/data.md).

On Jean Zay, keep the durable SQFS archives under `$STORE/croscim/sqfs` and
write extracted files and daily Zarr stores under `$SCRATCH/croscim`. The
Zarr layout contains too many small files for the default `$WORK` and `$STORE`
inode quotas.
