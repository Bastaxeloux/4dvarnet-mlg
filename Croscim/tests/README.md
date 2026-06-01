# tests

Standalone validation and debugging scripts. This is not a pytest suite.

Run scripts individually:

```bash
python3 tests/<script>.py
```

Most tests require the conda environment and access to SST Zarr data under
`/nwp/sst_malegu`.

Categories:

- model and architecture checks
- multi-resolution patch geometry checks
- loader and single-day checks
- memory and performance profiling
- end-to-end plotting checks
- interpolation and Zarr backend checks

Some scripts create figures or temporary Zarr outputs and some contain
hard-coded paths. See [../docs/tests-and-tools.md](../docs/tests-and-tools.md).
