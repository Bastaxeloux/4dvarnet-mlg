# contrib/SST

Active SST-specific implementation.

Useful files:

- `data.py`: single-resolution dataset, target fusion, normalization, SSL
  inpainting, patch validation.
- `data_multires.py`: nested x1/x3/x10 dataset and datamodule.
- `load_data.py`: satellite variable groups, covariates, file organization, fast
  coarsening helpers.
- `models.py`: SST Lightning module, cascade, losses, validation, test outputs.
- `solver.py`: SST GradSolver, GradSolvers container, observation cost.
- `model_components/`: learned solver components, split into priors and gradient
  modulators.
- `compute_statistics.py`: normalization statistics computation.
- `visualization.py`: plotting helpers.
- `UNet.py`: reserved for experiments.
- `norm_stats.yaml`: normalization statistics reference.

Current solver input construction is documented in
[../../docs/architecture.md](../../docs/architecture.md). Do not reuse older
139/85/49 channel dimensions from raw notes without rechecking the code.

See also:

- [../../docs/data.md](../../docs/data.md)
- [../../docs/configuration.md](../../docs/configuration.md)
