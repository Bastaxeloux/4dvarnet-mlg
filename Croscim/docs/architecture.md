# Architecture

This document describes the active SST architecture as implemented in the code.

## Goal

Croscim reconstructs global Sea Surface Temperature from incomplete satellite
observations. The model uses 4D-VarNet: a learned variational solver that
optimizes an SST state inside the forward pass, then uses standard backprop to
train the neural components.

## Hydra Execution Flow

Runtime is Hydra-driven:

```text
main.py
  -> config/main.yaml
  -> xp=SST/<experiment>
  -> trainer, datamodule, model, entrypoints
  -> src.train.base_training()
  -> trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
```

`main.py` does not parse task-specific CLI arguments. Use Hydra overrides:

```bash
python main.py xp=SST/multires_lite
python main.py xp=SST/multires +ckpt=/path/to/checkpoint.ckpt
```

Testing scripts override `entrypoints` to call `src.test.base_test()`.

## Main Components

- `src/`: base framework, training/test entrypoints, generic Lightning classes,
  ConvLSTM, utilities, metrics.
- `contrib/SST/data.py`: single-resolution SST dataset, patch loading,
  normalization, target fusion, SSL inpainting, patch validation.
- `contrib/SST/data_multires.py`: x1/x3/x10 nested patch extraction and
  multi-resolution datamodule.
- `contrib/SST/models.py`: active SST Lightning module, cascade forward pass,
  losses, validation, test output aggregation.
- `contrib/SST/solver.py`: SST-specific GradSolver, GradSolvers container, and
  observation cost.
- `contrib/SST/model_components/`: learned solver components. Priors currently
  include the bilinear baseline and experimental ResUNet; grad modulators
  currently include the ConvLSTM update model.

## Multi-Resolution Cascade

Training samples contain three geographically nested patches:

```text
x10 50 km  256x256  contains x3
x3  15 km  256x256  contains x1
x1   5 km  256x256  final high-resolution target
```

The active forward pass is coarse to fine:

1. Solve x10 directly from x10 observations.
2. Interpolate x10 prediction onto the x3 grid.
3. Convert x3 temperature inputs and targets to anomalies relative to
   interpolated x10.
4. Solve x3 residual.
5. Add x3 residual to interpolated x10 to get x3 prediction.
6. Repeat the same pattern from x3 to x1.

This is residual refinement, not a simple concatenation of coarse prediction as
an extra context channel.

## Data Assimilation Window

The model starts with 15 daily timesteps and crops the central temporal window
at finer resolutions:

| Resolution | Solver output | Meaning |
|---|---:|---|
| x10 | 15 days | Full context |
| x3 | 9 days | Central crop |
| x1 | 5 days | Central crop and final output |

The fixed DAW lengths are defined in `Lit4dVarNet_SST.len_daw`.

## Two Optimization Levels

4D-VarNet has two distinct optimization loops.

### Level 1: State Optimization

Inside `GradSolver.forward()`, the model optimizes the SST image state:

```text
state_0 = masked SST fusion
for step in n_step:
    var_cost = prior_cost(state, batch) + obs_cost(state, batch)
    grad = d(var_cost) / d(state)
    gmod = ConvLSTM(grad)
    state = state - update(gmod, grad)
```

The optimized variable is the SST state tensor, not the network weights.

### Level 2: Weight Optimization

After the unrolled forward pass, Lightning computes the training loss and
backpropagates through the solver to update model parameters.

The active SST loss is configurable and combines:

- MSE reconstruction loss
- Sobel gradient loss
- dynamic prior regularization

## Prior Variants

The baseline prior is `BilinReconstructorPriorCost`. The experimental
`ResUNetPriorCost` keeps the same solver interface and changes only the learned
reconstructor used by the prior cost:

```text
Phi([state, covariates]) -> reconstructed SST state
prior cost = MSE(state, reconstructed state)
```

It uses residual convolution blocks, an encoder-decoder hierarchy, and skip
connections. It does not replace the ConvLSTM update model or the unrolled
4D-VarNet optimization. The dedicated Hydra config is
`SST/multires_gefion_resunet`.

## Solver Input Channels

`format_batch_for_solver()` builds input tensors as:

```text
[masked fusion, satellite uncertainty/auxiliary channels, covariates, spatial/time]
```

Current code builds `8*T + 4` channels:

| Resolution | T | Channels |
|---|---:|---:|
| x10 | 15 | 124 |
| x3 | 9 | 76 |
| x1 | 5 | 44 |

Older notes mention 139/85/49 (original layout) or 124/70/34 (intermediate
"prior dynamique" layout without `slstr_std`/`aasti_std`). Those dimensions
are legacy and should not be copied into current docs.
