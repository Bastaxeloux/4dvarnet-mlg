# contrib/SST — SST Multi-Resolution Module

This is the core of the project. Everything specific to multi-resolution SST reconstruction lives here.

## File overview

| File | Role |
|---|---|
| `data.py` | Single-resolution dataset (`XrDataset`), patch extraction, validation, normalization |
| `data_multires.py` | Multi-resolution dataset (`XrDatasetMultiResTrain`), nested patch extraction, `BaseDataModuleMultiRes` |
| `load_data.py` | Variable definitions (sensors, covariates), fast pooling, file organization by resolution |
| `models.py` | `Lit4dVarNet_SST` Lightning module — training loop, multi-res forward pass, loss computation |
| `solver.py` | `GradSolver`, `BilinReconstructorPriorCost`, `BaseObsCost`, `ConvLstmGradModel` |
| `compute_statistics.py` | Computes normalization stats (mean/std) by sampling from Zarr files |
| `visualization.py` | Plotting utilities for patches, satellite coverage maps, regional visualizations |
| `callbacks.py` | `FilterMetricsCallback` — minimal callback for metric filtering |
| `UNet.py` | U-Net encoder-decoder architecture (reserved for future experiments) |
| `norm_stats.yaml` | Precomputed normalization statistics for all variables |

## Data pipeline

### Sensors and variables

4 satellite SST sources, each with mean (`av`) and uncertainty (`std`):

| Sensor | Coverage | Characteristics |
|---|---|---|
| `slstr` | Mid-latitudes, no poles | Sentinel-3 SLSTR. High accuracy. |
| `aasti` | Poles, sparse elsewhere | AATSR/Envisat. Essential for polar regions. |
| `avhrr` | Good global coverage | NOAA AVHRR. Reliable workhorse. |
| `pmw` | Very global, all-weather | Passive Microwave. Smooth, lower spatial detail. |

Additional variables: `sea_ice_fraction` (covariate), `surfmask` (land/ocean), `lat`, `lon`, `time`.

**Target SST** (`tgt_sst`): Fusion of SLSTR (priority in low-ice zones) and AASTI (high-ice zones).

### Multi-resolution patches

Each training sample contains 3 geographically nested patches:

```
x10 (50 km, 360x720)  ⊃  x3 (15 km, 1200x2400)  ⊃  x1 (5 km, 3600x7200)
     256x256 patch            256x256 patch              256x256 patch
     15 days                  15 days                    15 days
```

The extraction in `data_multires.py` guarantees that each coarser patch geographically contains the finer one, using `encompassing_patch()` from `src/utils.py`.

### Data Assimilation Window (DAW)

The temporal window narrows as we go from coarse to fine:

| Resolution | Full window | After DAW crop | Reason |
|---|---|---|---|
| x10 | 15 days | 15 days | Full context |
| x3 | 15 days | 9 days | Central crop from x10's output |
| x1 | 15 days | 5 days | Central crop from x3's output |

### Normalization

All variables are normalized before entering the model. Stats in `norm_stats.yaml`:
- Satellite variables: z-score (mean/std)
- `sea_ice_fraction`: min-max [0, 1]
- `tgt_sst`: z-score

The normalization pipeline in `data.py:apply_norm()` also handles:
- SSL inpainting mask generation (artificial masking of observed pixels for self-supervised learning)
- Target fusion (SLSTR + AASTI combination based on sea ice)

## Model architecture

### Forward pass (multi-resolution cascade)

```
                  x10 (coarse, 50 km)
                  ┌──────────────────┐
batch_x10.input → │ GradSolver_x10   │ → pred_x10
                  │ (10 iterations)  │
                  └──────────────────┘
                           │
                     interpolate ↓ (to x3 spatial grid)
                           │
                  x3 (medium, 15 km)
                  ┌──────────────────┐
batch_x3.input  → │ GradSolver_x3    │ → pred_x3
+ pred_x10_up   → │ (10 iterations)  │
                  └──────────────────┘
                           │
                     interpolate ↓ (to x1 spatial grid)
                           │
                  x1 (fine, 5 km)
                  ┌──────────────────┐
batch_x1.input  → │ GradSolver_x1    │ → pred_x1  (final output)
+ pred_x3_up    → │ (20 iterations)  │
                  └──────────────────┘
```

Each GradSolver receives the upsampled prediction from the coarser level as additional context.

### Inside a GradSolver

The GradSolver performs **learned variational optimization**. It iteratively refines an SST estimate (the "state") by minimizing a cost function:

```
J(state) = prior_cost(state, batch) + obs_cost(state, batch)
```

At each iteration:
1. **Prior cost**: `BilinReconstructorPriorCost` reconstructs SST from `[state, covariates]` and measures `||state - reconstruction||²`. This is a **dynamic prior** — it evolves with the state.
2. **Observation cost**: `BaseObsCost` measures `||state - observations||²` on non-NaN pixels only.
3. **Gradient**: `grad = d(J)/d(state)` via `torch.autograd.grad`
4. **Gradient modulation**: A `ConvLstmGradModel` (ConvLSTM) takes the raw gradient and outputs a modulated gradient — learning a better descent direction than vanilla gradient descent.
5. **State update**: `state = state - (gmod / (step+1) + lr * grad * (step+1) / n_step)`

### Two levels of optimization

This is the key insight of 4D-VarNet:

**Level 1 — State optimization** (inside `GradSolver.forward()`):
- Optimizes the *SST image* (state) using the learned gradient descent
- Network weights are *fixed* during this loop
- Runs for `n_step` iterations (e.g., 20 for x1)

**Level 2 — Weight optimization** (PyTorch backprop):
- After the full forward pass, computes the final loss against ground truth
- Backpropagates through the entire unrolled optimization to update *network weights*
- Loss = MSE + gradient_loss (Sobel) + prior_loss

### Input channel structure

For resolution x10 (15 timesteps):
```
channels 0-14:   tgt_sst fusion (15 T)
channels 15-28:  slstr_std (15 T)  — or aasti_std depending on config
channels 29-42:  aasti_std (15 T)
channels 43-72:  avhrr av + std (30 T)
channels 73-102: pmw av + std (30 T)
channels 103-117: sea_ice_fraction (15 T)
channels 118-121: lat, lon, time_encoded, surfmask (4 spatial)
```

For x3 (9 timesteps): same structure but with 9T → dim_in = 76
For x1 (5 timesteps): same structure but with 5T → dim_in = 44
