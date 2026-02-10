# 4D-VarNet Multi-Resolution SST Reconstruction

Global Sea Surface Temperature (SST) reconstruction from partial satellite observations, using a multi-resolution 4D-VarNet approach.

## The problem

Satellites measure SST every day, but each sensor only sees a fraction of the ocean. Clouds block infrared sensors, orbits leave gaps between swaths, and no single satellite covers the full globe at high resolution. The result: on any given day, most of the ocean's SST is unknown.

We want to reconstruct a complete, high-resolution (5 km) SST field from these sparse, noisy observations.

## The approach: 4D-VarNet

4D-VarNet frames SST reconstruction as a **variational data assimilation** problem solved by a neural network.

The idea: find an SST field (the "state") that is both **consistent with the observations** and **physically plausible**. In practice, this means minimizing a cost function:

```
J(state) = obs_cost(state, observations) + prior_cost(state)
```

where:
- **obs_cost** = how far the state is from the actual satellite measurements (only where data exists)
- **prior_cost** = how far the state is from what a learned reconstructor thinks it should look like

Instead of minimizing J with a classical optimizer, 4D-VarNet uses a **ConvLSTM** to learn a better gradient descent. At each iteration, the ConvLSTM takes the raw gradient `dJ/d(state)` and outputs a smarter update direction. After several iterations, the state converges to a good reconstruction.

### Two levels of optimization

This is the core mechanism:

**Level 1 — Optimize the SST image** (inside the forward pass):
- Start from an initial guess (e.g., the masked satellite fusion)
- Run N iterations of learned gradient descent to refine the SST
- Network weights are *frozen* during this loop

**Level 2 — Optimize the network weights** (standard backprop):
- Compare the final reconstruction to the ground truth
- Backpropagate through the entire unrolled optimization
- Update the weights of the ConvLSTM, the prior reconstructor, etc.

The network learns *how to optimize*, not just *what the answer is*.

### Multi-resolution cascade

Reconstructing 5 km SST directly is hard — the model needs both global context (ocean currents, large-scale patterns) and local detail (fronts, eddies). We solve this with a coarse-to-fine cascade:

```
x10 (50 km) ──→ x3 (15 km) ──→ x1 (5 km)
 GradSolver       GradSolver      GradSolver
 10 iters         10 iters        20 iters
```

1. **x10** sees the big picture: 256×256 pixels at 50 km = roughly 12,800 km across
2. Its output is **interpolated** and fed to **x3** as additional context
3. x3 refines with regional patterns, then feeds **x1**
4. x1 adds the fine details → final reconstruction

Each level has its own GradSolver with independent weights. The Data Assimilation Window (DAW) narrows temporally: x10 uses 15 days, x3 uses 9 (central crop), x1 uses 5.

## Data

### Satellite sensors

| Sensor | Variable | Coverage | Notes |
|---|---|---|---|
| SLSTR | `slstr_av`, `slstr_std` | Mid-latitudes | Sentinel-3. High accuracy, our primary source. |
| AASTI | `aasti_av`, `aasti_std` | Polar regions | AATSR heritage. Essential where SLSTR has no data. |
| AVHRR | `avhrr_av`, `avhrr_std` | Good global | NOAA. Reliable long-term record. |
| PMW | `pmw_av`, `pmw_std` | Near-global | Passive Microwave. All-weather but smooth. |

Plus: `sea_ice_fraction` (covariate), `surfmask`, `lat`, `lon`, `analysed_st`, `analysis_error`, `oi_data`.

### Format

Daily Zarr files at 3 resolutions, stored as:
```
/nwp/sst_malegu/data_2024/
├── 2024010112_x1.zarr    # 5 km  (3600 × 7200), chunks 768×768
├── 2024010112_x3.zarr    # 15 km (1200 × 2400)
├── 2024010112_x10.zarr   # 50 km (360 × 720)
├── ...
```

Training patches: **256×256 pixels**, **15 days** temporal window.

## Repository structure

```
Croscim/
├── main.py              Entry point (Hydra)
├── environment.yaml     Conda environment
│
├── config/              Hydra configuration (6 active SST experiment configs)
├── contrib/SST/         Core ML code: dataset, model, solver, normalization
├── src/                 Base 4D-VarNet framework (training loop, ConvLSTM, utilities)
│
├── scripts/             Shell scripts for training and monitoring
│   ├── local/           DMI machine (A40 GPUs)
│   └── gefion/          Gefion HPC cluster (H100 GPUs)
│
├── data/                Data preprocessing pipeline (squashfs → Zarr)
├── tests/               Standalone test and validation scripts
├── tools/               Analysis and data inspection utilities
│
├── csv/                 Exported training metrics
├── figs/                Generated figures (gitignored)
├── outputs/             Hydra run outputs (gitignored)
└── archive/             Legacy code and old documentation
```

Each folder has its own README with detailed file descriptions.

## Quickstart

### Environment setup

```bash
git clone <this-repo>
cd Croscim
conda create -n croscim
conda activate croscim
conda install -c conda-forge mamba
mamba env update -f environment.yaml
```

### Local training (DMI machine)

```bash
# Quick pipeline test (3 epochs, ~5 min)
./scripts/local/run_train_lite.sh 0

# Full training on GPU 0
./scripts/local/run.sh 0

# Resume from checkpoint
./scripts/local/run.sh 0 /dmidata/projects/4dvarnet/checkpoints_sst_multires/last.ckpt

# Monitor with TensorBoard
./scripts/local/tensorboard.sh
```

### Gefion HPC

```bash
# Single-GPU experiment with custom hyperparameters
sbatch scripts/gefion/submit_gefion_single.sh my_experiment +model.opt_lr=1e-4

# Multi-GPU DDP training
sbatch scripts/gefion/train_gefion.sh
```

### Test a checkpoint

```bash
./scripts/local/run_test_checkpoint.sh /path/to/checkpoint.ckpt
```

## Key technical points

### xarray/Dask deadlock

Using xarray with Dask inside PyTorch DataLoader workers causes deadlocks: Dask spawns threads, PyTorch forks workers, and threads don't survive `fork()`. The fix: **use pure Zarr** (no xarray) in `__getitem__()`, and set `DASK_SCHEDULER=synchronous` when xarray is unavoidable.

### Patch validation

Not all patches are useful for training. `is_valid_patch()` in `data.py` rejects patches that are:
- Mostly NaN (< 8% valid data)
- Uniform (variance < 0.05) — open ocean with no gradients
- Mostly land (< 5% ocean pixels)

Up to 50 retries per sample before falling back.

### SSL (Self-Supervised Learning)

During training, some observed pixels are artificially masked (inpainting). The model must reconstruct them from surrounding data. This forces the model to learn spatial structure rather than just memorizing observations. The observation cost only sees *unmasked* pixels; the final loss is computed on *all* pixels including artificially masked ones.

## References

- Fablet et al. (2021). *Learning Variational Data Assimilation Models and Solvers.* JAMES. [DOI](https://doi.org/10.1029/2021MS002572)
- Fablet et al. (2021). *End-to-End Physics-Informed Representation Learning for Satellite Ocean Remote Sensing Data.* ISPRS Annals. [DOI](https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021)
- Fablet et al. (2021). *Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields.* Frontiers. [DOI](https://doi.org/10.3389/fams.2021.655224)
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/)
