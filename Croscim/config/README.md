# Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. All training parameters, data paths, model architecture, and solver settings are defined in YAML files.

## How it works

`config/main.yaml` is the root config. It requires an experiment (`xp`) to be specified on the command line:

```bash
python main.py xp=SST/multires          # Main training config
python main.py xp=SST/multires_lite     # Quick pipeline test
```

Hydra searches for configs in two packages:
- `pkg://config` — this directory
- `pkg://contrib` — the contribution modules

The `__init__.py` registers:
- A `_singleton` resolver for sharing instantiated objects across the config
- Geographic domain definitions (NATL, global, arctic, etc.) in the Hydra ConfigStore

## Active experiment configs (`xp/SST/`)

| Config | Target machine | Purpose | Key differences |
|---|---|---|---|
| `multires.yaml` | DMI local (1 GPU) | **Main training config.** Full model, 84 epochs, 200 batches/epoch. | Full-size solvers (n_step: 20/10/10), bf16 mixed precision |
| `multires_lite.yaml` | DMI local (1 GPU) | Quick pipeline validation. 3 epochs, 20 batches. | Small solvers (n_step: 3/3/2, dim_hidden: 32), fp32 |
| `multires_lite_ddp.yaml` | DMI local (4 GPUs) | DDP debugging with lite config. | 4 devices, DDP strategy, persistent_workers: false |
| `multires_gefion.yaml` | Gefion (6 H100s) | **Production training.** 96 epochs, DDP. | 6 devices, DDPStrategy, 24 workers/GPU, bf16 |
| `multires_single_gefion.yaml` | Gefion (1 H100) | Hyperparameter experiments on a single GPU. | 28 workers, batch_size: 6, 96 epochs |
| `base_sst.yaml` | DMI local | Legacy config from early experiments. Uses old class names (x2/x10/x50). | `data_simple_multires`, `BilinAEPriorCost` — kept for reference |

All active configs (except `base_sst.yaml`) use:
- `contrib.SST.data_multires.BaseDataModuleMultiRes` for data loading
- `contrib.SST.models.Lit4dVarNet_SST` as the Lightning module
- `contrib.SST.solver.GradSolvers` with `BilinReconstructorPriorCost` as the solver
- Resolutions: [10, 3, 1] (50 km, 15 km, 5 km)
- Patch size: 256 x 256 pixels, temporal window: 15 days
