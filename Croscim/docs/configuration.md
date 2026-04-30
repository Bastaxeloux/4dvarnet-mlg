# Configuration

Croscim uses Hydra for experiment configuration.

## Root Config

`config/main.yaml` is the root config:

```yaml
defaults:
  - xp: ???
  - _self_
```

Every normal run must provide an experiment:

```bash
python main.py xp=SST/multires_lite
```

Hydra search paths include both `config` and `contrib`.

## Active SST Configs

| Config | Purpose | Notes |
|---|---|---|
| `SST/multires` | Main local training | Single GPU, full model |
| `SST/multires_lite` | Local smoke run | Short, reduced model |
| `SST/multires_lite_ddp` | Local DDP debug | Multi-GPU debug config |
| `SST/multires_gefion` | Gefion DDP | Production-style H100 run |
| `SST/multires_single_gefion` | Gefion single GPU | Hyperparameter experiments |
| `SST/base_sst` | Legacy | Keep for reference only |

`base_sst.yaml` references older classes and older dimensions. Do not use it as
the current implementation guide.

## Config Matrix

Values below come from the current YAML files and should be rechecked before a
long run.

| Config | Devices | Strategy | Batch | Workers | Epochs | Precision | Train dates |
|---|---:|---|---:|---:|---:|---|---|
| `multires` | 1 | none | 3 | 16 | 84 | `bf16-mixed` | 2022-01-01 to 2023-12-31 |
| `multires_lite` | 1 | none | 4 | 8 | 3 | `32` | 2024-01-01 to 2024-08-31 |
| `multires_gefion` | 8 | DDP | 6 | 24 | 96 | `bf16-mixed` | 2023-01-01 to 2024-09-30 |

The lite DDP and single Gefion configs are close variants intended for debugging
or experiments.

## Paths

Local/DMI configs and scripts currently mix several roots:

- `/home/malegu/4D-MLG/Croscim`
- `/dmidata/users/malegu/4D-MLG/Croscim`
- `/nwp/sst_malegu`
- `/dmidata/projects/4dvarnet`

Gefion configs use:

- `/dcai/users/guimae/4dvarnet-mlg/Croscim`
- `/dcai/projects/cu_0026/data_sst`
- `/dcai/projects/cu_0026/data_sst/sqfs`
- `/dcai/projects/cu_0026/guimae/croscim/outputs`
- `/dcai/projects/cu_0026/guimae/croscim/results`
- `/dcai/projects/cu_0026/guimae/croscim/checkpoints`
- `/dcai/projects/cu_0026/guimae/croscim/tmp`

`/dcai/projects/cu_0026/xfer/guimae/inbox` is only a transfer inbox. Do not use
it as the durable source for preprocessing.

These paths are not fully normalized. Treat scripts as machine runbooks, not as
portable commands.

## Solver Dimensions

The active `format_batch_for_solver()` implementation builds `8*T + 4`
channels:

| Resolution | T | Expected `dim_in` |
|---|---:|---:|
| x10 | 15 | 124 |
| x3 | 9 | 76 |
| x1 | 5 | 44 |

All active configs (`multires`, `multires_lite`, `multires_lite_ddp`,
`multires_gefion`, `multires_single_gefion`) are aligned on 124/76/44.
Only `base_sst.yaml` keeps the historical 139/85/49 layout, marked as legacy.

## Hydra Overrides

Common overrides:

```bash
python main.py xp=SST/multires trainer.devices=1
python main.py xp=SST/multires +ckpt=/path/to/last.ckpt
python main.py xp=SST/multires_single_gefion model.loss_weights.prior=0.2
```

The scripts under `scripts/` wrap these commands and add machine-specific
environment setup.
