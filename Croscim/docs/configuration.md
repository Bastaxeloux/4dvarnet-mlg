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
| `SST/multires_gefion_resunet` | Gefion DDP | Experimental ResUNet prior, 2017–2024 |
| `SST/multires_jeanzay_resunet` | Jean Zay DDP | Publication ResUNet run; train 2017–2022, validate 2023 |
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
| `multires_gefion_resunet` | 8 | DDP | 5 | 32 | 96 | `bf16-mixed` | 2017-01-01 to 2024-09-30 |
| `multires_jeanzay_resunet` | 8 | DDP | 3 | 6 | 96 | `bf16-mixed` | 2017-01-01 to 2022-12-31 |

The lite DDP and single Gefion configs are close variants intended for debugging
or experiments.

## Normalization Stats

Active configs use a shared `sst_common` scale for all mean-temperature fields:

- `aasti.av`
- `avhrr.av`
- `pmw.av`
- `slstr.av`
- `tgt_sst`

The baseline Gefion config still uses the statistics generated for its
2023–2024 data setup. The ResUNet 2017–2024 experiment uses statistics
regenerated from 1000 sampled x1 daily files:

```yaml
mean: 7.78095617993726
std: 20.58217902545633
type: zscore
```

Satellite `_std` fields keep their own generated stats. Do not invent these
values by hand. The Jean Zay publication config loads the train-only generated
YAML directly; older configs still contain copied historical values.

## ResUNet Memory Settings

The latest large Gefion configuration uses batch size 5 and accumulation 2.
It reached the x3 phase only after the smaller validation-graph memory issue
was fixed, then exceeded 80 GB at x3. This is a configuration-specific OOM,
not evidence that the ResUNet implementation never ran: earlier ResUNet runs
produced coherent training/validation figures, although their complete
checkpoint provenance was not exported.

The Jean Zay A100 publication config uses:

```yaml
batch_size: 2
accumulate_grad_batches: 4
limit_train_batches: 1000
```

With 8 GPUs this gives an effective batch of 64 and 250 optimizer updates per
epoch. Batch size 3 is not valid for this architecture: the first x1 training
batch exhausted an 80 GB A100. The `gpu_p5` node has 64 physical CPU cores, so
the launcher assigns 8 cores per DDP rank and uses 6 DataLoader workers per
rank.

The dedicated V100 launcher keeps the same scientific config but applies
runtime overrides for one four-GPU `v100-32g` node: FP16 mixed precision, batch
size 1, accumulation 18, 4,500 train batches and six DataLoader workers per
rank. This gives effective batch 72, 250 optimizer updates and 18,000 global
samples per epoch. Each rank has ten
physical CPU cores and each V100 has 32 GB, so a three-resolution smoke run
remains mandatory. See `docs/jeanzay-publication.md`.

## Validation Set

Active configs build a fixed validation cache at `datamodule.val_set_dir`.
The cache stores `n_viz` visualization patches first, then `n_loss` loss
patches. The builder scans `val_candidate_budget` unique candidates with
`val_set_seed`, ranks visualization patches by score, and samples loss patches
from candidates passing `patch_filter.val_loss`.

Gefion configs use `val_candidate_budget: 400` and `val_set_num_workers: 16`.
The Jean Zay publication config uses a budget of 600 to obtain its requested
16 visualization and 48 loss patches. Local configs use a smaller budget.
`val_set_max_scan` remains as a legacy fallback for older overrides.

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

Gefion environment setup is centralized in `scripts/gefion/env.sh`. It loads
`SciPy-bundle/2023.07` before activating the venv because `mpmath` and `pandas`
are provided by the module stack in the current setup.

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

Gefion checkpoint evaluation uses the training config with test-specific
overrides applied by `scripts/gefion/run_test_checkpoint_gefion.sh`:

- entrypoint changed to `src.test.base_test`
- `trainer.devices=1`
- `trainer.strategy=auto`
- checkpoint path passed as `ckpt_path`
- test window left to the config default unless an override is passed

Submit it through `scripts/gefion/test_checkpoint_gefion.slurm` so the test runs
inside a mono-GPU SLURM allocation.
