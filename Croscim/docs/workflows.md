# Workflows

This document lists practical commands and caveats. Many scripts use absolute
paths; inspect them before running on a new machine.

## Environment

```bash
conda create -n croscim
conda activate croscim
conda install -c conda-forge mamba
mamba env update -f environment.yaml
export PYTHONPATH=$PWD:$PYTHONPATH
```

`environment.yaml` has no `name:` field. The environment name is a local choice.

## Local Training

Quick smoke run:

```bash
./scripts/local/run_train_lite.sh 0
```

Main local training:

```bash
./scripts/local/run.sh 0
```

Resume from checkpoint:

```bash
./scripts/local/run.sh 0 /dmidata/projects/4dvarnet/checkpoints_sst_multires/last.ckpt
```

These scripts run in background with `nohup`, write a log file, and store the
process id in `process.pid`.

## TensorBoard

```bash
./scripts/local/tensorboard.sh
```

The TensorBoard script should be preferred over older comments that mention
`launch_tensorboard.sh`.

## Checkpoint Test

Local checkpoint test:

```bash
./scripts/local/run_test_checkpoint.sh /path/to/checkpoint.ckpt
```

This script overrides Hydra entrypoints to call `src.test.base_test`.

Gefion checkpoint test should be submitted through SLURM, not launched directly
on a login node:

```bash
mkdir -p logs
sbatch scripts/gefion/test_checkpoint_gefion.slurm \
  /dcai/projects/cu_0026/guimae/croscim/checkpoints/last.ckpt
```

The Gefion test runs mono-GPU and overrides the DDP training config with
`trainer.devices=1` and `trainer.strategy=auto`. Extra Hydra overrides can be
appended after the checkpoint path:

```bash
sbatch scripts/gefion/test_checkpoint_gefion.slurm \
  /dcai/projects/cu_0026/guimae/croscim/checkpoints/last.ckpt \
  datamodule.domains.test.time._args_=[2024-12-01,2024-12-31]
```

Logs are written to `logs/test_<jobid>.out`, `logs/test_<jobid>.err`, and
`logs/test_<jobid>.log`. TensorBoard test logs use
`/dcai/projects/cu_0026/guimae/croscim/outputs/test_results`. Model artifacts
from `Lit4dVarNet_SST` use `model.outputs_dir`, so NetCDF files and analysis
figures are under:

```text
/dcai/projects/cu_0026/guimae/croscim/outputs/<test_run_id>/test/
```

## Gefion

Single-GPU experiment:

```bash
sbatch scripts/gefion/submit_gefion_single.sh exp_name
```

With overrides:

```bash
sbatch scripts/gefion/submit_gefion_single.sh exp_prior02 model.loss_weights.prior=0.2
```

DDP training:

```bash
sbatch scripts/gefion/train_gefion.sh
```

Validation set debug override:

```bash
sbatch scripts/gefion/train_gefion.sh datamodule.val_candidate_budget=200 datamodule.val_set_num_workers=16 datamodule.rebuild_val_set=true
```

Gefion scripts source `scripts/gefion/env.sh`. This loads modules, including
`SciPy-bundle/2023.07`, before activating `/dcai/projects/cu_0026/croscim_env`,
then sets `PYTHONPATH`, `HYDRA_FULL_ERROR`, and `DASK_SCHEDULER=synchronous`.
Run artifacts are written under `/dcai/projects/cu_0026/guimae/croscim/`, not
directly at the shared project root.

Interactive Gefion setup:

```bash
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
source scripts/gefion/env.sh
```

Gefion preprocessing from SQFS archives:

```bash
sbatch data/process_year_gefion.slurm 2022
squeue -u "$USER"
tail -f logs/preprocess_2022_*.out
```

The preprocessing script uses `/dcai/projects/cu_0026/data_sst/sqfs` as the
durable archive source. `/dcai/projects/cu_0026/xfer/guimae/inbox` is only a
transfer inbox and should be empty or ignored after archives are moved.

Validation figures from training are written by rank 0 under:

```text
/dcai/projects/cu_0026/guimae/croscim/outputs/<train_run_id>/validation/epoch_XXX/
```

Find recent validation figures with:

```bash
find /dcai/projects/cu_0026/guimae/croscim/outputs \
  -type f \( -name "validation_all_patches_epoch_*.jpg" -o -name "validation_multires_patches_epoch_*.jpg" \) \
  | sort | tail -50
```

## Known Script Caveats

- Local scripts hard-code `/home/malegu/4D-MLG/Croscim`.
- Some configs use `/dmidata/users/malegu/4D-MLG/Croscim`.
- Gefion configs use `/dcai/users/guimae/4dvarnet-mlg/Croscim`.
- Gefion outputs/checkpoints/results use
  `/dcai/projects/cu_0026/guimae/croscim/`.
- Gefion preprocessing uses `/dcai/projects/cu_0026/data_sst/sqfs`, not the
  transfer inbox under `xfer/`.
- Gefion validation cache is built only by DDP rank 0; if validation selection
  looks stale after changing budget, dates, or thresholds, use
  `datamodule.rebuild_val_set=true`.
- Load all Gefion modules before activating the venv. Loading `SciPy-bundle`
  after activation can hide venv packages such as Hydra.
- Some comments still mention `conda activate 4denv`; current docs use
  `croscim`.
- Gefion DDP training has run on 8 H100 GPUs. The checkpoint test now has a
  SLURM wrapper, but each new test window/checkpoint should still be checked
  with the generated logs and output paths.

## Build and Test Policy

There is no package build step and no CI in this repository.

Tests are standalone Python scripts under `tests/`; they are not a pytest suite.
See [Tests and tools](tests-and-tools.md).
