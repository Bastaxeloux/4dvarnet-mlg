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

Gefion scripts load modules, activate `/dcai/projects/cu_0026/croscim_env`,
set `PYTHONPATH`, and export `DASK_SCHEDULER=synchronous`.

Gefion preprocessing from SQFS archives:

```bash
sbatch data/process_year_gefion.slurm 2022
squeue -u "$USER"
tail -f logs/preprocess_2022_*.out
```

The preprocessing script uses `/dcai/projects/cu_0026/data_sst/sqfs` as the
durable archive source. `/dcai/projects/cu_0026/xfer/guimae/inbox` is only a
transfer inbox and should be empty or ignored after archives are moved.

## Known Script Caveats

- Local scripts hard-code `/home/malegu/4D-MLG/Croscim`.
- Some configs use `/dmidata/users/malegu/4D-MLG/Croscim`.
- Gefion configs use `/dcai/users/guimae/4dvarnet-mlg/Croscim`.
- Gefion preprocessing uses `/dcai/projects/cu_0026/data_sst/sqfs`, not the
  transfer inbox under `xfer/`.
- Some comments still mention `conda activate 4denv`; current docs use
  `croscim`.
- Gefion scripts have never been exercised end-to-end. Read each one before
  the first real Gefion submission.

## Build and Test Policy

There is no package build step and no CI in this repository.

Tests are standalone Python scripts under `tests/`; they are not a pytest suite.
See [Tests and tools](tests-and-tools.md).
