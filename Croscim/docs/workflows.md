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

Local:

```bash
./scripts/local/tensorboard.sh
```

The TensorBoard script should be preferred over older comments that mention
`launch_tensorboard.sh`.

### TensorBoard From Gefion Citrix

TensorBoard and the MobaXterm tunnel must use the same Gefion login node. For
example, if the tunnel targets `login01.gefion.dcai.dk`, open a terminal on
`login01`, then run:

```bash
cd /dcai/users/guimae/4dvarnet-mlg/Croscim
source scripts/gefion/env.sh
tensorboard --logdir /dcai/projects/cu_0026/guimae/croscim/results \
  --host 127.0.0.1 --port 6123
```

Configure the MobaXterm local port forwarding with:

```text
forwarded/local port: 6123
remote server: 127.0.0.1
remote port: 6123
SSH server: login01.gefion.dcai.dk
SSH login: guimae
```

Then open `http://127.0.0.1:6123` in Firefox inside the Citrix desktop. If
TensorBoard runs on `login02`, the tunnel must also target `login02`.

Before debugging Firefox, verify the service from a second terminal on the same
login node:

```bash
hostname
curl -I http://127.0.0.1:6123
```

`Connection refused` means TensorBoard is not listening on that node/port. A
working tunnel without a matching TensorBoard process cannot serve the page.

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

Experimental ResUNet-prior DDP training on 2017–2024:

```bash
sbatch scripts/gefion/train_gefion_resunet.sh
```

Both scripts load modules and activate the Gefion venv themselves through
`scripts/gefion/env.sh`; activating the environment before `sbatch` is not
required. Do not pass the experiment through `sbatch --export`: each script
hard-codes its Hydra config, job name, and time limit.

The baseline path has completed on 8 H100 GPUs after the residual-target and
`sst_common` normalization fixes. The ResUNet path is experimental. Its first
batch-size-3 run failed during backward at approximately 79.1 GiB GPU memory;
batch size 2 also failed during the validation sanity check. The current config
still preserves an effective batch of 96, but the open issue is graph retention
across the unrolled ResUNet/ConvLSTM steps rather than input batch size alone.

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

Use the script default partition (`defq`) for now. A run forced onto `cpuq`
failed at `import numpy` because `SciPy-bundle/2023.07` loaded a NumPy build
requiring `AVX512_SPR`; the script now checks the Python stack before
extracting the SQFS archive.

The preprocessing script uses `/dcai/projects/cu_0026/data_sst/sqfs` as the
durable archive source. `/dcai/projects/cu_0026/xfer/guimae/inbox` is only a
transfer inbox and should be empty or ignored after archives are moved.

The validated complete range is 2017–2024. Do not train on 2014–2015 because
SLSTR is absent, or on 2016 because SLSTR coverage is incomplete, without
changing the target/data policy.

Normalization statistics for this range are generated with:

```bash
sbatch scripts/gefion/compute_statistics_gefion.slurm
```

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
- A job in `UserEnv retrieval failed, requeued held` failed before the batch
  script ran. Cancel that held job and resubmit the dedicated script without
  custom `--export` arguments.

## Build and Test Policy

There is no package build step and no CI in this repository.

Tests are standalone Python scripts under `tests/`; they are not a pytest suite.
See [Tests and tools](tests-and-tools.md).
