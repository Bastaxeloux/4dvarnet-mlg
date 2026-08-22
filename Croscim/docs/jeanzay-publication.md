# Jean Zay Publication Run

This runbook covers the frozen SST publication training protocol. It does not
yet cover final hidden-pixel evaluation, which requires the deterministic test
mask exporter described in `notes/manuscript/appendix_sst_artifact_contract.md`.

## Frozen Split

- training and normalization: 2017--2022;
- validation and checkpoint selection: 2023 only;
- final evaluation: the 352 central dates in 2024 whose complete 15-day
  context stays inside 2024.

The publication model uses the three-level ResUNet-prior cascade. The training
job does not run the final test automatically.

## Paths

```text
$WORK/croscim/repo/Croscim              code checkout
$SCRATCH/croscim/sqfs                   staged SQFS archives
$SCRATCH/croscim/data_sst/data_YYYY     preprocessed daily Zarr stores
$WORK/croscim/publication/manifests     normalization and provenance files
$WORK/croscim/publication/tensorboard   TensorBoard events
$WORK/croscim/publication/checkpoints   checkpoints
$WORK/croscim/publication/runs          per-job manifests and logs
```

## 1. Verify Preprocessing

From the Croscim root:

```bash
for y in {2017..2024}; do
  n=365
  if [ "$y" = 2020 ] || [ "$y" = 2024 ]; then n=366; fi
  printf "%s expected=%s " "$y" "$n"
  for r in x1 x3 x10; do
    c=$(find "$SCRATCH/croscim/data_sst/data_$y" -mindepth 2 -maxdepth 2 \
      -path "*_${r}.zarr/.zmetadata" -type f | wc -l)
    printf "%s=%s " "$r" "$c"
  done
  echo
done
```

Every count must be 365, except 2020 and 2024, which must be 366. The
statistics and training scripts repeat these checks and fail rather than use a
partial year.

## 2. Compute Train-Only Statistics

Pull the current code, create the log directory before submission, then submit:

```bash
cd "$WORK/croscim/repo/Croscim"
git pull
mkdir -p logs
sbatch scripts/jeanzay/compute_statistics_publication.slurm
```

Monitor the reported job ID:

```bash
squeue -j JOB_ID -o "%.18i %.10P %.20j %.8T %.12M %R"
tail -F logs/jz_stats_JOB_ID.out
```

The job deterministically samples 1,500 x1 daily stores from 2017--2022 with
seed `20260821`. It writes:

```text
$WORK/croscim/publication/manifests/norm_stats_2017_2022.yaml
$WORK/croscim/publication/manifests/norm_stats_2017_2022.txt
$WORK/croscim/publication/manifests/norm_stats_sample_2017_2022.yaml
$WORK/croscim/publication/manifests/norm_stats_2017_2022.sha256
```

Verify completion and hashes:

```bash
cd "$WORK/croscim/publication/manifests"
sha256sum -c norm_stats_2017_2022.sha256
grep -E "years:|n_files_available:|n_files_sampled:|sampling_seed:" \
  norm_stats_2017_2022.yaml
```

## 3. Preflight The Training Job

```bash
cd "$WORK/croscim/repo/Croscim"
source scripts/jeanzay/env.sh
python scripts/jeanzay/check_environment.py
python main.py xp=SST/multires_jeanzay_resunet --cfg job --resolve \
  > "$WORK/croscim/publication/resolved_preflight.yaml"
bash -n scripts/jeanzay/train_resunet_publication.slurm
sbatch --test-only scripts/jeanzay/train_resunet_publication.slurm
```

Fix every missing import reported by the environment check before requesting a
GPU. The active configuration uses seed `20260821`, 8 A100 GPUs, `bf16-mixed`,
batch size 2 per GPU, gradient accumulation 4, 1,000 batches per epoch and 96
epochs. It performs 250 optimizer updates per epoch and cycles through x10, x3
and x1 every four epochs. The A100 launcher requests 8 physical CPU cores per
rank and the config uses 6 DataLoader workers per rank. These values must be
changed through reviewed config edits or recorded Hydra overrides, not by
editing a generated resolved config.

## 4. A100 Submission

The first complete A100 smoke established that batch size 3 exhausts 80 GB on
the first x1 training batch. The active batch size 2 must therefore complete
one epoch at each resolution before production:

```bash
export CROSCIM_RUN_ID="smoke_a100_$(date +%Y%m%d_%H%M%S)"
sbatch --qos=qos_gpu_a100-dev --time=02:00:00 \
  scripts/jeanzay/train_resunet_publication.slurm \
  trainer.max_epochs=3 \
  model.epochs_per_res_cycle=1 \
  trainer.limit_train_batches=12 \
  trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.accumulate_grad_batches=4
```

Only after x10, x3 and x1 complete without OOM, choose one stable identifier
for every 20-hour A100 allocation:

```bash
export CROSCIM_RUN_ID=resunet_a100_publication_20260822
mkdir -p logs
sbatch scripts/jeanzay/train_resunet_publication.slurm
```

Then:

```bash
squeue -j JOB_ID -o "%.18i %.10P %.20j %.8T %.12M %R"
tail -F logs/jz_train_JOB_ID.out
```

The first checks in the allocation validate all 2017--2024 x1/x3/x10 stores,
the train-only normalization file, CUDA visibility and the Python environment.
The A100 launcher loads `arch/a100` before the PyTorch module, as required by
the Jean Zay A100 software stack.
The job writes its Git revision, dirty state, package versions and statistics
hash under:

```text
$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/
```

Stable output locations are:

```text
$WORK/croscim/publication/checkpoints/$CROSCIM_RUN_ID/last.ckpt
$WORK/croscim/publication/tensorboard/resunet_publication_2017_2024/$CROSCIM_RUN_ID
$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/train.log
```

Submitting the same script again with the same `CROSCIM_RUN_ID` automatically
resumes `last.ckpt`. Slurm sends `SIGUSR1` five minutes before walltime; the
trainer saves at the next optimizer boundary and exits cleanly. Set
`CROSCIM_RESUME_CKPT=/absolute/path.ckpt` only to override this automatic
choice.

### Chained Two-Hour Development Jobs

The development QoS can be used for a sequence of short allocations when its
queue is substantially faster than the normal A100 QoS. Inspect current usage
and configured limits first:

```bash
squeue -p gpu_p5 -q qos_gpu_a100-dev \
  -o "%.18i %.10u %.8T %.10M %.20j %R"
sacctmgr show qos qos_gpu_a100-dev \
  format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,GrpJobs,GrpSubmit
```

After the three-resolution smoke succeeds, submit an initial 12-job chain. A
dependency on the smoke can be supplied immediately, so the chain enters the
queue now but cannot start unless the smoke completes successfully:

```bash
export CROSCIM_RUN_ID=resunet_a100_dev_publication_20260822
export CROSCIM_CHAIN_AFTER=SMOKE_JOB_ID
bash scripts/jeanzay/submit_resunet_dev_chain.sh 12
unset CROSCIM_CHAIN_AFTER
```

Every job requests eight A100 GPUs for two hours. The jobs use `afterok`, one
shared run identifier, one TensorBoard directory and one `last.ckpt`. A real
failure blocks all dependent jobs; a walltime signal produces a checkpoint and
a successful exit. Do not leave another pending job using the same run
identifier, because concurrent writers would corrupt checkpoint and event
provenance.

The validation scan is also shared across jobs, independently of the run ID.
The first smoke or training job writes the selected 2023 patch indices to:

```text
$WORK/croscim/publication/validation_set_2023/val_indices.json
```

With `datamodule.rebuild_val_set=false` (the publication default), subsequent
jobs validate the cached seed, filters, candidate budget and dataset length,
then load the same 16 visualization and 48 loss indices. The expected log line
is `[VAL SET] Loaded 64 indices ...`; `[VAL SET] Scanning ...` should therefore
appear only once. Keep the smoke as the first `afterok` dependency so the cache
is complete before the production chain begins.

The submission manifest and final dependency ID are stored under:

```text
$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/dev_chain_*.txt
$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/dev_chain_last_job.txt
```

Append another chain without overlap by reusing both the run identifier and
the previous final job as the first dependency:

```bash
export CROSCIM_CHAIN_AFTER=$(cat \
  "$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/dev_chain_last_job.txt")
bash scripts/jeanzay/submit_resunet_dev_chain.sh 12
unset CROSCIM_CHAIN_AFTER
```

TensorBoard can be started on the Jean Zay login node with:

```bash
tensorboard --logdir "$WORK/croscim/publication/tensorboard" \
  --host 127.0.0.1 --port 6006
```

Use the existing SSH tunnel and open `http://127.0.0.1:6006` locally.
The run appears below
`resunet_publication_2017_2024/$CROSCIM_RUN_ID`. In particular, inspect:

- `general/train_resolution`;
- `train/x10/*`, `train/x3/*`, `train/x1/*` and `val/x1/loss`;
- `perf/gpu_memory_gb`, `perf/gpu_reserved_memory_gb` and
  `perf/gpu_peak_memory_gb`;
- `perf/batch_size_per_gpu`, `perf/effective_batch_size`, batch time and
  throughput.

The peak-memory counter is reset at each training epoch, so the three-epoch
smoke run gives one directly comparable high-water mark for x10, x3 and x1.

## 5. V100 Preflight, Smoke Test And Continuation

The V100 path is separate from the A100 reference job. It requests one
quad-GPU node with four 32 GB V100 GPUs (`v100-32g`), uses native FP16 mixed
precision and runs for at most 50 hours. The default runtime overrides are:

```text
batch size per GPU:       1
gradient accumulation:   18
effective batch size:    72
train batches per epoch: 4500
optimizer updates/epoch: 250
global samples/epoch:     18000
DataLoader workers/rank:  6
```

Before the full run, submit a short job that forces one x10, one x3 and one x1
epoch. This is a memory and integration test, not a scientific experiment:

```bash
cd "$WORK/croscim/repo/Croscim"
git pull
mkdir -p logs
bash -n scripts/jeanzay/train_resunet_publication_v100.slurm
CROSCIM_RUN_ID=v100_preflight \
  sbatch --test-only scripts/jeanzay/train_resunet_publication_v100.slurm

CROSCIM_RUN_ID="smoke_v100_$(date +%Y%m%d_%H%M%S)" \
sbatch --qos=qos_gpu-dev --time=02:00:00 \
  scripts/jeanzay/train_resunet_publication_v100.slurm \
  trainer.max_epochs=3 \
  model.epochs_per_res_cycle=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.accumulate_grad_batches=1
```

The smoke job succeeds only if all three resolution messages appear, GPU peak
memory remains below 32 GB and `last.ckpt` is written. If batch size 1 fails,
the current model cannot be trained on V100 without activation checkpointing
or a smaller architecture; use A100 rather than starting the full V100 run.

For the publication run, choose one stable identifier and keep it for every
50-hour continuation:

```bash
export CROSCIM_RUN_ID=resunet_v100_publication_20260822
sbatch scripts/jeanzay/train_resunet_publication_v100.slurm
```

The script writes an epoch-level checkpoint to:

```text
$WORK/croscim/publication/checkpoints/$CROSCIM_RUN_ID/last.ckpt
```

Submitting the same command with the same `CROSCIM_RUN_ID` automatically
resumes that checkpoint. The first cycle-boundary checkpoint and TensorBoard
epoch number must continue rather than restart from zero. To resume an explicit
checkpoint instead:

```bash
export CROSCIM_RESUME_CKPT=/absolute/path/to/checkpoint.ckpt
sbatch scripts/jeanzay/train_resunet_publication_v100.slurm
```

Do not alternate A100 and V100 launchers inside one run identifier: they use
different numerical precision and micro-batch settings. Start a separate run
identifier when changing hardware. Unset `CROSCIM_RESUME_CKPT` after an
explicit recovery so it cannot override later automatic resumes.

Monitor each allocation with:

```bash
squeue -j JOB_ID -o "%.18i %.10P %.20j %.8T %.12M %R"
tail -F logs/jz_v100_JOB_ID.out
```

Do not assume a 40-hour completion time from the earlier H100 runs. Record the
stage-specific batch times from the three-epoch smoke test, then estimate the
full duration from 4,500 batches per epoch before deciding how many 50-hour
continuations are required.

## 6. Acceptance Before Evaluation

Do not select a checkpoint from 2024. The callback compares only complete
12-epoch cycle boundaries using `val/x1/loss` on the fixed 2023 validation set;
it does not compare stage-specific x10/x3/x1 losses. Retain both the best and
final checkpoints. Before publication evaluation, archive:

- the resolved Hydra config and Git state;
- the normalization YAML, sample manifest and hashes;
- the validation-index cache;
- TensorBoard events and scheduler logs;
- best and final checkpoints.

Final metric tables remain blocked until deterministic hidden/visible masks,
date manifests, common-support x10/x3/x1 exports and the DMI-OI comparator are
implemented and verified.
