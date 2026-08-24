# Jean Zay Publication Run

This runbook covers the frozen SST publication training and evaluation
protocol. The hidden-pixel evaluator is independent of the historical
Lightning `test_step` and writes resumable daily publication artifacts.

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
$SCRATCH/croscim/data_sst_v2/data_YYYY  corrected daily Zarr stores
$WORK/croscim/publication/manifests     normalization and provenance files
$WORK/croscim/publication/tensorboard   TensorBoard events
$WORK/croscim/publication/checkpoints   checkpoints
$WORK/croscim/publication/runs          per-job manifests and logs
```

## 1. Verify Preprocessing

The original Jean Zay `data_sst` conversion must not be used: its filename glob
could read `_std_av.asc` as a mean field. Submit one corrected CPU job per year;
all jobs write to a separate root and may run concurrently:

```bash
cd "$WORK/croscim/repo/Croscim"
bash data/submit_reprocess_publication_jeanzay.sh
```

Each annual job validates that mean and uncertainty fields differ and reports
the x10 coverage of every sensor and the fused target. The submission script
also schedules train-only statistics with an `afterok` dependency on all annual
jobs. After all eight jobs complete, verify counts from the Croscim root:

```bash
for y in {2017..2024}; do
  n=365
  if [ "$y" = 2020 ] || [ "$y" = 2024 ]; then n=366; fi
  printf "%s expected=%s " "$y" "$n"
  for r in x1 x3 x10; do
    c=$(find "$SCRATCH/croscim/data_sst_v2/data_$y" -mindepth 2 -maxdepth 2 \
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
$WORK/croscim/publication/manifests/norm_stats_2017_2022_v2.yaml
$WORK/croscim/publication/manifests/norm_stats_2017_2022_v2.txt
$WORK/croscim/publication/manifests/norm_stats_sample_2017_2022_v2.yaml
$WORK/croscim/publication/manifests/norm_stats_2017_2022_v2.sha256
```

Verify completion and hashes:

```bash
cd "$WORK/croscim/publication/manifests"
sha256sum -c norm_stats_2017_2022_v2.sha256
grep -E "years:|n_files_available:|n_files_sampled:|sampling_seed:" \
  norm_stats_2017_2022_v2.yaml
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
GPU. The active configuration uses seed `20260821`, 8 A100 GPUs, `bf16-mixed`
and resolution-specific train settings: batch 4/4/2, accumulation 2/2/4 and
250/250/500 batches for x10/x3/x1. Every resolution sees 8,000 global samples
and 125 updates per epoch. The 192-epoch run trains each resolution for eight
epochs at a time. The A100 launcher requests 8 physical CPU cores per rank and
the config uses 6 DataLoader workers per rank. These values must be
changed through reviewed config edits or recorded Hydra overrides, not by
editing a generated resolved config.

## 4. A100 Submission

The first complete A100 smoke established that batch size 3 exhausts 80 GB on
the first x1 training batch. A smoke must exercise the active x10/x3/x1 batch
schedule before production:

```bash
export CROSCIM_RUN_ID="smoke_a100_$(date +%Y%m%d_%H%M%S)"
sbatch --qos=qos_gpu_a100-dev --time=02:00:00 \
  scripts/jeanzay/train_resunet_publication.slurm \
  trainer.max_epochs=3 \
  model.epochs_per_res_cycle=1 \
  trainer.limit_train_batches=12 \
  trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 \
  trainer.check_val_every_n_epoch=1
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

After the three-resolution smoke succeeds, submit an initial 10-job chain. A
dependency on the smoke can be supplied immediately, so the chain enters the
queue now but cannot start unless the smoke completes successfully:

```bash
export CROSCIM_RUN_ID=resunet_a100_dev_publication_20260822
export CROSCIM_CHAIN_AFTER=SMOKE_JOB_ID
bash scripts/jeanzay/submit_resunet_dev_chain.sh 10
unset CROSCIM_CHAIN_AFTER
```

Every job requests eight A100 GPUs for two hours and stops after one complete
epoch by default. The jobs use `afterok`, one shared run identifier, one
TensorBoard directory and one `last.ckpt`; every continuation therefore starts
from an epoch boundary. A real failure blocks all dependent jobs. Do not leave
another pending job using the same run identifier, because concurrent writers
would corrupt checkpoint and event provenance.

The validation scan is also shared across jobs, independently of the run ID.
The first smoke or training job writes the selected 2023 patch indices to:

```text
$WORK/croscim/publication/validation_set_2023_v2/val_indices.json
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
bash scripts/jeanzay/submit_resunet_dev_chain.sh 10
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
full duration from 2,250 batches per epoch before deciding how many 50-hour
continuations are required.

## 6. Acceptance Before Evaluation

Do not select a checkpoint from 2024 or from an incomplete resolution block.
The native `val/x1/loss` is computed without artificial withholding, so it
mostly measures reconstruction of already visible target pixels. It remains a
training diagnostic but is not the publication selection criterion.

Preserve every complete 24-epoch cycle checkpoint, evaluate each candidate on
the same controlled 24-date 2023 pilot, and rank candidates once by
latitude-weighted x1 RMSE on hidden pixels. Freeze the selected checkpoint and
the selection report before any quantitative 2024 evaluation. Until that
comparison is complete, do not run the finalization command below.

The native callback state can still be inspected for debugging:

```bash
cd "$WORK/croscim/repo/Croscim"
export CROSCIM_RUN_ID=resunet_resbatch_publication_20260822
python -m contrib.SST.evaluation.checkpoint \
  --checkpoint-dir "$WORK/croscim/publication/checkpoints/$CROSCIM_RUN_ID"
```

After controlled selection, the finalization step will create an immutable
`publication_best.ckpt`, its SHA-256 sidecar, selection report and
`publication_best.json` under:

```text
$WORK/croscim/publication/checkpoints/publication_frozen/$CROSCIM_RUN_ID/
```

The corrected two-hour chain stops after one complete epoch. Still derive the
actual optimizer-update and sample counts per resolution from the job logs
before reporting training effort:

```bash
python scripts/publication/audit_training.py \
  "$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/train.log" \
  --output-dir "$WORK/croscim/publication/training_audit/$CROSCIM_RUN_ID"
```

Before publication evaluation, archive:

- the resolved Hydra config and Git state;
- the normalization YAML, sample manifest and hashes;
- the validation-index cache;
- TensorBoard events and scheduler logs;
- the immutable best checkpoint and its manifest;
- the training audit.

## 7. One-Day Diagnostic Test

This diagnostic checks checkpoint compatibility, the x10 -> x3 -> x1 cascade,
edge-complete global assembly and the current qualitative figures. It does not
apply the frozen artificial test mask and its displayed errors are therefore
not publication metrics.

After pushing the local scripts and pulling them on Jean Zay:

```bash
cd "$WORK/croscim/repo/Croscim"
source scripts/jeanzay/env.sh
export CROSCIM_RUN_ID=resunet_resbatch_publication_20260822
bash scripts/jeanzay/submit_resunet_diagnostic.sh
```

Always pass the complete-cycle checkpoint being diagnosed explicitly. By
default the date index is 183, which is 2 July 2024; this date is for visual
debugging only and must not rank checkpoints. A checkpoint and index are passed
as follows:

```bash
bash scripts/jeanzay/submit_resunet_diagnostic.sh \
  "$WORK/croscim/publication/checkpoints/cycle_snapshots/$CROSCIM_RUN_ID/CYCLE.ckpt" 183
```

The wrapper prints `job_id` and `artifacts`. Monitor the returned job with:

```bash
tail -F "logs/jz_test_JOB_ID.out" "logs/jz_test_JOB_ID.err"
```

Outputs are isolated under:

```text
$WORK/croscim/publication/evaluations/diagnostic/TEST_ID/
```

The directory contains the checkpoint hash and metadata, logs, TensorBoard
events, NetCDF files, coverage maps and analysis figures. Do not use the
randomly selected patch figures or native-support RMSE values directly in the
paper.

The historical global NetCDF labels can be shifted at x3/x1 relative to the
requested target date. Treat this command as a qualitative compatibility
diagnostic only. The publication evaluator below writes the requested central
date explicitly.

## 8. Frozen Appendix-B Evaluation

### 8.1 Prepare manifests and static artifacts

Extract one original DMI-OI NetCDF from an SQFS archive and set
`RAW_OI_NETCDF` to its path. The exact comparison is mandatory because the
Zarr conversion does not preserve the original unit attribute.

```bash
cd "$WORK/croscim/repo/Croscim"
PROTOCOL_ROOT="$WORK/croscim/publication/evaluation_protocol_v2"
RAW_DATE=20240702
SQFS="$SCRATCH/croscim/sqfs/L4_all_2024_GBL_0.05_REAN_4_production_test.sqfs"
mkdir -p "$PROTOCOL_ROOT/raw_reference"
MEMBER=$(unsquashfs -l "$SQFS" | \
  grep "/${RAW_DATE}0000-DMI-L4_GHRSST-STskin-DMI_OI-GLOB-v02.0-fv01.0.nc$" | \
  head -1)
MEMBER=${MEMBER#squashfs-root/}
test -n "$MEMBER"
RAW_OI_NETCDF="$PROTOCOL_ROOT/raw_reference/${RAW_DATE}_dmi_oi.nc"
unsquashfs -cat "$SQFS" "$MEMBER" > "$RAW_OI_NETCDF"
sbatch scripts/jeanzay/prepare_evaluation_protocol.slurm \
  "$RAW_OI_NETCDF" 2024-07-02
```

This creates deterministic donor, 24-date pilot and 352-date test manifests,
the 50 km coastal mask, and the DMI-OI verification report. The controlled
mask copies only missing ocean observations from its 2017--2022 donor, then
applies the manifest's deterministic longitude shift. A coarse x3 or x10 cell
is withheld whenever any x1 pixel in its footprint is withheld, preventing the
coarse cascade from leaking the hidden target.

### 8.2 Run and accept the 2023 pilot

```bash
RUN_ID=resunet_resbatch_publication_20260822
PROTOCOL_ROOT="$WORK/croscim/publication/evaluation_protocol_v2"
CKPT_ROOT="$WORK/croscim/publication/checkpoints/publication_frozen/$RUN_ID"
CKPT="$CKPT_ROOT/publication_best.ckpt"
PILOT="$PROTOCOL_ROOT/manifests/pilot.json"
PILOT_ID=appendix_b_pilot_2023
export CROSCIM_EVAL_LIMIT_DATES=1
sbatch --ntasks=1 --gres=gpu:1 --time=01:58:00 \
  scripts/jeanzay/evaluate_resunet_publication.slurm \
  "$CKPT" "$PILOT" appendix_b_pilot_smoke controlled
unset CROSCIM_EVAL_LIMIT_DATES
```

Inspect that single-date integration test before submitting the full pilot.
It exercises controlled masking and the new exporter, unlike the historical
qualitative diagnostic. Then submit:

```bash
bash scripts/jeanzay/submit_evaluation_chain.sh \
  4 "$CKPT" "$PILOT" "$PILOT_ID" controlled
```

Each allocation starts eight independent A100 workers. Daily `.done.json`
markers make the chain resumable. When all 24 dates are present, validate the
maps, run the conditional spatial-scale diagnostic and freeze the protocol on
a CPU node:

```bash
sbatch scripts/jeanzay/finalize_evaluation_pilot.slurm \
  "$RUN_ID" "$PILOT_ID"
```

The validator recomputes every sufficient statistic from the NetCDF maps,
requires finite predictions on the exact common support, and checks zero
uncovered pixels at x10, x3 and x1.

### 8.3 Freeze before opening 2024

The pilot finalization job is intentionally one-shot. It hashes the checkpoint,
manifests, static inputs, pilot markers and evaluation source files. A 2024
evaluator refuses to start without this file or when any frozen source differs.

### 8.4 Run and validate the final 2024 test

```bash
RUN_ID=resunet_resbatch_publication_20260822
PROTOCOL_ROOT="$WORK/croscim/publication/evaluation_protocol_v2"
CKPT="$WORK/croscim/publication/checkpoints/publication_frozen/$RUN_ID/publication_best.ckpt"
TEST="$PROTOCOL_ROOT/manifests/test.json"
FROZEN="$PROTOCOL_ROOT/frozen_protocol.json"
TEST_ID=appendix_b_test_2024
bash scripts/jeanzay/submit_evaluation_chain.sh \
  10 "$CKPT" "$TEST" "$TEST_ID" controlled "$FROZEN"
```

Append another chain after the recorded final job when necessary:

```bash
export CROSCIM_CHAIN_AFTER=$(cat \
  "$WORK/croscim/publication/evaluations/$TEST_ID/chains/last_job.txt")
bash scripts/jeanzay/submit_evaluation_chain.sh \
  10 "$CKPT" "$TEST" "$TEST_ID" controlled "$FROZEN"
unset CROSCIM_CHAIN_AFTER
```

After all 352 dates complete, run the validation, aggregation and figure
generation on a CPU node:

```bash
sbatch scripts/jeanzay/finalize_evaluation_test.slurm "$TEST_ID"
```

The main table is `table_main_croscim_vs_dmi_oi.csv`; the resolution ablation,
monthly/seasonal summaries, block-bootstrap intervals and figure inputs remain
separate artifacts. `runtime_daily.csv` records stage and end-to-end timings,
while `runtime_summary.csv` separates accumulated single-GPU work from observed
parallel wall time. DMI-OI is an operational reference that may have assimilated
the withheld observations, not an input-matched baseline.

### 8.5 Secondary modes

Run the rectangular-mask sensitivity with a separate evaluation identifier and
the same frozen checkpoint, manifest and protocol:

```bash
RECT_ID=appendix_b_rectangles_2024
bash scripts/jeanzay/submit_evaluation_chain.sh \
  10 "$CKPT" "$TEST" "$RECT_ID" rectangles "$FROZEN"
```

Validate and aggregate it with `--mode rectangles`; never mix its files with
the primary controlled run. The evaluator also supports `natural` mode for
qualitative reconstructions of native gaps. Natural outputs have no hidden-
pixel metrics and are not a replacement for the controlled 2024 table.
