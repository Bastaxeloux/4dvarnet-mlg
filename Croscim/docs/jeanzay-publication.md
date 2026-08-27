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
jobs. The source archives use CCI AVHRR/SLSTR through 2021 and C3S from
2022-01-01; duplicate files at that boundary are resolved with the same product
for the mean and uncertainty. After all eight jobs complete, verify counts from
the Croscim root:

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
and resolution-specific train settings: batch 6/4/2, accumulation 2/2/4 and
126/188/376 batches for x10/x3/x1. This gives 6,048/6,016/6,016 global samples
and 63/94/94 updates. The shorter x1 epoch targets roughly 40--45 minutes. The
192-epoch run trains each resolution for eight epochs at a time. The A100
launcher requests 8 physical CPU cores per rank and the config uses 6
DataLoader workers per rank. These values must be changed through reviewed
config edits or recorded Hydra overrides, not by editing a generated resolved
config.

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

Every job requests eight A100 GPUs for two hours. The submission script reads
the next epoch from `last.ckpt` and schedules at most four complete x10 epochs,
four x3 epochs or two x1 epochs per job. A job never crosses an eight-epoch
resolution boundary. The jobs use `afterok`, one shared run identifier, one
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
$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/dev_chain_next_epoch.txt
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
- `perf/gpu_peak_epoch_gib` and `perf/gpu_peak_reserved_epoch_gib`;
- `perf/batch_size_per_gpu`, `perf/effective_batch_size`, batch time and
  throughput.

The peak-memory counter is reset at each training epoch, so the three-epoch
smoke run gives one directly comparable high-water mark for x10, x3 and x1.
Per-batch memory polling is intentionally disabled. Set
`CROSCIM_NUMERICS_DEBUG=1` only for a diagnostic run that needs finite checks at
every variational iteration; the normal path still checks each solver's final
state.

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

Select the publication checkpoint from the fixed 2023 validation results. The
2024 test data must not influence this choice. Pass the selected checkpoint
directly to the evaluator; no snapshot or finalization step is required.

```bash
cd "$WORK/croscim/repo/Croscim"
export CROSCIM_RUN_ID=resunet_v2_hotpath_publication_20260824
ls -lh "$WORK/croscim/publication/checkpoints/$CROSCIM_RUN_ID"/cycle_end_epoch=*.ckpt
```

The two-hour chain currently schedules at most four complete x10 epochs, four
complete x3 epochs, or two complete x1 epochs per allocation. Still derive the
actual optimizer-update and sample counts per resolution from the job logs
before reporting training effort:

```bash
python scripts/publication/audit_training.py \
  "$WORK/croscim/publication/runs/$CROSCIM_RUN_ID/train.log" \
  --output-dir "$WORK/croscim/publication/training_audit/$CROSCIM_RUN_ID"
```

The evaluator records the checkpoint path and hash, epoch, resolved Hydra
configuration, normalization path and software versions once in its output
directory.

## 7. One-Day Diagnostic Test

This diagnostic checks checkpoint compatibility, the x10 -> x3 -> x1 cascade,
edge-complete global assembly and the current qualitative figures. It does not
apply the artificial evaluation mask and its displayed errors are therefore
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

## 8. Appendix-B Evaluation

### 8.1 Prepare dates and static inputs once

This CPU job creates the deterministic 24-date 2023 pilot manifest, the
352-date 2024 manifest and the coastal mask. It also checks DMI-OI units, grid,
valid time and exclusion from model inputs using the corrected Zarr files.

```bash
cd "$WORK/croscim/repo/Croscim"
PREP_JOB=$(sbatch --parsable scripts/jeanzay/prepare_evaluation_protocol.slurm)
echo "$PREP_JOB"
tail -F "logs/jz_eval_prep_${PREP_JOB}.out" \
        "logs/jz_eval_prep_${PREP_JOB}.err"
```

The original DMI-OI NetCDF comparison can be performed once before reporting
the final 2024 comparison. It is not needed for the checkpoint-47 pilot.

### 8.2 Evaluate checkpoint epoch 47

The historical test remains the quickest qualitative check of natural gaps.
The controlled pilot below evaluates all 24 dates, computes the publication
metrics and exports enough patches for manual visual selection.

```bash
RUN_ID=resunet_v2_hotpath_publication_20260824
PROTOCOL_ROOT="$WORK/croscim/publication/evaluation_protocol_v2"
CKPT=$(find "$WORK/croscim/publication/checkpoints/$RUN_ID" \
  -maxdepth 1 -type f -name 'cycle_end_epoch=047-*.ckpt' -print -quit)
test -s "$CKPT"
PILOT="$PROTOCOL_ROOT/manifests/pilot.json"
PILOT_ID="${RUN_ID}_cycle047_pilot_2023"
bash scripts/jeanzay/submit_evaluation_chain.sh \
  1 "$CKPT" "$PILOT" "$PILOT_ID" controlled
```

The allocation is resumable by daily `.done.json` markers. Submit the same
command again only if one two-hour allocation does not finish the 24 dates.
After completion, aggregate the dates with a standard bootstrap over dates and
generate the diagnostic curves and patch gallery:

```bash
sbatch scripts/jeanzay/postprocess_evaluation.slurm \
  "$PILOT_ID" "$PILOT" 1
```

The useful outputs are deliberately simple:

- `results/table_main_croscim_vs_dmi_oi.csv` and the detailed metric CSV files;
- `results/bootstrap_intervals.csv` with 2,000 date-bootstrap replicates;
- `results/runtime_summary.csv` for inference cost;
- `results/diagnostics/quantitative_diagnostics.*` for the first metric curves;
- `results/summary/evaluation_summary.*` for a one-page PNG/PDF report of the
  main metrics, regimes, assembly diagnostics and runtime;
- `results/patch_catalog.csv` for all admissible candidates;
- `results/patch_gallery/` with about 60 review images showing withheld input,
  revealed target, x10, x3, x1 and hidden-pixel error.

Choose the paper patches manually from this gallery. Producing figures B1--B3
from that short list is a separate final formatting step and cannot affect the
quantitative metrics.

### 8.3 Evaluate the final selected checkpoint

After training, rerun the same pilot by changing only the explicit checkpoint
and evaluation identifier:

```bash
RUN_ID=resunet_v2_hotpath_publication_20260824
CKPT=/absolute/path/to/the/selected/checkpoint.ckpt
PROTOCOL_ROOT="$WORK/croscim/publication/evaluation_protocol_v2"
PILOT="$PROTOCOL_ROOT/manifests/pilot.json"
PILOT_ID="${RUN_ID}_selected_pilot_2023"
bash scripts/jeanzay/submit_evaluation_chain.sh \
  1 "$CKPT" "$PILOT" "$PILOT_ID" controlled
sbatch scripts/jeanzay/postprocess_evaluation.slurm \
  "$PILOT_ID" "$PILOT" 1
```

Each allocation starts eight independent A100 workers. Daily `.done.json`
markers make the chain resumable. Use a new evaluation identifier whenever the
checkpoint changes.

### 8.4 Run the final 2024 test

```bash
PROTOCOL_ROOT="$WORK/croscim/publication/evaluation_protocol_v2"
CKPT=/absolute/path/to/the/selected/checkpoint.ckpt
TEST="$PROTOCOL_ROOT/manifests/test.json"
TEST_ID=appendix_b_test_2024
bash scripts/jeanzay/submit_evaluation_chain.sh \
  10 "$CKPT" "$TEST" "$TEST_ID" controlled
```

Append another chain after the recorded final job when necessary:

```bash
export CROSCIM_CHAIN_AFTER=$(cat \
  "$WORK/croscim/publication/evaluations/$TEST_ID/chains/last_job.txt")
bash scripts/jeanzay/submit_evaluation_chain.sh \
  10 "$CKPT" "$TEST" "$TEST_ID" controlled
unset CROSCIM_CHAIN_AFTER
```

After all 352 dates complete, aggregate with a 30-day circular block bootstrap
and generate the curves and patch gallery:

```bash
sbatch scripts/jeanzay/postprocess_evaluation.slurm \
  "$TEST_ID" "$TEST" 30
```

The main table is `table_main_croscim_vs_dmi_oi.csv`; the resolution ablation,
monthly/seasonal summaries, block-bootstrap intervals and figure inputs remain
separate artifacts. `runtime_daily.csv` records stage and end-to-end timings,
while `runtime_summary.csv` separates accumulated single-GPU work from observed
parallel wall time. DMI-OI is an operational reference that may have assimilated
the withheld observations, not an input-matched baseline.
