# Current State

This page is for agent handoff. It summarizes current practical state without
copying raw session logs.

## Working State

- Multi-resolution SST training and test paths exist for local DMI/Ohm and
  Gefion.
- The active implementation is SST-specific and lives mostly in `contrib/SST/`.
- The model cascade, SSL masking, dynamic prior, validation patch selection, and
  test aggregation are implemented.
- The residual cascade now trains x3/x1 against residual targets, not absolute
  SST targets. Before the next long run, recompute normalization stats and
  update active configs so mean-temperature channels use the shared
  `sst_common` scale.
- Gefion DDP has reached training on 8 H100 GPUs after Dask/DDP fixes.
- Gefion checkpoint testing should use `scripts/gefion/test_checkpoint_gefion.slurm`
  for a mono-GPU SLURM allocation.

## Known Technical Caveats

- Dask threading must remain disabled in DataLoader contexts. Keep
  `DASK_SCHEDULER=synchronous` and the worker init safeguards.
- In DDP, do not add explicit `shuffle=True` in train dataloaders; Lightning
  manages distributed sampling.
- `persistent_workers: false` is used in Gefion DDP configs to avoid worker
  deadlocks.
- `format_batch_for_solver()` builds `8*T + 4` input channels (124/76/44 for
  x10/x3/x1). All active configs are aligned on this layout.
- Gefion DDP is configured for one full node: 8 H100 GPUs.
- Gefion SQFS archives are stored durably under
  `/dcai/projects/cu_0026/data_sst/sqfs`; `xfer/guimae/inbox` is only transfer
  staging.
- Gefion run artifacts are stored under
  `/dcai/projects/cu_0026/guimae/croscim/` to avoid cluttering the shared
  project root.
- Gefion environment setup must load modules before venv activation. Use
  `source scripts/gefion/env.sh`; missing `mpmath`/`pandas` usually means
  `SciPy-bundle/2023.07` was not loaded first.
- Validation patch selection uses a fixed candidate budget. DDP rank 0 builds
  `val_indices.json`; the other ranks wait and load it. If the visual subset is
  poor or stale, rebuild with `datamodule.rebuild_val_set=true` and adjust
  `datamodule.val_candidate_budget`.
- Checkpoints produced before the residual-target fix should not be used to
  judge x3/x1 scientific quality; they trained finer solvers against absolute
  targets and then added the coarse prediction at inference. Do not launch the
  next long run until `compute_statistics.py` has regenerated stats and the
  active configs have been updated from those generated values.

## Script Caveats

- Local and Gefion scripts are not fully path-portable; paths are consistent
  per machine but hard-coded.
- Comments mentioning `4denv` are legacy environment notes.
- Gefion DDP training uses `scripts/gefion/train_gefion.sh`.
- Gefion checkpoint evaluation uses `scripts/gefion/test_checkpoint_gefion.slurm`;
  `run_test_checkpoint_gefion.sh` is the worker called inside that allocation.
- `run_gefion_single.sh` remains an interactive helper and should be inspected
  before use.
- Gefion preprocessing should use `data/process_year_gefion.slurm` per year.
  The older `data/process_all_years_gefion.sh` is not the preferred runbook.

## Open Work

From the active notes, the real next topics are:

- Understand and fix any remaining odd day/date display behavior.
- Tune learning rate, loss weights, and architecture variants.
- Confirm the DDP/Gefion configuration before long production runs.
- Inspect full Gefion run results, then run checkpoint evaluation through the
  mono-GPU SLURM wrapper.
- Run longer Gefion training and compare performance against existing methods.
- Quantify channel importance, especially `sea_ice_fraction`.
- Compare timing and quality against OI and, later, in situ validation.

## Non-Authoritative Notes

- `notes/prompt_clean.txt` is the old session bootstrap context. It contains
  useful explanations but also stale dimensions and old assumptions.
- `notes/memo_multires.md` contains useful architecture history but includes
  resolved problem statements.
- `notes/Point.txt` is old SSH/NATL60 learning context, not current SST state.
- `notes/session continue Gefion.txt` is a session summary, not a runbook.
