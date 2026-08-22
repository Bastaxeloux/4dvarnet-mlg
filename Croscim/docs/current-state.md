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
  SST targets.
- Normalization stats have been regenerated with
  `contrib/SST/compute_statistics.py`. Active configs now use the shared
  `sst_common` scale for `aasti.av`, `avhrr.av`, `pmw.av`, `slstr.av`, and
  `tgt_sst`; satellite `_std` fields keep their own generated stats.
- A full Gefion DDP run from scratch completed on 8 H100 GPUs after the
  residual-target and normalization fixes. Validation figures indicate that the
  x10 -> x3 -> x1 intensity drift is resolved.
- Global test reconstruction now covers domain edges and applies land masking
  only for visualization; the missing border bands and coastal display
  artifacts have been resolved.
- Daily x1/x3/x10 data for 2017–2024 have been validated as the expanded
  training range. Years 2014–2015 lack SLSTR and 2016 has incomplete SLSTR
  coverage.
- An experimental ResUNet prior and dedicated Gefion config/script exist.
  Validation graph retention was fixed and ResUNet runs produced coherent
  validation figures. A later batch-size-5 run exhausted an 80 GB H100 when
  training switched from x10 to the larger x3 solver; this is distinct from
  the resolved validation leak.
- A Jean Zay publication workflow is prepared with train-only
  normalization on 2017–2022, validation on 2023 and final evaluation on the
  352 eligible dates in 2024. The active A100 schedule uses batch 4 for x10/x3
  and batch 2 for x1, with matching accumulation and batch budgets that keep
  effective batch 64, 125 updates and 8,000 global samples per epoch at every
  resolution. Training now uses 192 epochs in eight-epoch resolution blocks,
  preserving the former total sample and optimizer-update budgets. The
  20-hour A100 and 50-hour V100 launchers both resume under a stable run ID.
- Gefion checkpoint testing uses `scripts/gefion/test_checkpoint_gefion.slurm`
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
  targets and then added the coarse prediction at inference.
- ResUNet memory is dominated by unrolled-solver activations rather than its
  parameter count or Zarr file size. Validation avoids retaining higher-order
  graphs; training memory still varies by resolution and x3/x1 must be checked
  before a long batch-size change is accepted.

## Script Caveats

- Local and Gefion scripts are not fully path-portable; paths are consistent
  per machine but hard-coded.
- Comments mentioning `4denv` are legacy environment notes.
- Gefion DDP training uses `scripts/gefion/train_gefion.sh`.
- Gefion ResUNet-prior DDP training uses
  `scripts/gefion/train_gefion_resunet.sh`.
- Jean Zay A100 and V100 publication launchers are respectively
  `scripts/jeanzay/train_resunet_publication.slurm` and
  `scripts/jeanzay/train_resunet_publication_v100.slurm`. Each hardware path
  must pass its x10/x3/x1 smoke test before a long run.
- Gefion checkpoint evaluation uses `scripts/gefion/test_checkpoint_gefion.slurm`;
  `run_test_checkpoint_gefion.sh` is the worker called inside that allocation.
- `run_gefion_single.sh` remains an interactive helper and should be inspected
  before use.
- Gefion preprocessing should use `data/process_year_gefion.slurm` per year.
  The older `data/process_all_years_gefion.sh` is not the preferred runbook.

## Open Work

From the active notes, the real next topics are:

- Complete the Jean Zay publication run and preserve its checkpoint provenance.
- Implement deterministic 2024 hidden-pixel evaluation and matched DMI-OI
  metrics. The archived bilinear run remains qualitative context only.
- Design a Swin Transformer prior after the ResUNet experiment is validated.
- Later, investigate replacing or augmenting the ConvLSTM gradient modulator.
- Quantify channel importance, especially `sea_ice_fraction`.
- Compare timing and quality against OI and, later, in situ validation.

## Non-Authoritative Notes

- `notes/prompt_clean.txt` is the old session bootstrap context. It contains
  useful explanations but also stale dimensions and old assumptions.
- `notes/memo_multires.md` contains useful architecture history but includes
  resolved problem statements.
- `notes/Point.txt` is old SSH/NATL60 learning context, not current SST state.
- `notes/session continue Gefion.txt` is a session summary, not a runbook.
