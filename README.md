# 4dvarnet-mlg

Repository for the Croscim SST reconstruction work.

The active project is [`Croscim/`](Croscim/): a multi-resolution 4D-VarNet
pipeline for global Sea Surface Temperature reconstruction from sparse
satellite observations. It uses daily Zarr files at three resolutions, trains
with PyTorch Lightning and Hydra, and targets coarse-to-fine reconstruction:

- `x10`: 50 km context
- `x3`: 15 km intermediate refinement
- `x1`: 5 km final high-resolution SST

## Directory Layout

```text
4dvarnet-mlg/
├── Croscim/              active SST 4D-VarNet project
├── data/                 preprocessing scripts and historical data pipeline work
└── 4dvarnet-starter/     original starter repository snapshot, kept as legacy reference
```

## Active Project: Croscim

Use [`Croscim/README.md`](Croscim/README.md) as the main project README.

Important Croscim docs:

- [`Croscim/AGENTS.md`](Croscim/AGENTS.md): onboarding and operating guide for AI agents.
- [`Croscim/docs/architecture.md`](Croscim/docs/architecture.md): Hydra flow, model cascade, solver logic.
- [`Croscim/docs/data.md`](Croscim/docs/data.md): Zarr layout, variables, normalization, patch extraction.
- [`Croscim/docs/workflows.md`](Croscim/docs/workflows.md): local and Gefion run commands.
- [`Croscim/docs/current-state.md`](Croscim/docs/current-state.md): known caveats and current work state.

Typical entry points:

```bash
cd Croscim
./scripts/local/run_train_lite.sh 0
./scripts/local/run.sh 0
./scripts/local/run_test_checkpoint.sh /path/to/checkpoint.ckpt
```

## Root `data/`

The root-level `data/` directory contains preprocessing scripts and older data
pipeline work. It is kept separate from `Croscim/data/` for now to avoid mixing
active project code with historical preprocessing experiments.

Before reusing anything from this directory, compare it with the maintained
preprocessing docs in [`Croscim/docs/tests-and-tools.md`](Croscim/docs/tests-and-tools.md).

## `4dvarnet-starter/`

`4dvarnet-starter/` is the original starter repository snapshot that served as
a base for Croscim. Its `.git` directory was removed, and it is now a legacy
reference rather than the active codebase.

Do not treat it as the current implementation source of truth. Use it only to
understand historical design choices or compare with the current `Croscim/`
implementation.
