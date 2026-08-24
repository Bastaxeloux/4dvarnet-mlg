# Croscim

Multi-resolution 4D-VarNet experiments for global Sea Surface Temperature
reconstruction from sparse satellite observations.

The active project reconstructs SST fields from daily Zarr files at three
resolutions:

- `x10`: 50 km, coarse global context
- `x3`: 15 km, intermediate context
- `x1`: 5 km, final high-resolution target

Training uses a coarse-to-fine cascade with PyTorch Lightning and Hydra. The
main SST implementation lives in `contrib/SST/`; the reusable 4D-VarNet
framework pieces live in `src/`.

## Documentation

Start here:

- [Agent guide](AGENTS.md) for project-specific onboarding and caveats.
- [Architecture](docs/architecture.md) for the Hydra flow, model cascade, and
  two optimization levels.
- [Data](docs/data.md) for Zarr layout, variables, normalization, SSL masking,
  and patch extraction.
- [Configuration](docs/configuration.md) for active Hydra configs and machine
  settings.
- [Workflows](docs/workflows.md) for local, Gefion, checkpoint, and TensorBoard
  commands.
- [Jean Zay publication run](docs/jeanzay-publication.md) for the frozen
  2017--2022 / 2023 / 2024 ResUNet training and evaluation protocol.
- [Tests and tools](docs/tests-and-tools.md) for validation scripts,
  preprocessing scripts, and utilities.
- [Current state](docs/current-state.md) for known caveats and open work.

Raw session notes are kept in `notes/` and historical docs in `archive/docs/`.
They are useful context, but they are not the source of truth.

## Repository Layout

```text
Croscim/
├── main.py              Hydra entry point
├── environment.yaml     Conda environment specification
├── config/              Hydra configs
├── contrib/SST/         Active SST datasets, model, solver, normalization
├── src/                 Base 4D-VarNet framework and utilities
├── scripts/             Local and Gefion launch scripts
├── data/                SST preprocessing scripts
├── tests/               Standalone validation scripts
├── tools/               Inspection and plotting utilities
├── docs/                Maintained documentation
├── notes/               Raw context and session notes
└── archive/             Legacy code and documentation
```

## Quickstart

Create or update the conda environment:

```bash
conda create -n croscim
conda activate croscim
conda install -c conda-forge mamba
mamba env update -f environment.yaml
```

Run the lightweight local smoke training:

```bash
./scripts/local/run_train_lite.sh 0
```

Run the main local training config:

```bash
./scripts/local/run.sh 0
```

Test a checkpoint:

```bash
./scripts/local/run_test_checkpoint.sh /path/to/checkpoint.ckpt
```

Launch a single-GPU Gefion experiment:

```bash
sbatch scripts/gefion/submit_gefion_single.sh exp_name
```

Launch the baseline or experimental ResUNet full-node Gefion run:

```bash
sbatch scripts/gefion/train_gefion.sh
sbatch scripts/gefion/train_gefion_resunet.sh
```

Some scripts contain machine-specific absolute paths. Read
[Workflows](docs/workflows.md) before using them on a new machine.
