# src

Base 4D-VarNet framework code shared by the SST implementation.

Useful files:

- `train.py`: Hydra training entrypoints.
- `test.py`: Hydra test entrypoint.
- `models.py`: generic Lightning model, generic solver pieces, metrics hooks.
- `ConvLSTM.py`: standalone ConvLSTM implementation.
- `utils.py`: temporal weights, metrics, and multi-resolution patch geometry.
- `versioning_cb.py`: Lightning callback for git/code version tracking.

The active SST training logic overrides much of the generic model behavior in
`contrib/SST/models.py`.

See:

- [../docs/architecture.md](../docs/architecture.md)
- [../docs/data.md](../docs/data.md)
