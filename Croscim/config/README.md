# config

Hydra configuration for Croscim.

Root config:

- `main.yaml`: requires `xp=...` and calls configured `entrypoints`.

Active SST configs:

- `xp/SST/multires.yaml`
- `xp/SST/multires_lite.yaml`
- `xp/SST/multires_lite_ddp.yaml`
- `xp/SST/multires_gefion.yaml`
- `xp/SST/multires_single_gefion.yaml`

Legacy:

- `xp/SST/base_sst.yaml`

Before long training, verify solver dimensions against
`contrib/SST/models.py::format_batch_for_solver`.

See [../docs/configuration.md](../docs/configuration.md).
