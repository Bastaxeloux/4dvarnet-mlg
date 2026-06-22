# Gefion Documentation Notes

This directory contains cleaned Markdown notes from Gefion cluster
documentation. The text was reconstructed from photos/OCR, so treat it as a
local reference and verify critical details against official Gefion
documentation or support before production runs.

Use this README to choose the relevant page instead of reading every file.

## Files

| File | Use it for |
|---|---|
| [submit.md](submit.md) | SLURM basics, batch jobs, `sbatch`, job arrays, `squeue`, `sacct`, job scripts |
| [Interactive_session.md](Interactive_session.md) | Interactive `srun` sessions, login-node safety, tmux workflow |
| [librairies.md](librairies.md) | Python environments, modules, PyTorch multi-node example, toolchains, software list |
| [data_and_slurm.md](data_and_slurm.md) | Data import/export, DCAI SFTP, `rclone`, SLURM accounting reports |
| [nccl_and_gitlab.md](nccl_and_gitlab.md) | NCCL benchmark tests, Enroot benchmark container, DCAI GitLab and registry |
| [usefull_commands.md](usefull_commands.md) | General terminal commands useful on Gefion |

## For Croscim Agents

Most Croscim-specific run commands are documented in
[../workflows.md](../workflows.md). Use these Gefion notes only when you need
cluster-level details, such as:

- how SLURM resource requests work
- how to start an interactive session
- how to load modules or create Python environments
- how to debug multi-node/NCCL communication
- how to transfer data into Gefion
- how to submit and monitor jobs at the cluster level

The Croscim-specific TensorBoard-over-MobaXterm procedure and dedicated
baseline/ResUNet launch commands are in [../workflows.md](../workflows.md).

Known Croscim Gefion paths are documented in
[../configuration.md](../configuration.md) and [../current-state.md](../current-state.md).
