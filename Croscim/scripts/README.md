# scripts

Launch scripts for local DMI/Ohm and Gefion.

Local scripts:

- `local/run_train_lite.sh`: short smoke training.
- `local/run.sh`: main local training, optional checkpoint resume.
- `local/run_train_ddp.sh`: local DDP experiment.
- `local/run_test_checkpoint.sh`: checkpoint evaluation.
- `local/tensorboard.sh`: TensorBoard.
- `local/kill_train.sh`: stop background training processes.

Gefion scripts:

- `gefion/submit_gefion_single.sh`: SLURM single-GPU experiments.
- `gefion/train_gefion.sh`: SLURM DDP training.
- `gefion/test_checkpoint_gefion.slurm`: SLURM mono-GPU checkpoint test.
- `gefion/run_gefion.sh`: interactive multi-GPU launcher.
- `gefion/run_gefion_single.sh`: interactive single-GPU launcher, needs review.
- `gefion/run_test_checkpoint_gefion.sh`: checkpoint test worker called by the
  SLURM wrapper; do not run on login nodes without an allocation.

Many scripts contain absolute paths. Read
[../docs/workflows.md](../docs/workflows.md) before running on a new machine.
