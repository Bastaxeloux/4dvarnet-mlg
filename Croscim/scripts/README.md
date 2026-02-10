# Scripts

Shell scripts for training, testing, and monitoring. Organized by target machine.

## Local machine (DMI ohm server)

Located in `local/`. The machine has 64 CPU cores and 4x Nvidia A40 (40 GB each).

| Script | Usage | What it does |
|---|---|---|
| `run.sh` | `./scripts/local/run.sh [GPU_ID] [CKPT_PATH]` | Main training launcher. Runs `multires` config on a single GPU in background (`nohup`). If a checkpoint path is given, resumes from it. |
| `run_train_lite.sh` | `./scripts/local/run_train_lite.sh [GPU_ID]` | Quick pipeline test with `multires_lite` config (3 epochs, 20 batches). Useful for debugging changes before a real run. |
| `run_train_ddp.sh` | `./scripts/local/run_train_ddp.sh` | Multi-GPU DDP training (4 GPUs). Experimental — not fully stable yet. |
| `run_test_checkpoint.sh` | `./scripts/local/run_test_checkpoint.sh /path/to/ckpt` | Evaluate a saved checkpoint on the test set. Runs in foreground. |
| `kill_train.sh` | `./scripts/local/kill_train.sh` | Kills all running training processes (`pkill`). |
| `tensorboard.sh` | `./scripts/local/tensorboard.sh [PORT]` | Launches TensorBoard on the results directory. Default port 6006. |

## Gefion HPC cluster

Located in `gefion/`. The cluster has 2 nodes x 8 H100 GPUs (80 GB each). No internet access.

| Script | Usage | What it does |
|---|---|---|
| `submit_gefion_single.sh` | `sbatch scripts/gefion/submit_gefion_single.sh EXP_NAME [OVERRIDES...]` | **Main Gefion script.** SLURM job for single-GPU experiments. Accepts Hydra overrides for hyperparameter sweeps (learning rate, loss weights, network size, etc.). |
| `train_gefion.sh` | `sbatch scripts/gefion/train_gefion.sh` | SLURM job for multi-GPU DDP training (6 H100s). |
| `run_gefion.sh` | `./scripts/gefion/run_gefion.sh` | Interactive multi-GPU launcher (no SLURM). |
| `run_gefion_single.sh` | `./scripts/gefion/run_gefion_single.sh [GPU_ID]` | Interactive single-GPU launcher (no SLURM). |
| `run_test_checkpoint_gefion.sh` | `./scripts/gefion/run_test_checkpoint_gefion.sh /path/to/ckpt [GPU_ID]` | Test a checkpoint on Gefion. |

## Gefion setup

Before running on Gefion, load the required modules and activate the environment:

```bash
module load GCC/12.3.0 Boost/1.82.0 snappy/1.1.10 GSL/2.7 Eigen/3.4.0 CUDA/12.8.0 Python/3.11.3
source /dcai/projects/cu_0026/croscim_env/bin/activate
cd /dcai/projects/cu_0026/Croscim
export PYTHONPATH=$PWD:$PYTHONPATH
```
