# Software and Libraries

This page summarizes Gefion notes about Python environments, installed modules,
toolchains, communication libraries, and available software.

## Installing Python Packages

### Method 1: Create a New Python Environment

This method creates a standalone Python virtual environment and installs
packages with `pip`.

Start an interactive shell:

```bash
srun --pty -n 4 --gpus=1 --time=01:00:00 /bin/bash
```

Load the Python interpreter:

```bash
module load GCC/12.3.0 Python/3.11.3
```

Create a Python environment:

```bash
python -m venv $PWD/env_dir
```

Activate the environment:

```bash
source $PWD/env_dir/bin/activate
```

Install packages:

```bash
pip install plotly
pip list | grep plotly
```

To use the environment in another job, load the Python interpreter again and
activate the environment again.

### Method 2: Use a Pre-Installed Software Module

This method uses a pre-installed software module and installs additional
packages in a custom directory.

Start an interactive shell:

```bash
srun --pty -n 4 --gpus=1 --time=01:00:00 /bin/bash
```

Load a pre-installed software module:

```bash
module load foss/2023a PyTorch-Lightning/2.2.1-PyTorch-2.3.1-CUDA-12.4.0
```

Install new packages in a target directory:

```bash
pip install --target $PWD/target_dir plotly
```

Add the target directory to the interpreter:

```bash
export PYTHONPATH=$PWD/target_dir:$PYTHONPATH
pip list | grep plotly
```

To reload the environment, repeat the module load and `PYTHONPATH` export.

### Add a Python Package Directory Permanently

Add the export command to `~/.bashrc`:

```bash
vim ~/.bashrc
```

Add:

```bash
export PYTHONPATH=$PWD/target_dir:$PYTHONPATH
```

Save and reload:

```bash
source ~/.bashrc
```

## Scaling PyTorch on Multiple Nodes

Example script for high-performance communication using Open MPI, UCX, and
NCCL:

```bash
#!/bin/bash

#SBATCH --job-name=ddp_test
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=ddp_test_%j.out
#SBATCH --error=ddp_test_%j.err

module load foss/2023a PyTorch/2.3.1-CUDA-12.4.0

# Set the head address to the first node allocated by SLURM
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"

set -a
LOGLEVEL=INFO
CUDA_LAUNCH_BLOCKING=1
OMPI_MCA_pml=ucx
OMPI_MCA_btl=vader,tcp,openib,uct
UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
NCCL_SOCKET_IFNAME=ens6f0
NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_8,mlx5_9,mlx5_10,mlx5_11
OMP_NUM_THREADS=56
set +a

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=8 \
  --rdzv-id=$RANDOM \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$head_node_ip:25900 \
  multi_node_multi_gpu_test.py 1000 10
```

## Compiler Toolchains

Available toolchains in the HPC environment:

```bash
module load foss/2023a
module load foss/2023b
module load foss/2024a
module load NVHPC/24.5-CUDA-12.4.0
```

The NVIDIA HPC SDK is available up to `24.11-CUDA-12.6.0`.

## Communication Libraries

NCCL:

```bash
module load GCCcore/12.3.0 NCCL/2.18.3-CUDA-12.1.1
```

Newer NCCL versions are also available.

OpenMPI:

```bash
module load GCC/12.3.0 OpenMPI/4.1.5
```

OpenMPI is available up to `5.0.3`.

## Applications Reference List

Use `module spider <name>` to find the specific load command.

| Category | Software / Library |
|---|---|
| Data and build | Arrow, Bazel, CMake, dask, einops, freetype-py, ImageHash, imageio |
| ML and AI | ColabFold, Gradio, jax, jaxtyping, jsonargparse, Kornia, Neural-LAM, PyTorch, PyTorch-Lightning, PyTorch-Geometric, scikit-image, scikit-learn, torchvision, Transformers |
| Simulation and visualization | LAMMPS, Matplotlib, mediapy, bokeh, OpenCV, Seaborn, TensorBoard |
| Processing | mllam-data-prep, MMseqs2, motmetrics, neptune-client, SentencePiece, tokenizers |
| Cryo-EM / Bio | AreTomo2, Topaz, RELION, Xmipp, pyem, Scipion |
| Utilities | invisible-watermark, paho-mqtt, psycopg2, PyAV, PyOpenGL, PyYAML, slack-sdk, synology-api, timm, wandb, weather-model-graphs |
| Languages | Python, Python-bundle, R |
