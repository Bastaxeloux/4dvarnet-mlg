# NCCL Tests and GitLab

This page groups Gefion notes about NCCL benchmark tests and DCAI GitLab/code
repository services.

## NCCL Test

### Introduction

NVIDIA Collective Communications Library (NCCL) is a high-performance library
for multi-GPU and multi-node communication. It is commonly used by deep learning
frameworks such as TensorFlow, PyTorch, and MXNet.

NCCL optimizes GPU-to-GPU data transfers and is important for efficient
distributed training.

### Download the NVIDIA Benchmark Container

On a login node, move to a project directory:

```bash
cd /dcai/projects/<DCAI project name>
```

Example:

```bash
cd /dcai/projects/cu_0099
```

Home directory note: when accessing a login node via MobaXterm, users may start
in `/dcai/users/<DCAI username>`. The home directory is private and not meant
for large files. Use project storage for large datasets and project files, for
example `/dcai/projects/<DCAI project name>`.

Download and convert the NVIDIA HPC benchmark container into Gefion:

```bash
enroot import docker://docker.gefion.dcai.dk#nvidia/hpc-benchmarks:24.09
```

This creates a file similar to:

```text
nvidia+hpc-benchmarks+24.09.sqsh
```

in the directory where the command was run.

### Create an sbatch Script

Create an sbatch script such as `nccl_test.slurm`:

```bash
#!/bin/bash

#SBATCH --job-name=nccl_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=0:05:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --exclusive

export NCCL_SOCKET_IFNAME=ens6f0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export SHARP_COLL_ENABLE_PCT_RELAXED_ORDERING=1
export OMPI_MCA_coll_hcoll_enable=0

echo "NODELIST=${SLURM_NODELIST}"

srun --mpi=pmix \
  --cpu-bind=none \
  --mem-bind=none \
  --container-image=<container location>/nvidia+hpc-benchmarks+24.09.sqsh
```

Replace `<container location>` with the path where the container was downloaded,
for example under `/dcai/projects/cu_0099/...`.

### Run the NCCL Test

Submit the script:

```bash
sbatch --account=<group name> nccl_test.slurm
```

Monitor progress with `squeue`.

After completion, expected output files include:

- `dcgm-gpu-stats-<node_name>_gefion.dcai.dk-<SLURM_job_ID>.stats`
- `<job_name>.<SLURM_job_ID>.out`

## Code Repositories and GitLab

DCAI offers internal code repository hosting through GitLab. This supports
importing external code that would otherwise be blocked by compliance
restrictions preventing Internet access from Gefion.

### Key Services

- Local mirrors of external projects from GitHub, GitLab, Bitbucket, etc.
- Local hosting of internal projects.
- CI/CD pipelines.
- Container registry.

### Important Considerations

- External mirrors are one-way. Exporting code requires a separate agreement.
- Development should be done on a separate branch to prevent conflicts with
  mirrors.
- Syncs occur every 60 minutes by default.

## Access Tokens

### Generate an Access Token for GitHub.com

Visit:

```text
https://github.com/settings/personal-access-tokens/new
```

Minimum permissions:

- Contents: read only
- Metadata: read only

Provide the token value and name to Gefion support to set up a mirror.

### Generate an Access Token for DCAI GitLab

Use DCAI GitLab tokens to access locally hosted projects or for automation and
sharing without needing a full account.

## GitLab Container Registry

GitLab is configured as a container registry for Gefion.

Registry URL:

```text
https://gitlab.gefion.dcai.dk:5050
```

Internet assets are not available. Pipelines must use local resources.

### Pull Container Images From Local GitLab

Generate a deploy token with `read_registry` scope.

Login to local GitLab:

```bash
TOKEN=gldt-AAABbbbb5555hghg73jdk
echo "$TOKEN" | docker login gitlab.gefion.dcai.dk:5050 \
  -u deploy_token_name \
  --password-stdin
```

Use Enroot to pull the image.

Create a credential file in `~/.config/enroot/.credentials`:

```text
machine gitlab.gefion.dcai.dk login deploy_token_name password gldt-AAABbbbb5555hghg73jdk
```

Set the environment variable and import:

```bash
export ENROOT_CONFIG_PATH=~/.config/enroot/
enroot import docker://gitlab.gefion.dcai.dk:5050/#cu_0000/projectx/busybox:latest
```
