---
title: Gefion HPC
---

# Submitting Jobs to the Queue System

## Introduction

Gefion HPC, also called Gefion Compute or simply Gefion, is a multi-user
environment. Jobs are managed by a scheduler, also called a resource manager.

Users do not run heavy jobs directly. Instead, they ask the system to run jobs,
usually by submitting a batch job.

## Batch Jobs

A batch job is a tool, program, or sequence of commands processed in batch mode.
Commands are listed in a file often called:

- batch script
- command file
- job script
- shell script

At DCAI, jobs are submitted and monitored with SLURM.

## Node Information

Gefion has approximately 190 NVIDIA DGX nodes.

One node has:

- 224 CPU cores with hyperthreading, 112 physical cores
- 2,064 GB of memory, about 2 TB
- 8 GPUs

## Access Gefion Compute

### Citrix Access

If you have access through Citrix-DCAI:

1. Connect to Citrix DCAI at `frontend.dcai.dk` with username, password, and
   two-factor authentication.
2. Open **Gefion HPC Desktop**.
3. Open **MobaXterm**.
4. Double-click one of the sessions, for example `login01.gefion.dcai.dk`.

## Submit a Batch Job

Use `sbatch`:

```bash
sbatch \
  --account=<DCAI group name> \
  --nodes=<no of nodes> \
  --ntasks-per-node=<no of CPUs per node> \
  --gpus=<total no of GPUs>
```

### Key sbatch Options

| Option | Meaning |
|---|---|
| `--account=<DCAI group name>` | Account/group to charge |
| `--nodes=<no of nodes>` | Number of nodes |
| `--ntasks-per-node=<no of CPUs per node>` | CPUs/tasks per node, range 1 to 224 |
| `--gpus=<total no of GPUs>` | Total GPUs, max 8 GPUs per node |
| `--mem=<memory>` | Memory request, for example `--mem=100GB` |
| `--time=<time>` | Wall time in minutes or `hours:minutes:seconds` |

### Memory and Time Defaults

- If you need a full node, specify `--mem=0`.
- Default CPUs per GPU: `--cpus-per-gpu=6`.
- Default memory per GPU: `--mem-per-gpu=31000`.
- Default wall time: `8:00:00`.

## Example: Create and Submit a Batch Script

Create `verify_python_setup.sh`:

```bash
#!/bin/bash

# Check if Python and scikit-learn are installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3."
    exit
else
    echo "Python3 is installed."
    python3 --version
fi

if ! python3 -c "import sklearn" &> /dev/null; then
    echo "scikit-learn is not installed. Please install scikit-learn using 'pip install scikit-learn'."
    exit
else
    echo "scikit-learn is installed."
fi
```

Submit it:

```bash
sbatch \
  --account=cu_0001 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --mem=10GB \
  --time=10 \
  verify_python_setup.sh
```

## Submit Many Batch Jobs

To avoid overloading the queue system, add a pause between submissions:

```bash
#!/bin/bash

num_jobs=20
sleep_time=30

for ((i=1; i<=num_jobs; i++)); do
    echo "Submitting job $i"
    sbatch -A cu_0001 -N 1 -n 1 --mem=10GB --time=60 job_$i.sh
    sleep "$sleep_time"
done
```

## Job Arrays

Job arrays submit many identical jobs in a single command using `-a`.

Example:

```bash
sbatch -a 1-100%10 -N 1 job.sh
```

Array syntax:

- `%10`: run at most 10 jobs concurrently.
- `1-100:10`: use a step size of 10.

## Monitoring and Managing Jobs

### Output Files

Specify output and error locations with `--output` or `-o`, and `--error` or
`-e`:

```bash
sbatch --output=output_file.txt --error=error_file.txt myscript.sh
```

### Check Job Status

```bash
squeue -u "$USER"
```

Detailed information for a specific job:

```bash
scontrol show job <jobid>
```

Status codes:

| Code | Meaning |
|---|---|
| `PD` | Pending |
| `R` | Running |
| `CD` | Completed |
| `F` | Failed |
| `TO` | Timeout |

### Cancel a Job

```bash
scancel <jobid>
```

### Resource Usage With sacct

Evaluate completed jobs and inspect memory/time usage:

```bash
sacct --format=JobID,JobName,State,MaxRSS,Elapsed,TotalCPU,AllocTRES
```

### Real-Time GPU Monitoring

Find the node name with `squeue`, for example `dgx011`.

```bash
ssh <node name>
nvtop
```

## Submit With SLURM Job Scripts

It is possible to set `sbatch` options inside a SLURM job script. The options
are specified as `#SBATCH` lines.

### Example SLURM Script

```bash
#!/bin/bash

#SBATCH --account=cu_0099
#SBATCH --job-name=job_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=100GB
#SBATCH --time=3:00:00
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

cd "$SLURM_SUBMIT_DIR"

module load GCCcore/12.3.0
module load Python-bundle-PyPI/2023.06

python /dcai/projects/cu_0099/scripts/python_script.py
```

The script has four sections:

1. `#!/bin/bash` tells the system this is a shell script.
2. `#SBATCH` lines specify requested resources.
3. Environment setup loads modules and changes directory.
4. Final commands execute the workload.

`%j` is the job ID placeholder. `%x` is the job name placeholder.

Submit the script:

```bash
sbatch <slurm_job_script>
```

## More sbatch Directives

| Directive | Description |
|---|---|
| `#SBATCH --nodes=<n>` | Request `<n>` nodes |
| `#SBATCH --ntasks-per-node=<p>` | CPU tasks per node |
| `#SBATCH --gpus=<number>` | Total number of GPUs |
| `#SBATCH --gpus-per-node=<n>` | GPUs per node |
| `#SBATCH --gpus-per-task=<n>` | GPUs per task |
| `#SBATCH --mem=<m>` | Total memory across all nodes, e.g. `4GB` |
| `#SBATCH --time=hh:mm:ss` | Maximum wall time |
| `#SBATCH --job-name=<job_name>` | Job name, no spaces |
| `#SBATCH --error=%x-%j.err` | Standard error file |
| `#SBATCH --output=%x-%j.out` | Standard output file |
| `#SBATCH --export=ALL` | Export all environment variables |
| `#SBATCH --begin=YYYY-MM-DDThh:mm` | Delay execution until a date/time |
| `#SBATCH --dependency=afterany:<jobid>` | Run only after another job finishes |
