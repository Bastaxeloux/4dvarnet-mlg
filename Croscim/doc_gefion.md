# Gefion HPC Wiki

## Submitting jobs to the queue system

### Introduction

Gefion HPC, referred to as Gefion Compute or just Gefion, is a multi-user environment. To ensure effective usage of the available resources, a job-scheduler (aka resource manager) is used to manage the workload. The job-scheduler assigns resources to users according to the system's current and expected load. Meaning that users do not run their job(s) directly, but instead "ask" the system to run their job(s), usually by the means of a batch job, see the how to section below.

### Batch job

A batch job is a tool/program or set of tools/programs processed in batch mode. This means that a sequence of commands, to be executed by the operating system, is listed in a file, often referred to as a batch script, command file, job script, or shell script. The batch script is submitted for execution, or sent to the job queue, as a single unit. The job-scheduler parses the script and tries to optimize the usage of the available resources, scheduling the execution of all incoming batch jobs at different times, and on different nodes/cores in the Gefion cluster. At DCAI, SLURM is used for submitting and monitoring jobs.

The opposite of a batch job is interactive processing, in which a user enters individual commands to be processed immediately. Here is a guide on how to start an interactive session. An interactive session is suitable for developing and testing a batch script.

### Node information Gefion

In Gefion there are approximately 190 Nvidia DGX nodes available.

A node is equipped with:

- 224 CPU cores (using hyperthreading, 112 physical cores)
- 2,064 gigabytes (GB), approximately two terabytes (2TB), of memory
- 8 Graphical Processing Units (GPUs)

## User creation and modification

As a user of DCAI services and Gefion, you have access to submitting jobs to the queue system by default.

### Logging into the system for the first time

No special requirements.

### Username and password

There is no password restriction on this solution, as long as you have access to Gefion Compute, you have access to submitting batch jobs.

## How to

### Access Gefion Compute

First you need to access Gefion Compute.

#### Citrix access

If you have access to Gefion Compute through Citrix-DCAI, and the access was set up according to the user guide:

- Connect to Citrix-DCAI (frontend.dcai.dk) with your username, password and two factor authentication in the Entrust Identity app.
- Open Gefion HPC Desktop (it may automatically launch)
- Open MobaXterm and double click one of the Custom sessions, e.g. `login01.gefion.dcai.dk`

You are now on a login node on Gefion and can submit batch jobs.

### How to submit a batch job/script

You can submit batch jobs via the command `sbatch`. The general command for submitting a job/script into the queue system is as follows:

```bash
sbatch --account=<DCAI group name> --nodes=<no of nodes> --ntasks-per-node=<no of CPUs per node> --gpus=<total no of GPUs>
```

The sbatch options specified in the above command are the following:

1. The group name of your team/project: `--account=<DCAI group name>`
2. How many nodes are needed: `--nodes=<no of nodes>`
3. How many CPUs are needed: `--ntasks-per-node=<no of CPUs per node>` Can range from 1 to 224.
4. How many GPUs are needed for the full job: `--gpus=<total no of GPUs for job>` There are maximum 8 GPUs per node.
5. How much memory the job requires: `--mem=<memory>` is specified in MB. SLURM accepts memory specifications in various units, including K, M, G, and T for kilobytes, megabytes, gigabytes, and terabytes, respectively, e.g. `--mem=100GB`.
6. How long time you expect the job to run: `--time=<time>` with <time> specified in minutes (i.e. `time=60` corresponds to 60 minutes). Time can also be specified in the format hours:minutes:seconds (i.e. `time=01:00:00` corresponding to an hour).

For more `sbatch` options see the manual page of the command:

```bash
man sbatch
```

#### Memory and Time:

If you need a full node, not sharing memory resources with other jobs, you should specify `--mem=0`. It is not enough to use the `--exclusive` option.

Do not request all the memory of a node as the node OS uses some of the memory. Allocating maximum 2TB of memory to the actual job would be OK.

If you do not specify the `--time` option, the default wall time for a job is eight hours (8:00:00).

The default number of CPUs per GPU is six `--cpus-per-gpu=6`.

The default amount of memory per GPU is 31,000MB `--mem-per-gpu=31000`.

#### Some sbatch examples:

```bash
# Submitting a job for group "group_VIP" with 1 CPU, and 10 min max runtime
sbatch --account=group_VIP --nodes=1 --ntasks-per-node=1 --time=10 script.sh

# Submitting a job with for group "group_VIP" with 20 CPUs, 100GB memory, 2 GPUs per node, and 60 min max runtime
sbatch --account=group_VIP --nodes=1 --ntasks-per-node=20 --mem=100GB --gpus-per-node=2 --time=60 script.sh

# For more options on using sbatch
man sbatch
```

**Note**, the first line in your batch script must start with `#!` followed by the path to an interpreter. For instance `#!/bin/bash` for a bash shell script.

### Example on how to create and submit a batch script with sbatch

1. Create a shell script using a text editor, e.g. Vim.
   - Open (or create and open if not already existing) the script `verify_python_setup.sh` in Vim using this command:
   
   ```bash
   vim verify_python_setup.sh
   ```
   
   - Press `i` for insert.
   - Add your commands to the script.
   
   Example of a simple shell script that checks python installation and access to sklearn python package:
   
   ```bash
   #!/bin/bash
   # Check if Python and scikit-learn are installed
   if ! command -v python3 &> /dev/null; then
       echo "Python3 is not installed. Please install python3."
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
   
   - Press `Esc` to exit the insert mode in Vim and press `:x` to exit Vim and save your changes.

2. Submit your shell script using sbatch.

   ```bash
   # Submit shell script with sbatch
   sbatch --account=<mygroup> --nodes=1 --ntasks-per-node=1 --mem=10GB --time=10 verify_python_setup.sh
   # e.g. for group cu_0001
   sbatch --account=cu_0001 --nodes=1 --ntasks-per-node=1 --mem=10GB --time=10 verify_python_setup.sh
   ```

### Submitting many batch jobs simultaneously

Not to overload the queue system on Gefion, and preventing your batch jobs from being blocked, it is a good idea to incorporate a pause between multiple batch job submissions. In a shell script you can use the command "sleep ", to make the execution of the next command delay for a specified number of seconds. Here is an example of a shell script with an incorporated pause in between batch job submissions:

```bash
#!/bin/bash

# Number of jobs to submit
num_jobs=20

# Time to sleep between submissions (in seconds)
sleep_time=30

for ((i=1; i<=num_jobs; i++)); do
    echo "Submitting job $i"
    # Submit the job using sbatch
    sbatch -A cu_0001 -N 1 -n 1 --mem=10GB --time=60 job_$i.sh
    # Sleep for the specified time
    sleep $sleep_time
done
```

Job arrays can also be used to control submissions. Job arrays serve as a mechanism to submit many identical or similar jobs in a single job submission command. SLURM will then launch multiple jobs based on the job array specification.

This is done with the `-a` switch to sbatch, followed by a range of array ids.

The range can be linear like 1-100 or a comma separated list like 1,5,15,10,100.

Two modifiers are available:

`%` is used to limit the number of concurrently running jobs, for example `100%10` means 100 total jobs, but at max 10 concurrently running.

`:` is used to specify a step size when dealing with ranges, for example `1-100:10` would start the job array ids at 1, then go to 11, 21, 31 etc.

Example: `sbatch -a 1-100%10 -N1 job.sh`

```
START = 1
END = 100
STEP = 10
```

This will submit a job array with 100 total jobs to be run. The %10 part tells SLURM to run at maximum 10 jobs at any point in time, this is helpful to stay within usage limits while submitting large numbers of array jobs.

Job arrays will have additional environment variables set:

- SLURM_ARRAY_JOB_ID will be set to the first job ID of the array.
- SLURM_ARRAY_TASK_ID will be set to the job array index value.
- SLURM_ARRAY_TASK_COUNT will be set to the number of tasks in the job array.
- SLURM_ARRAY_TASK_MAX will be set to the highest job array index value.
- SLURM_ARRAY_TASK_MIN will be set to the lowest job array index value.

These can be used in the job script to identify which particular array job is running the script and then vary the input, output or other files and parameters.

Full job array documentation: https://slurm.schedmd.com/job_array.html

### How to monitor, manage and cancel submitted jobs

In the following, you will find useful commands in regards to monitor, manage and cancel submitted jobs.

#### Output files

You can specify a directory for the output files using the `--chdir` or `-D` option in SLURM.

Here is an example on how to specify the output:

```bash
sbatch -A <mygroup> -N 1 -n 1 --mem=10GB --chdir="path/to/output/folder" myscript.sh
# Or
sbatch -A <mygroup> -N 1 -n 1 --mem=10GB -D "path/to/output/folder" myscript.sh
```

If not specified, the output files will be placed in the directory where the sbatch command was submitted. For example, if you submit your job from the `/dcai/users/jandoe/job_scripts` directory, the output files will be created in that directory. This applies to files generated in the script without a specified path, as well as the standard output and standard error files.

You can explicitly specify the output and error file locations using the `--output` and `--error` options, respectively. For example:

```bash
sbatch -A <mygroup> -N 1 -n 1 --mem=10GB --output=output_file.txt --error=error_file.txt myscript.sh
```

In the standard out files you can find useful information in regards to debugging your batch script as they will contain output text and error messages from your batch commands.

**Note:**

Without specifying the `--error` option, SLURM defaults to merging standard output (stdout) and standard error (stderr) into a single file. e.g. "slurm-105645.out".

#### Job name

You can specify a name for a job in SLURM using the `--job-name` (or `-J`) option in the sbatch command. This helps you identify your job more easily when you check the queue or job status. Here is an example:

```bash
sbatch --job-name=my_job_name myscript.sh
```

In this example, "my_job_name" is the name you have assigned to the job.

When you check the job queue using e.g. `squeue` or any other job management command, you will see the job listed with the specified name. This can make it easier to manage and track your jobs.

#### Check job status

To check the status of jobs that are currently running, pending, or recently completed you can use the `squeue` command:

```bash
# Basic command
squeue
# Showing only your own jobs
squeue -u $USER
# With a more verbose specified output format
squeue --format="%.18i %.9P %.16j %.8u %.2t %.10M %.6D %.10C %.10m %.8l %.6T %.10r %R"
# For more info about squeue
man squeue
```

Some codes for the the status (ST) of jobs:

- PD (PENDING): The job is waiting in the queue and has not yet started running.
- R (RUNNING): The job is currently running.
- CD (COMPLETED): The job has finished successfully.
- F (FAILED): The job terminated with a non-zero exit code.
- TO (TIMEOUT): The job reached its time limit and was terminated.
- CA (CANCELLED): The job was cancelled by the user or the system administrator.
- NF (NODE_FAIL): The job terminated due to a failure of one or more allocated nodes.

To get detailed information about a specific job:

```bash
scontrol show job <jobid>
# E.g.
scontrol show job 105974
```

You can see the jobid of your submitted job when submitting.

```
jandoe@login02:~/scripts$ sbatch -A cu_0002 -N 1 -n 1 -t 10 verify_python_setup.sh
Submitted batch job 19079
```

You can also find the jobID of your job(s) using `sacct`.

Note, the `scontrol` information is only available during and a short time after the completion of your job.

#### Cancel a submitted job

To cancel a submitted job you can use the command:

```bash
scancel <jobid>
# E.g.
scancel 105952
```

#### Estimating job resource requirements

The job-scheduler reserves the requested resources for your job, and run it when there are enough resources (avoiding conflicts with other users' jobs). It is, therefore, important that the resources requested correspond to what really is needed by your job.

First time you run your job you may not have a clear picture of resources needed. To get a rough estimate, you can submit a job to a full node, with a large time limit (e.g. 48 hours).

```bash
# Test run using a full node, to estimate resources needed
sbatch -A <mygroup> -N 1 -n 224 time=48:00:00 job.sh
```

When your job has completed you can check the actual resource usage.

To evaluate the completed job(s) you can use the `sacct` command:

```bash
# Basic command
sacct
# With specified output format
sacct --format=JobID,JobName,State,MaxRSS,MaxVMSize,Elapsed,TotalCPU,AllocTRES
# For more info about sacct
man sacct
```

With the `sacct` command you can check for example used time and memory (see "Elapsed" and "MaxRSS") of a completed job.

The `sacct` will by default show jobs run after 00:00:00 on the same day. You can override this by setting a start time using the option `--starttime` or `-S`:

```bash
sacct -S 2025-01-01 --format=JobID,JobName,State,MaxRSS,MaxVMSize,Elapsed,TotalCPU,AllocTRES
# Also showing when job started and finished
sacct -S 2025-01-01 --format=JobID,JobName,State,MaxRSS,MaxVMSize,Elapsed,TotalCPU,AllocTRES,Start,End
```

You can also set an end time using the option `--endtime` or `-E` to limit jobs displayed:

```bash
sacct -S 2025-01-01 -E 2025-01-05 --format=JobID,JobName,State,MaxRSS,MaxVMSize,Elapsed,TotalCPU,AllocTRES,Start,End
```

When assigning memory to your future submissions of the same batch job/script, it is recommended to ad some extra memory resources to your job (e.g. 10-20% extra) as the resource usage can fluctuate somewhat from run to run.

#### Check resources used in real-time

To check the resources used by a job in real-time, you can use the tool `nvtop`.

After having submitted your SLURM job using `sbatch`, you can check submitted and running jobs with the `squeue` command.

```bash
squeue -u jandoe
```

This will give you the name of the node (red circle) to which the job was submitted.

```
jandoe@login02:~$ squeue -u jandoe
        JOBID PARTITION     NAME   USER ST     TIME  NODES NODELIST(REASON)
        35172   defq nccl_tes  jandoe  R     0:05      1 dgx011
```

While the job is still running you can SSH into the specific node:

```bash
ssh <node name>
# E.g.
ssh dgx011
```

When on the node use the `nvtop` command to check the resources being used.

```bash
nvtop
```

Use `exit` to leave the compute node.

## Interactive session

### Introduction

When working in an interactive session, a defined set of Gefion's resources are allocated to you. Consequently, you cannot by mistake use all the available resources of the login node, e.g. by running a very heavy tool or command. Contrary to when sending a job to the queue system as a batch job, you will see error messages and standard output of your commands directly in the terminal.

An interactive session is a good way of trying out your code before submitting your commands or script as a batch job using sbatch.

**Note**

If you run compute heavy tools and commands directly from one of the login nodes, these can be subjected to enforced shut down by the Gefion team without notice.

### How to use an interactive session

#### Citrix access

If you have access to Gefion Compute through Citrix-DCAI, and the access was set up according to the user guide:

- Connect to Citrix-DCAI (frontend.dcai.dk) with your username, password and two factor authentication in the Entrust Identity app.
- Open Gefion HPC Desktop (it may automatically launch)
- Open MobaXterm and double click one of the Custom sessions, e.g. `login01.gefion.dcai.dk`

You are now on a login node on Gefion and can start an interactive session.

#### Start interactive session

You can then start an interactive session using the following `srun` command:

```bash
# Start an interactive session on Gefion
srun --pty -N 1 -n 1 --mem=1000 --time=4:00:00 /bin/bash
```

Requesting one node, one task (1CPU), and 1000 MB of memory, while providing a pseudo-terminal for interaction.

This gives you access to an interactive sessions where you can run commands and see immediate output.

You can see that `<username>@login<node_no>` has changed to `<username>@<compute_node>` in this case compute node `dgx018`.

```
[jandoe @login02 ~]$ srun --pty -N 1 -n 1 --mem=1000 --time=4:00:00 /bin/bash
[jandoe @dgx018 ~]$
```

This means that you have access to the above specified resources on a compute node.

**Note**

Your interactive session will close down after the time chosen and all running processes within the session will be aborted. If time is not set in the srun command, the default value of the time option is eight hours.

To exit your interactive session write "exit" and press **Enter**.

```bash
# exit interactive session
exit
```

To check the status of your interactive session (and other jobs), use the `squeue` command:

```bash
# Check running jobs
squeue
# Check Your running jobs
squeue -u $USER
```

If you do not know (or have forgotten) your group's name on Gefion, you can write "id" in the terminal to see info about your user, including the group(s) you belong to.

```bash
# See user information
id
```

To prevent disconnection issues when running an interactive session, you can start a tmux session and run the `srun` command inside it. This way, if your connection drops, you can reattach to the tmux session and continue working.

```bash
tmux
srun --pty -N 1 -n 1 --mem=1000 --time=4:00:00 /bin/bash
```

To detach from tmux session, press `Ctrl`+`b` and then `d`.

To attach to latest tmux session (with your running interactive session), use the command:

```bash
tmux a
```

To exit both the interactive session and the tmux session write `exit` twice.

You should now be able to run interactive sessions on Gefion through the Terminal.

## Software and Libraries

### Getting Started

#### Installing Python Packages

##### Method 1: Creating a New Python Environment

This method involves creating a standalone Python virtual environment and installing packages via pip.

Follow these steps:

1. Start an interactive shell:

   ```bash
   srun --pty -n 4 --gpus=1 --time=01:00:00 /bin/bash
   ```

2. Load the Python interpreter:

   ```bash
   module load GCC/12.3.0 Python/3.11.3
   ```

3. Create a Python environment:

   ```bash
   python -m venv $PWD/env_dir
   ```

4. Activate the environment and install Python packages:

   ```bash
   source $PWD/env_dir/bin/activate
   ```

5. Install Python packages:

   ```bash
   pip install plotly
   pip list | grep plotly
   ```

6. To use the environment in another job, repeat steps **2** and **4** to load the interpreter and activate the environment.

##### Method 2: Using a Pre-Installed Software Module

This method utilises a pre-installed software module and allows you to install additional packages in a custom directory.

1. Start an interactive shell:

   ```bash
   srun --pty -n 4 --gpus=1 --time=01:00:00 /bin/bash
   ```

2. Load a pre-installed software module:

   ```bash
   module load foss/2023a PyTorch-Lightning/2.2.1-PyTorch-2.3.1-CUDA-12.4.0
   ```

3. Install new packages in a target directory:

   ```bash
   pip install --target $PWD/target_dir plotly
   ```

4. Add the target directory to the interpreter:

   ```bash
   export PYTHONPATH=$PWD/target_dir:$PYTHONPATH
   pip list | grep plotly
   ```

5. To reload the environment, repeat steps **2 and 4** to load the module and set the `PYTHONPATH`.

##### Adding Python package directory permanently

To make the change of the $PYTHONPATH permanent, you can add the export command to your shell configuration file (e.g., .bashrc, .bash_profile, .zshrc, etc.). On a Unix-based systems (Linux/macOS), open your shell configuration file (e.g., .bashrc) in a text editor:

```bash
vim ~/.bashrc
```

Add the following line to the file:

```bash
export PYTHONPATH=$PWD/target_dir:$PYTHONPATH
```

Save the file and reload the shell configuration:

```bash
source ~/.bashrc
```

### Scaling PyTorch on Multiple Nodes

This batch script example demonstrates how to configure environment variables for high-performance communication libraries, including **Open MPI**, **UCX**, and **NCCL**, to enable efficient distributed training in PyTorch. These configurations optimise inter-node and intra-node communication, ensuring scalability and performance across multi-node setups.

```bash
#!/bin/bash

#SBATCH --job-name=ddp_test
#SBATCH --nodes=4#SBATCH --ntasks=4
#SBATCH --gpus-per-task=8#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --output=ddp_test_%j.out    # Standard output
#SBATCH --error=ddp_test_%j.out     # Redirect stderr to stdout

module load foss/2023a PyTorch/2.3.1-CUDA-12.4.0

# Set the head address to the first node allocated by SLURM
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

set -a
LOGLEVEL=INFO
CUDA_LAUNCH_BLOCKING=1
OMPI_MCA_pml=ucx
OMPI_MCA_btl=^vader,tcp,openib,uct
UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
NCCL_SOCKET_IFNAME=ens6f0
NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
OMP_NUM_THREADS=56
set +a

srun torchrun \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node 8 \
    --rdzv-id $RANDOM \
    --rdzv-backend c10d \
    --rdzv-endpoint $head_node_ip:25900 \
    multi_node_multi_gpu_test.py 1000 10
```

## Applications

This is a list of some of the tools and applications available on the cluster, along with their descriptions and usage instructions.

### Arrow

Apache Arrow is a cross-language development platform for in-memory data.

```bash
module load GCC/12.3.0
module load SciPy-bundle/2023.07 Arrow/14.0.1
# Or another version
module load SciPy-bundle/2024.05 Arrow/16.1.0
```

### Bazel

Bazel is a build tool that builds code quickly and reliably. It is used to build the majority of Google's software.

Versions:

```bash
module load GCC/12.3.0 Bazel/6.3.1
```

### bokeh

Bokeh is an interactive visualization library for Python that enables the creation of interactive plots, dashboards, and data applications in web browsers.

Versions:

```bash
module load GCC/12.3.0
module load SciPy-bundle/2023.07 bokeh/3.2.2
# Or other version
module load SciPy-bundle/2024.05 bokeh/3.4.1
```

### CMake

CMake, the cross-platform, open-source build system. CMake is a family of tools designed to build, test and package software.

Versions:

```bash
module load GCCcore/12.3.0 CMake/3.26.3
```

## Compiler Toolchains

The following commands load various compiler toolchains available in the HPC environment.

```bash
module load foss/2023a
module load foss/2023b
module load foss/2024a
module load NVHPC/24.5-CUDA-12.4.0
module load NVHPC/24.7-CUDA-12.5.0
module load NVHPC/24.9-CUDA-12.6.0
module load NVHPC/24.11-CUDA-12.6.0
```

## Communication Libraries

### NCCL (NVIDIA Collective Communications Library)

The NVIDIA Collective Communications Library (NCCL) implements multi-GPU and multi-node collective communication primitives that are performance optimised for NVIDIA GPUs.

Versions:

```bash
module load GCCcore/12.3.0 NCCL/2.18.3-CUDA-12.1.1
module load GCCcore/12.3.0 NCCL/2.18.3-CUDA-12.4.0
module load GCCcore/13.2.0 NCCL/2.20.5-CUDA-12.5.0
module load GCCcore/13.3.0 NCCL/2.20.5-CUDA-12.6.0
module load GCCcore/13.3.0 NCCL/2.22.3-CUDA-12.6.0
```

### OpenMPI (Open Message Passing Interface)

The Open MPI Project is an open source MPI-3 implementation.

Versions:

```bash
module load GCC/12.3.0 OpenMPI/4.1.5
module load GCC/13.2.0 OpenMPI/4.1.6
module load GCC/13.3.0 OpenMPI/5.0.3
```

## Error handling

Nothing relevant.

## Support

If you experience issues please start by consulting our FAQ - Frequently Asked Questions page.

You can also check out our News & Announcements page if there are any recently occurring issues.

You can get help with issues related to the DCAI services by writing an e-mail to support@dcai.dk. You can significantly speed up the support process by providing complete and detailed information for faster troubleshooting and error correction.