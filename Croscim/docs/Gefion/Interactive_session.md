# Interactive Sessions

## Introduction

In an interactive session, a defined set of Gefion resources is allocated to
you. This prevents accidental heavy use of the login node.

Unlike a batch job submitted with `sbatch`, an interactive session shows errors
and standard output directly in the terminal. It is useful for trying code
before submitting a full batch job.

> **Note:** Heavy commands run directly on login nodes may be shut down by the
> Gefion team without notice.

## User Creation and Modification

DCAI users with access to Gefion have default access to interactive sessions.

## First Login

No special requirements are listed for first login.

## Username and Password

There is no password restriction specific to interactive sessions. If you have
access to Gefion, you can run interactive sessions.

## How to Use an Interactive Session

### Access Gefion

First, access Gefion Compute.

#### Citrix Access

If you have access through Citrix-DCAI:

1. Connect to Citrix-DCAI at `frontend.dcai.dk` with username, password, and
   two-factor authentication in the Entrust Identity app.
2. Open **Gefion HPC Desktop**. It may automatically launch.
3. Open **MobaXterm**.
4. Double-click one of the custom sessions, for example
   `login01.gefion.dcai.dk`.

You are now on a Gefion login node and can start an interactive session.

### Start an Interactive Session

Use `srun`:

```bash
srun --pty -N 1 -n 1 --mem=1000 --time=4:00:00 /bin/bash
```

This requests:

- one node
- one task, equivalent to one CPU here
- 1000 MB of memory
- a pseudo-terminal for interaction
- a four-hour time limit

The prompt changes from a login node to a compute node. Example:

```text
[jandoe@login02 ~]$ srun --pty -N 1 -n 1 --mem=1000 --time=4:00:00 /bin/bash
[jandoe@dgx018 ~]$
```

This means the specified resources are available on compute node `dgx018`.

If `--time` is not set, the default wall time is eight hours. When the session
time ends, the interactive session closes and running processes in that session
are aborted.

### Exit an Interactive Session

```bash
exit
```

## Check Session or Job Status

All jobs:

```bash
squeue
```

Your jobs:

```bash
squeue -u "$USER"
```

If you do not know the Gefion group name, use:

```bash
id
```

## Prevent Disconnection Issues With tmux

Start `tmux` before launching the interactive session:

```bash
tmux
srun --pty -N 1 -n 1 --mem=1000 --time=4:00:00 /bin/bash
```

Detach from tmux:

```text
Ctrl+b then d
```

Reattach to the latest tmux session:

```bash
tmux a
```

To exit both the interactive session and tmux, run `exit` twice.

## Conclusion

You should now be able to run interactive sessions on Gefion through the
terminal.

## Related

- Submitting jobs to the queue system: [submit.md](submit.md)

## Error Handling

No specific error-handling notes were included in the source text.
