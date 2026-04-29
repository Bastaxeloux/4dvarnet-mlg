# Useful Terminal Commands

This page lists terminal commands that may be useful when accessing Gefion
through a terminal.

## Basic Commands

| Command | Description |
|---|---|
| `man <program>` | Show a command manual and available options |
| `man ls` | Show the manual for `ls` |
| `man -k string` | Search manual page names/descriptions |
| `pwd` | Print current working directory |
| `ls` | List files in current directory |
| `ls -ltrh` | Long listing, human-readable sizes, reverse time order |
| `mkdir <directory>` | Create a directory |
| `cd <path>` | Change directory |
| `cd ..` | Move up one directory |
| `cd ../..` | Move up two directories |
| `cd -` | Return to previous directory |
| `cd` or `cd ~` | Go to home directory |
| `cp <file> <path>` | Copy a file |
| `cp -r <directory> <path>` | Copy a directory recursively |
| `mv <file/dir> <path>` | Move a file or directory |
| `mv <old_name> <new_name>` | Rename a file or directory |
| `rm <file>` | Remove a file |
| `rm -r <directory>` | Remove a directory recursively |
| `history` | Show recently used commands |
| `!<number>` | Run previous command by history number |
| `bc` | Basic terminal calculator, type `quit` to exit |
| `wc` | Word count, use `-l` for line count |

Be careful with `rm` and especially recursive/forced variants.

## Find and Compare

| Command | Description |
|---|---|
| `find . -name "*pattern*"` | Search filenames containing pattern, case-sensitive |
| `find . -iname "*pattern*"` | Search filenames containing pattern, case-insensitive |
| `find ~ -type f -mtime -2` | Find files modified in home during the last two days |
| `ln -s <original_file> <new_file>` | Create a symbolic link |
| `cmp <file1> <file2>` | Test whether two files are identical |
| `diff <file1> <file2>` | Show differences between files |
| `sdiff -s <file1> <file2>` | Side-by-side diff showing only differing lines |

## Monitor Processes and Resources

| Command | Description |
|---|---|
| `df -h` | Disk free space per volume, human-readable |
| `du -sh * \| sort -hr \| head -n10` | Show largest files/directories in current directory |
| `free -g` | Memory information in GB |
| `top` | Show top CPU/memory consumers |
| `htop` | More detailed process monitor |
| `who` | Show logged-in users |
| `w` | Show logged-in users and activity |
| `ps -u <user>` | Show processes for one user |
| `ps aux \| grep <string>` | Search processes by text |
| `kill <process-ID>` | Send default termination signal |
| `kill -9 <process-ID>` | Force kill a process |
| `kill -l` | List all process signals |
| `kill -s SIGSTOP <PID>` | Suspend a process |
| `kill -s SIGCONT <PID>` | Resume a suspended process |
| `id <username>` | Show user and group information |

## Compress and Extract Files

| Command | Description |
|---|---|
| `tar -cvf my.tar mydir/` | Create tar archive |
| `tar -czvf my.tar.gz mydir/` | Create gzip-compressed tar archive |
| `zip -r mydir.zip mydir/` | Create zip archive from directory |
| `tar -xvf my.tar` | Extract tar archive |
| `tar -xzvf my.tar.gz` | Extract gzip-compressed tar archive |
| `gunzip my_file.gz` | Unzip gzip file |
| `unzip my_file.zip` | Unzip zip file |

### Important tar Options

| Option | Meaning |
|---|---|
| `-c` | Create new archive |
| `-x` | Extract archive |
| `-t` | List archive contents |
| `-z` | Filter through gzip |
| `-j` | Filter through bzip2 |
| `-f` | Archive file, must be last option before filename |
| `-v` | Verbose output |
| `--exclude=PATTERN` | Exclude files matching pattern |

## Working With Files

| Command | Description |
|---|---|
| `more <file>` | View text, use space to browse |
| `less <file>` | Flexible text viewer, `G` end, `g` beginning, `/` search |
| `cat <file>` | Print file contents |
| `paste <f1> <f2> > out` | Merge lines from files with tabs |
| `head -<n> <file>` | Print first `<n>` lines |
| `tail -<n> <file>` | Print last `<n>` lines |
| `sort <file>` | Sort lines |
| `uniq <file>` | Remove duplicate adjacent lines, usually after sorting |
| `cut -d , -f 1 <file>` | Print selected CSV field |
| `>` | Redirect stdout to file, overwrite |
| `>>` | Redirect stdout to file, append |
| `<` | Read stdin from file |
| `2>` | Redirect stderr |
| `&>` | Redirect stdout and stderr |
