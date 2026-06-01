# Data Transfer and SLURM Accounting

This page groups Gefion notes about importing/exporting data and generating
SLURM usage reports.

## Import/Export Data With Gefion

Gefion nodes do not have direct Internet access. Data import/export must use
preconfigured transfer methods. The right method depends on file size and
operational constraints.

### Transfer Methods Overview

| Method | Scale | Notes |
|---|---|---|
| Citrix drive mapping | Small files, below 4 GB | Can be used by regular users when convenient |
| FileX Portal | Tens of GB | Browser-based transfer |
| DCAI hosted SFTP | Indefinite file size limit | 850 TB total data size limit |
| External cloud or hosted services | Indefinite file size and total storage limit | Hosted AWS, SFTP, private direct links, etc. |

## DCAI Hosted SFTP

DCAI provides SFTP servers as intermediaries between Gefion nodes and remote
locations. Depending on the total amount of data to transfer, support can create
an account where users upload data from their host institution.

To use this service, provide support with:

- a public SSH key
- the source IP address of the system that will connect to the server

Typical Linux SSH public key location:

```bash
cat ~/.ssh/id_ed25519.pub
```

Find the source IP from the local server:

```bash
curl http://ifconfig.io
```

After support creates the account, connect with the notified port:

```bash
sftp -P 2234 <user>@xfer.dcai.dk
```

or:

```bash
sftp -P 22123 <user>@xfer.dcai.dk
```

## SFTP With rclone

For large transfers or preserving directory trees, DCAI suggests `rclone`.
The example below connects from a local machine to the external DCAI SFTP
server.

### 1. Configure the connection

Run:

```bash
rclone config
```

This creates a `~/.config/rclone/rclone.conf` file.

Example config:

```ini
[dcai_sftp]
type = sftp
host = xfer.dcai.dk
port = 2234
user = username
key_file = ~/.ssh/id_ecdsa
shell_type = cmd
md5sum_command = none
sha1sum_command = none
```

### 2. Transfer data

For large transfers, use `screen` or `tmux`.

```bash
SRC="/home/username/upload_to_gefion"
DST="dcai_sftp:/to_gefion"
LOG="/tmp/dcai_sftp.log"

rclone copy "$SRC" "$DST" \
  --transfers 4 \
  --create-empty-src-dirs \
  --sftp-skip-links \
  --log-file "$LOG" \
  --log-level INFO
```

## SLURM Accounting

### SLURM Usage Report Tool

The `usage.sh` script is a wrapper around the SLURM report generator. It
simplifies report generation and uses efficient data caching.

Load the module first:

```bash
module load usage
```

Show help:

```text
$ usage.sh --help

Options:
  -h, --help                Show this help message
  -u, --user USER           Generate report for the specified user
  -a, --account ACCOUNT     Generate report for the specified account
  -s, --start-date DATE     Start date (YYYY-MM-DD format)
  -e, --end-date DATE       End date (YYYY-MM-DD format)
  -o, --output FILE         Output file (defaults to stdout, .pdf for PDF, .csv for CSV)
  -t, --title TITLE         Custom report title
  -m, --monthly             Generate monthly report (default)
  -y, --yearly              Generate yearly report
  -f, --force-refresh       Force refresh of cached data
  -p, --pdf                 Generate PDF output instead of text
  -c, --csv                 Generate CSV output instead of text
  --prev-month, -pm         Report on the entire previous month
  --all-accounts, -aa       Report on all accounts (no percentage calculations)
```

### Common Usage Patterns

Report for a specific user:

```bash
usage.sh -u jdoe
```

Report for an account:

```bash
usage.sh -a chemistry_dept
```

Specify a date range:

```bash
usage.sh -s 2025-05-01 -e 2025-05-31
```

Generate a yearly PDF report:

```bash
usage.sh -a biology_dept -y -p -o yearly_account_report.pdf
```

Generate an invoice-style CSV report for the previous month:

```bash
usage.sh --prev-month --all-accounts --csv -o invoice.csv
```

### Advanced Usage and Filtering

Filter GPU hours:

```bash
usage.sh | grep "GPU Hours"
```

Extract user info summary:

```bash
usage.sh | grep -A 20 "User Usage Summary"
```

Multiple accounts:

```bash
usage.sh -a physics_dept,chemistry_dept
```

### Utilization Percentage

For the `physics_dept` account, utilization is calculated as:

```text
Utilization % = Node Hours Used / (nodes x Hours in period) x 100
```

The source example used 8 full-time nodes for this calculation.
