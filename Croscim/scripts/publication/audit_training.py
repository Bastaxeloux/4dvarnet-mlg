#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from contrib.SST.evaluation.io import atomic_write_json, write_sha256_sidecar


SCHEDULE_RE = re.compile(
    r"\[TRAIN SCHEDULE\] Epoch (?P<epoch>\d+) x(?P<resolution>\d+): "
    r"batch=(?P<batch>\d+)/GPU, accumulation=(?P<accumulation>\d+), "
    r"batches=(?P<total>\d+), global_samples=(?P<global_samples>\d+), "
    r"optimizer_updates=(?P<optimizer_updates>\d+)"
)
PROGRESS_RE = re.compile(r"Epoch (?P<epoch>\d+):.*?\|\s*(?P<done>\d+)/(?P<total>\d+)")


def parse_log(path: Path, world_size: int) -> list[dict]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    schedules = []
    for match in SCHEDULE_RE.finditer(text):
        schedules.append({
            "position": match.start(),
            "epoch": int(match.group("epoch")),
            "resolution": int(match.group("resolution")),
            "batch_per_gpu": int(match.group("batch")),
            "accumulation": int(match.group("accumulation")),
            "scheduled_batches": int(match.group("total")),
            "scheduled_global_samples": int(match.group("global_samples")),
            "scheduled_optimizer_updates": int(match.group("optimizer_updates")),
        })
    rows = []
    for index, schedule in enumerate(schedules):
        end = schedules[index + 1]["position"] if index + 1 < len(schedules) else len(text)
        section = text[schedule["position"]:end]
        completed = [
            int(match.group("done"))
            for match in PROGRESS_RE.finditer(section)
            if int(match.group("epoch")) == schedule["epoch"]
        ]
        executed_batches = max(completed, default=0)
        is_complete = executed_batches >= schedule["scheduled_batches"]
        if is_complete:
            executed_updates = math.ceil(executed_batches / schedule["accumulation"])
        else:
            executed_updates = executed_batches // schedule["accumulation"]
        rows.append({
            "log": str(path),
            **{key: value for key, value in schedule.items() if key != "position"},
            "executed_batches": executed_batches,
            "executed_global_samples": executed_batches * schedule["batch_per_gpu"] * world_size,
            "executed_optimizer_updates_lower_bound": executed_updates,
            "complete_epoch": is_complete,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit effective Croscim training work from SLURM logs")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = sorted({Path(path) for pattern in args.logs for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("No training logs matched")
    rows = [row for path in paths for row in parse_log(path, args.world_size)]
    if not rows:
        raise RuntimeError("No TRAIN SCHEDULE entries found in logs")

    totals = defaultdict(lambda: {"batches": 0, "samples": 0, "updates_lower_bound": 0, "partial_epochs": 0})
    for row in rows:
        total = totals[f"x{row['resolution']}"]
        total["batches"] += row["executed_batches"]
        total["samples"] += row["executed_global_samples"]
        total["updates_lower_bound"] += row["executed_optimizer_updates_lower_bound"]
        total["partial_epochs"] += int(not row["complete_epoch"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_execution_audit.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "training_execution_audit.json"
    atomic_write_json(report_path, {
        "schema_version": 1,
        "world_size": args.world_size,
        "logs": [str(path) for path in paths],
        "totals_by_resolution": dict(totals),
        "caveat": "Interrupted-epoch optimizer updates are conservative lower bounds inferred from progress logs.",
        "executions": rows,
    })
    write_sha256_sidecar(csv_path)
    write_sha256_sidecar(report_path)
    print(json.dumps(dict(totals), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
