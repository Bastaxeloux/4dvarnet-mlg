#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from contrib.SST.evaluation.io import atomic_write_json, sha256_file, write_sha256_sidecar


SELECTION_MONITOR = "controlled_2023/hidden/global/croscim_x1/rmse_c"


def load_checkpoint_metadata(path: Path) -> tuple[int, int]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Invalid Lightning checkpoint: {path}")
    return int(payload.get("epoch", -1)), int(payload.get("global_step", -1))


def _pilot_rmse(path: Path) -> float:
    with path.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["period_type"] == "annual"
            and row["period"] == "2023"
            and row["method"] == "croscim_x1"
            and row["support"] == "hidden"
            and row["regime"] == "global"
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one controlled 2023 hidden x1 row in {path}, found {len(rows)}")
    return float(rows[0]["rmse_c"])


def read_candidate(checkpoint: str | Path, evaluation_root: str | Path) -> dict:
    checkpoint = Path(checkpoint).expanduser().resolve()
    evaluation_root = Path(evaluation_root).expanduser().resolve()
    validation_path = evaluation_root / "pilot_validation.json"
    results_dir = evaluation_root / "results"
    aggregation_path = results_dir / "aggregation_complete.json"
    metrics_path = results_dir / "metrics_summary.csv"
    for path in (checkpoint, validation_path, aggregation_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Candidate artifact is missing: {path}")

    checkpoint_hash = sha256_file(checkpoint)
    validation = json.loads(validation_path.read_text())
    aggregation = json.loads(aggregation_path.read_text())
    if not validation.get("accepted") or validation.get("mode") != "controlled":
        raise RuntimeError(f"Pilot validation was not accepted: {validation_path}")
    if validation.get("n_dates") != 24 or validation.get("frozen_protocol_sha256") is not None:
        raise RuntimeError(f"Candidate must be an unfrozen 24-date pilot: {validation_path}")
    if aggregation.get("mode") != "controlled" or aggregation.get("n_dates") != 24:
        raise RuntimeError(f"Candidate aggregation is not a controlled 24-date pilot: {aggregation_path}")
    if validation.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"Validation checkpoint does not match {checkpoint}")
    if aggregation.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"Aggregation checkpoint does not match {checkpoint}")
    if aggregation.get("manifest", {}).get("sha256") != validation.get("manifest_sha256"):
        raise RuntimeError(f"Candidate manifest hashes disagree under {evaluation_root}")
    if aggregation.get("artifacts", {}).get(metrics_path.name) != sha256_file(metrics_path):
        raise RuntimeError(f"Metrics summary changed after aggregation: {metrics_path}")

    epoch, global_step = load_checkpoint_metadata(checkpoint)
    if epoch < 0 or (epoch + 1) % 24:
        raise RuntimeError(f"Checkpoint epoch {epoch} is not a complete 24-epoch cycle boundary")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "epoch": epoch,
        "global_step": global_step,
        "evaluation_root": str(evaluation_root),
        "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        "aggregation": {"path": str(aggregation_path), "sha256": sha256_file(aggregation_path)},
        "metrics_summary": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "pilot_manifest_sha256": validation["manifest_sha256"],
        "hidden_global_x1_rmse_c": _pilot_rmse(metrics_path),
    }


def select_candidate_records(candidate_pairs: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    if not candidate_pairs:
        raise ValueError("At least one candidate is required")
    candidates = [read_candidate(checkpoint, evaluation_root) for checkpoint, evaluation_root in candidate_pairs]
    manifest_hashes = {candidate["pilot_manifest_sha256"] for candidate in candidates}
    if len(manifest_hashes) != 1:
        raise RuntimeError(f"Candidates used different pilot manifests: {manifest_hashes}")
    selected = min(candidates, key=lambda item: (item["hidden_global_x1_rmse_c"], item["epoch"]))
    return candidates, selected


def snapshot_selected(candidate: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "publication_best.ckpt"
    report_path = output_dir / "selection_report.json"
    manifest_path = output_dir / "publication_best.json"
    if any(path.exists() for path in (snapshot, report_path, manifest_path)):
        raise RuntimeError(f"Refusing to replace an existing frozen selection under {output_dir}")

    temporary = snapshot.with_name(f".{snapshot.name}.tmp-{os.getpid()}")
    shutil.copy2(candidate["checkpoint"], temporary)
    if sha256_file(temporary) != candidate["checkpoint_sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Checkpoint snapshot hash mismatch")
    os.replace(temporary, snapshot)
    snapshot.chmod(0o440)
    write_sha256_sidecar(snapshot)
    return snapshot, report_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a complete-cycle checkpoint on the controlled 2023 pilot")
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("CHECKPOINT", "EVALUATION_ROOT"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    candidates, selected = select_candidate_records([tuple(pair) for pair in args.candidate])
    output_dir = Path(args.output_dir).expanduser().resolve()
    snapshot, report_path, manifest_path = snapshot_selected(selected, output_dir)
    report = {
        "schema_version": 1,
        "selection_monitor": SELECTION_MONITOR,
        "selection_mode": "min",
        "pilot_manifest_sha256": selected["pilot_manifest_sha256"],
        "candidates": candidates,
        "selected": selected,
    }
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    manifest = {
        "schema_version": 1,
        "path": str(snapshot),
        "sha256": sha256_file(snapshot),
        "epoch": selected["epoch"],
        "global_step": selected["global_step"],
        "score": selected["hidden_global_x1_rmse_c"],
        "monitor": SELECTION_MONITOR,
        "source_checkpoint": selected["checkpoint"],
        "source_evaluation_root": selected["evaluation_root"],
        "pilot_manifest_sha256": selected["pilot_manifest_sha256"],
        "selection_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
    }
    atomic_write_json(manifest_path, manifest)
    write_sha256_sidecar(manifest_path)
    report_path.chmod(0o440)
    manifest_path.chmod(0o440)
    print(f"checkpoint={snapshot}")
    print(f"epoch={selected['epoch']}")
    print(f"controlled_hidden_x1_rmse_c={selected['hidden_global_x1_rmse_c']:.8f}")
    print(f"selection_report={report_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
