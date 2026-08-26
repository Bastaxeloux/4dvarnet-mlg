#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contrib.SST.evaluation.io import atomic_write_json, sha256_file, write_sha256_sidecar
from contrib.SST.evaluation.protocol import load_manifest


SELECTION_CRITERION = "val/x1/loss"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the appendix-B protocol after pilot acceptance")
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--pilot-evaluation-root", required=True)
    parser.add_argument("--pilot-validation", required=True)
    parser.add_argument("--spatial-diagnostic", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preparation = json.loads(Path(args.preparation).read_text())
    oi_path = Path(preparation["dmi_oi_verification"]["path"])
    oi_report = json.loads(oi_path.read_text())
    if not oi_report.get("raw_archive_verified", False):
        raise RuntimeError("DMI-OI original-NetCDF verification is still pending")
    oi_requirements = {
        "time_matches_filename",
        "grid_matches_x1",
        "excluded_from_model_inputs",
        "valid_date_matches_archive",
        "valid_time_matches_archive",
        "values_exact_after_float32_conversion",
        "grid_exact",
    }
    failed_oi_checks = sorted(key for key in oi_requirements if not oi_report.get(key, False))
    if failed_oi_checks:
        raise RuntimeError(f"DMI-OI verification failed: {failed_oi_checks}")
    pilot_manifest = load_manifest(args.pilot_manifest)
    if len(pilot_manifest["records"]) != 24:
        raise RuntimeError("Pilot manifest must contain exactly 24 dates")
    validation = json.loads(Path(args.pilot_validation).read_text())
    if not validation.get("accepted") or validation.get("n_dates") != 24:
        raise RuntimeError("The complete 24-date pilot has not passed export validation")
    if validation.get("mode") != "controlled" or validation.get("frozen_protocol_sha256") is not None:
        raise RuntimeError("Protocol freeze requires the unfrozen controlled 2023 pilot")
    checkpoint = json.loads(Path(args.checkpoint_manifest).read_text())
    if (
        checkpoint.get("selection") != "explicit_checkpoint_path"
        or checkpoint.get("selection_criterion") != SELECTION_CRITERION
        or (int(checkpoint["epoch"]) + 1) % 24
    ):
        raise RuntimeError("Checkpoint was not selected by val/x1/loss at a cycle boundary")
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise RuntimeError("Frozen checkpoint path or SHA-256 does not match its manifest")
    if validation.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise RuntimeError("Pilot validation and checkpoint manifest use different checkpoints")
    if validation.get("manifest_sha256") != sha256_file(args.pilot_manifest):
        raise RuntimeError("Pilot validation and pilot manifest do not match")
    spatial_diagnostic = json.loads(Path(args.spatial_diagnostic).read_text())
    if spatial_diagnostic.get("pilot_dates") != 24 or spatial_diagnostic.get("diagnostic") not in {
        "radially_averaged_psd_and_coherence",
        "second_order_structure_function",
    }:
        raise RuntimeError("Invalid 2023 spatial diagnostic")

    for artifact_name in ("dmi_oi_verification", "coastal_mask", "normalization"):
        artifact = preparation[artifact_name]
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Prepared {artifact_name} changed before protocol freeze")
    for name, manifest in preparation["manifests"].items():
        if sha256_file(manifest["path"]) != manifest["sha256"]:
            raise RuntimeError(f"Prepared manifest changed before protocol freeze: {name}")

    evaluation_root = Path(args.pilot_evaluation_root)
    done_files = sorted((evaluation_root / "done" / "controlled").glob("*.done.json"))
    if len(done_files) != 24:
        raise RuntimeError(f"Expected 24 controlled pilot markers, found {len(done_files)}")
    runtime_files = sorted((evaluation_root / "provenance").glob("runtime_controlled_rank_*.json"))
    if not runtime_files:
        raise RuntimeError("Pilot runtime provenance is missing")
    runtime_payloads = [json.loads(path.read_text()) for path in runtime_files]
    expected_normalization_hash = preparation["normalization"]["sha256"]
    expected_manifest_hash = sha256_file(args.pilot_manifest)
    for payload in runtime_payloads:
        if payload["checkpoint"]["sha256"] != checkpoint["sha256"]:
            raise RuntimeError("Pilot runtime provenance uses a different checkpoint")
        if payload["manifest"]["sha256"] != expected_manifest_hash:
            raise RuntimeError("Pilot runtime provenance uses a different manifest")
        for artifact_name in ("resolved_config", "normalization"):
            artifact = payload[artifact_name]
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError(f"Pilot runtime artifact changed: {artifact_name}")
    if any(
        payload["normalization"]["sha256"] != expected_normalization_hash
        for payload in runtime_payloads
    ):
        raise RuntimeError("Pilot used normalization statistics different from the prepared protocol")
    config_hashes = {payload["resolved_config"]["sha256"] for payload in runtime_payloads}
    if len(config_hashes) != 1:
        raise RuntimeError(f"Pilot workers used different resolved configurations: {config_hashes}")
    project_root = Path(args.project_root)
    source_paths = [
        project_root / "contrib/SST/data.py",
        project_root / "contrib/SST/load_data.py",
        project_root / "contrib/SST/models.py",
        project_root / "contrib/SST/solver.py",
        project_root / "contrib/SST/statistics_io.py",
        project_root / "contrib/SST/model_components/grad_mods/convlstm.py",
        project_root / "contrib/SST/model_components/priors/resunet.py",
        project_root / "config/main.yaml",
        project_root / "config/xp/SST/multires_jeanzay_resunet.yaml",
        project_root / "contrib/SST/evaluation/assembly.py",
        project_root / "contrib/SST/evaluation/coast.py",
        project_root / "contrib/SST/evaluation/evaluator.py",
        project_root / "contrib/SST/evaluation/io.py",
        project_root / "contrib/SST/evaluation/masking.py",
        project_root / "contrib/SST/evaluation/metrics.py",
        project_root / "contrib/SST/evaluation/oi.py",
        project_root / "contrib/SST/evaluation/protocol.py",
        project_root / "scripts/publication/aggregate_metrics.py",
        project_root / "scripts/publication/select_qualitative_cases.py",
        project_root / "scripts/publication/spatial_diagnostics.py",
        project_root / "scripts/publication/validate_exports.py",
    ]
    payload = {
        "schema_version": 1,
        "status": "frozen_after_2023_pilot_before_2024_test",
        "checkpoint": checkpoint,
        "artifacts": {
            "preparation": sha256_file(args.preparation),
            "pilot_manifest": sha256_file(args.pilot_manifest),
            "pilot_validation": sha256_file(args.pilot_validation),
            "spatial_diagnostic": sha256_file(args.spatial_diagnostic),
            "dmi_oi_verification": sha256_file(oi_path),
        },
        "manifests": preparation["manifests"],
        "static_artifacts": {
            "coastal_mask": preparation["coastal_mask"],
            "dmi_oi_verification": preparation["dmi_oi_verification"],
            "normalization": preparation["normalization"],
        },
        "runtime_contract": {
            "resolved_config_sha256": next(iter(config_hashes)),
            "normalization_sha256": expected_normalization_hash,
            "provenance_files": {path.name: sha256_file(path) for path in runtime_files},
        },
        "pilot_done_markers": {path.name: sha256_file(path) for path in done_files},
        "evaluation_sources": {str(path.relative_to(project_root)): sha256_file(path) for path in source_paths},
    }
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Frozen protocol already exists: {output}")
    atomic_write_json(output, payload)
    write_sha256_sidecar(output)
    os.chmod(output, 0o440)
    print(f"frozen_protocol={output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
