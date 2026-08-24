#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import zarr

from contrib.SST.evaluation.coast import build_coastal_mask, save_coastal_mask
from contrib.SST.evaluation.io import atomic_write_json, sha256_file, write_sha256_sidecar
from contrib.SST.evaluation.masking import date_sequence, find_daily_store
from contrib.SST.evaluation.oi import (
    compare_zarr_with_original_netcdf,
    verification_payload,
    verify_dmi_oi,
)
from contrib.SST.evaluation.protocol import (
    EvaluationRecord,
    build_publication_manifests,
    load_manifest,
)


def verify_record_files(data_root: Path, records: list[EvaluationRecord]) -> dict:
    target_dates = set()
    donor_dates = set()
    for record in records:
        target_dates.update(date_sequence(record.context_start, record.context_end))
        donor_dates.update(date_sequence(record.donor_context_start, record.donor_context_end))
    checked = {
        find_daily_store(data_root, date, resolution)
        for date in sorted(target_dates)
        for resolution in ("x1", "x3", "x10")
    }
    checked.update(find_daily_store(data_root, date, "x1") for date in sorted(donor_dates))
    return {"n_unique_stores": len(checked), "all_present": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate the frozen appendix-B protocol")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--dmi-oi-units", choices=("kelvin", "celsius"), required=True)
    parser.add_argument("--raw-netcdf", help="One original DMI-OI NetCDF for exact Zarr comparison")
    parser.add_argument("--raw-date", help="YYYY-MM-DD date of --raw-netcdf when absent from its filename")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    normalization_path = Path(args.normalization).resolve()
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Normalization file not found: {normalization_path}")
    manifest_dir = output_dir / "manifests"
    paths = build_publication_manifests(manifest_dir)
    pilot = load_manifest(paths["pilot"])
    final = load_manifest(paths["test"])
    pilot_records = [EvaluationRecord(**record) for record in pilot["records"]]
    final_records = [EvaluationRecord(**record) for record in final["records"]]

    availability = {
        "pilot": verify_record_files(data_root, pilot_records),
        "test": verify_record_files(data_root, final_records),
    }
    sample_dates = [dt.date.fromisoformat(record.central_date) for record in pilot_records]
    oi_report = verification_payload(
        verify_dmi_oi(data_root, sample_dates, units=args.dmi_oi_units)
    )
    for check in ("time_matches_filename", "grid_matches_x1", "excluded_from_model_inputs"):
        if not oi_report[check]:
            raise RuntimeError(f"DMI-OI preparation check failed: {check}")
    if args.raw_netcdf:
        raw_path = Path(args.raw_netcdf).resolve()
        if args.raw_date:
            raw_date = dt.date.fromisoformat(args.raw_date)
        else:
            match = re.search(r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})", raw_path.name)
            if match is None:
                raise RuntimeError("Cannot infer --raw-netcdf date; pass --raw-date YYYY-MM-DD")
            raw_date = dt.date(
                int(match.group("year")), int(match.group("month")), int(match.group("day"))
            )
        oi_report.update(
            compare_zarr_with_original_netcdf(
                find_daily_store(data_root, raw_date, "x1"),
                raw_path,
                expected_units=args.dmi_oi_units,
                expected_date=raw_date,
            )
        )
        oi_report["original_netcdf_sha256"] = sha256_file(raw_path)
    else:
        oi_report["raw_archive_verified"] = False
        oi_report["raw_archive_verification_status"] = "pending_original_netcdf"
    oi_path = output_dir / "dmi_oi_verification.json"
    atomic_write_json(oi_path, oi_report)
    write_sha256_sidecar(oi_path)

    first_date = dt.date.fromisoformat(final_records[0].central_date)
    store = zarr.open(str(find_daily_store(data_root, first_date, "x1")), mode="r")
    latitude = np.asarray(store["lat"][:])
    longitude = np.asarray(store["lon"][:])
    if not (np.all(np.diff(latitude) > 0) or np.all(np.diff(latitude) < 0)):
        raise RuntimeError("x1 latitude is not strictly monotonic")
    if not (np.all(np.diff(longitude) > 0) or np.all(np.diff(longitude) < 0)):
        raise RuntimeError("x1 longitude is not strictly monotonic")
    coastal_path = output_dir / "static" / "coastal_mask_50km.npz"
    if coastal_path.exists():
        with np.load(coastal_path) as payload:
            if payload["coastal_mask"].shape != store["surfmask"].shape:
                raise RuntimeError("Existing coastal mask has the wrong grid shape")
            if float(payload["threshold_km"]) != 50.0:
                raise RuntimeError("Existing coastal mask has the wrong distance threshold")
    else:
        coastal_mask = build_coastal_mask(
            latitude, longitude, np.asarray(store["surfmask"][:]), threshold_km=50.0
        )
        save_coastal_mask(coastal_path, coastal_mask, threshold_km=50.0)
    write_sha256_sidecar(coastal_path)

    report = {
        "schema_version": 1,
        "data_root": str(data_root),
        "availability": availability,
        "manifests": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "dmi_oi_verification": {"path": str(oi_path), "sha256": sha256_file(oi_path)},
        "coastal_mask": {"path": str(coastal_path), "sha256": sha256_file(coastal_path)},
        "normalization": {
            "path": str(normalization_path),
            "sha256": sha256_file(normalization_path),
        },
        "status": "prepared_not_frozen_until_pilot_acceptance",
    }
    report_path = output_dir / "protocol_preparation.json"
    atomic_write_json(report_path, report)
    write_sha256_sidecar(report_path)
    print(f"protocol={report_path}")
    print(f"pilot_manifest={paths['pilot']}")
    print(f"test_manifest={paths['test']}")
    print(f"coastal_mask={coastal_path}")
    print(f"dmi_oi={oi_path}")


if __name__ == "__main__":
    main()
