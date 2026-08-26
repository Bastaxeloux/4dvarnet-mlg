#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import zarr

from contrib.SST.evaluation.coast import build_coastal_mask, save_coastal_mask
from contrib.SST.evaluation.io import atomic_write_json
from contrib.SST.evaluation.masking import find_daily_store
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the appendix-B evaluation dates and masks")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dmi-oi-units", choices=("kelvin", "celsius"), required=True)
    parser.add_argument("--raw-netcdf", help="One original DMI-OI NetCDF for exact Zarr comparison")
    parser.add_argument("--raw-date", help="YYYY-MM-DD date of --raw-netcdf when absent from its filename")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = output_dir / "manifests"
    paths = build_publication_manifests(manifest_dir)
    pilot = load_manifest(paths["pilot"])
    final = load_manifest(paths["test"])
    pilot_records = [EvaluationRecord(**record) for record in pilot["records"]]
    final_records = [EvaluationRecord(**record) for record in final["records"]]

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
    else:
        oi_report["raw_archive_verified"] = False
        oi_report["raw_archive_verification_status"] = "pending_original_netcdf"
    oi_path = output_dir / "dmi_oi_verification.json"
    atomic_write_json(oi_path, oi_report)

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
    print(f"pilot_manifest={paths['pilot']}")
    print(f"test_manifest={paths['test']}")
    print(f"coastal_mask={coastal_path}")
    print(f"dmi_oi={oi_path}")


if __name__ == "__main__":
    main()
