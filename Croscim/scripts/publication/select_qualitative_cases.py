#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr

from contrib.SST.evaluation.coast import load_coastal_mask
from contrib.SST.evaluation.io import atomic_write_json
from contrib.SST.evaluation.protocol import load_manifest


SLOTS = (
    ("open_ocean", "moderate", 0.10, 0.35),
    ("open_ocean", "high", 0.35, 0.70),
    ("coastal", "moderate", 0.10, 0.35),
    ("coastal", "high", 0.35, 0.70),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a patch catalog and deterministic review gallery")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--coastal-mask", required=True)
    parser.add_argument("--mode", default="controlled")
    parser.add_argument("--output", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--n-per-slot", type=int, default=1)
    args = parser.parse_args()
    if args.n_per_slot <= 0:
        raise ValueError("--n-per-slot must be positive")
    manifest = load_manifest(args.manifest)
    records = manifest["records"]
    sample_positions = np.rint(np.linspace(0, len(records) - 1, min(24, len(records)))).astype(int)
    sampled_records = [records[index] for index in np.unique(sample_positions)]
    coastal = load_coastal_mask(args.coastal_mask)
    candidates = []
    starts_y = np.rint(np.linspace(0, 3600 - 256, 15)).astype(int)
    starts_x = np.rint(np.linspace(0, 7200 - 256, 30)).astype(int)
    for record_index, record in enumerate(sampled_records, start=1):
        date = record["central_date"]
        print(f"[PATCH CATALOG] {record_index}/{len(sampled_records)} {date}", flush=True)
        path = Path(args.evaluation_root) / "daily" / args.mode / f"{date}.nc"
        with xr.open_dataset(path) as dataset:
            target = dataset["target_sst"].values
            predictions = {
                name: dataset[f"pred_sst_{name}"].values
                for name in ("x10", "x3", "x1")
            }
            hidden = dataset["hidden_mask"].values.astype(bool)
            surfmask = dataset["surfmask"].values
            latitude = dataset["lat"].values
            longitude = dataset["lon"].values
        for y0 in starts_y:
            for x0 in starts_x:
                selection = np.s_[y0:y0 + 256, x0:x0 + 256]
                ocean = surfmask[selection] != 0
                finite_target = ocean & np.isfinite(target[selection])
                if ocean.mean() < 0.50 or finite_target.sum() < 5000:
                    continue
                hidden_patch = hidden[selection] & finite_target
                missingness = hidden_patch.sum() / finite_target.sum()
                if hidden_patch.sum() < 1000:
                    continue
                coast_fraction = (coastal[selection] & ocean).sum() / ocean.sum()
                category = "coastal" if coast_fraction >= 0.05 else "open_ocean" if coast_fraction <= 0.01 else None
                if category is None:
                    continue
                errors = {
                    name: prediction[selection][hidden_patch] - target[selection][hidden_patch]
                    for name, prediction in predictions.items()
                }
                patch_id = f"{date}_y{int(y0):04d}_x{int(x0):04d}"
                candidates.append({
                    "patch_id": patch_id,
                    "date": date,
                    "lat_start": int(y0),
                    "lon_start": int(x0),
                    "latitude_center": float(latitude[y0 + 128]),
                    "longitude_center": float(longitude[x0 + 128]),
                    "category": category,
                    "missingness": float(missingness),
                    "coastal_fraction": float(coast_fraction),
                    "ocean_fraction": float(ocean.mean()),
                    "hidden_pixels": int(hidden_patch.sum()),
                    "hidden_rmse_x10_c": float(np.sqrt(np.mean(errors["x10"] ** 2))),
                    "hidden_rmse_x3_c": float(np.sqrt(np.mean(errors["x3"] ** 2))),
                    "hidden_rmse_x1_c": float(np.sqrt(np.mean(errors["x1"] ** 2))),
                    "hidden_mae_x1_c": float(np.mean(np.abs(errors["x1"]))),
                    "hidden_bias_x1_c": float(np.mean(errors["x1"])),
                })

    selected = []
    for category, bin_name, lower, upper in SLOTS:
        slot = [
            candidate for candidate in candidates
            if candidate["category"] == category and lower <= candidate["missingness"] < upper
        ]
        if not slot:
            raise RuntimeError(f"No qualitative candidate for {category}/{bin_name} missingness [{lower}, {upper})")
        ordered = sorted(
            slot,
            key=lambda candidate: (
                candidate["hidden_rmse_x1_c"], candidate["date"],
                candidate["lat_start"], candidate["lon_start"],
            ),
        )
        median = float(np.median([candidate["hidden_rmse_x1_c"] for candidate in slot]))
        n_best = min((args.n_per_slot + 1) // 2, len(ordered))
        choices = [({**candidate}, "best") for candidate in ordered[:n_best]]
        used = {candidate["patch_id"] for candidate, _ in choices}
        median_order = sorted(
            (candidate for candidate in slot if candidate["patch_id"] not in used),
            key=lambda candidate: (
                abs(candidate["hidden_rmse_x1_c"] - median), candidate["date"],
                candidate["lat_start"], candidate["lon_start"],
            ),
        )
        choices.extend(
            ({**candidate}, "median")
            for candidate in median_order[:args.n_per_slot - len(choices)]
        )
        for choice, gallery_group in choices:
            selected.append({
                **choice,
                "missingness_class": bin_name,
                "gallery_group": gallery_group,
                "class_median_rmse_x1_c": median,
            })

    if args.catalog:
        catalog_path = Path(args.catalog)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with catalog_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
            writer.writeheader()
            writer.writerows(candidates)
    payload = {
        "schema_version": 1,
        "selection_rule": "half_lowest_rmse_half_closest_to_class_median",
        "candidate_dates": [record["central_date"] for record in sampled_records],
        "slots": [
            {"category": category, "name": name, "lower": lower, "upper": upper}
            for category, name, lower, upper in SLOTS
        ],
        "cases": selected,
    }
    output = Path(args.output)
    atomic_write_json(output, payload)
    print(f"catalog_candidates={len(candidates)}")
    print(f"gallery_cases={len(selected)}")
    print(f"gallery_manifest={output}")


if __name__ == "__main__":
    main()
