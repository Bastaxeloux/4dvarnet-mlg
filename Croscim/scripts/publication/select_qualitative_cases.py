#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from contrib.SST.evaluation.coast import load_coastal_mask
from contrib.SST.evaluation.io import atomic_write_json, write_sha256_sidecar
from contrib.SST.evaluation.protocol import load_manifest


SLOTS = (
    ("open_ocean", "moderate", 0.10, 0.35),
    ("open_ocean", "high", 0.35, 0.70),
    ("coastal", "moderate", 0.10, 0.35),
    ("coastal", "high", 0.35, 0.70),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select four deterministic qualitative cases")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--coastal-mask", required=True)
    parser.add_argument("--mode", default="controlled")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    records = manifest["records"]
    sample_positions = np.rint(np.linspace(0, len(records) - 1, min(24, len(records)))).astype(int)
    sampled_records = [records[index] for index in np.unique(sample_positions)]
    coastal = load_coastal_mask(args.coastal_mask)
    candidates = []
    starts_y = np.rint(np.linspace(0, 3600 - 256, 15)).astype(int)
    starts_x = np.rint(np.linspace(0, 7200 - 256, 30)).astype(int)
    for record in sampled_records:
        date = record["central_date"]
        path = Path(args.evaluation_root) / "daily" / args.mode / f"{date}.nc"
        with xr.open_dataset(path) as dataset:
            target = dataset["target_sst"].values
            prediction = dataset["pred_sst_x1"].values
            hidden = dataset["hidden_mask"].values.astype(bool)
            surfmask = dataset["surfmask"].values
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
                error = prediction[selection][hidden_patch] - target[selection][hidden_patch]
                candidates.append({
                    "date": date,
                    "lat_start": int(y0),
                    "lon_start": int(x0),
                    "category": category,
                    "missingness": float(missingness),
                    "coastal_fraction": float(coast_fraction),
                    "hidden_rmse_c": float(np.sqrt(np.mean(error * error))),
                })

    selected = []
    for category, bin_name, lower, upper in SLOTS:
        slot = [
            candidate for candidate in candidates
            if candidate["category"] == category and lower <= candidate["missingness"] < upper
        ]
        if not slot:
            raise RuntimeError(f"No qualitative candidate for {category}/{bin_name} missingness [{lower}, {upper})")
        median = float(np.median([candidate["hidden_rmse_c"] for candidate in slot]))
        choice = min(
            slot,
            key=lambda candidate: (
                abs(candidate["hidden_rmse_c"] - median),
                candidate["date"],
                candidate["lat_start"],
                candidate["lon_start"],
            ),
        )
        selected.append({**choice, "missingness_class": bin_name, "class_median_rmse_c": median})
    payload = {
        "schema_version": 1,
        "selection_rule": "closest_to_class_median_hidden_rmse",
        "candidate_dates": [record["central_date"] for record in sampled_records],
        "slots": [
            {"category": category, "name": name, "lower": lower, "upper": upper}
            for category, name, lower, upper in SLOTS
        ],
        "cases": selected,
    }
    output = Path(args.output)
    atomic_write_json(output, payload)
    write_sha256_sidecar(output)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
