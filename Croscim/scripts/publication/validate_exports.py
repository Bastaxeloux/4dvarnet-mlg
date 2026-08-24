#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from contrib.SST.evaluation.coast import load_coastal_mask
from contrib.SST.evaluation.io import atomic_write_json, sha256_file, write_sha256_sidecar
from contrib.SST.evaluation.metrics import regime_masks, weighted_sufficient_stats
from contrib.SST.evaluation.protocol import load_manifest


REQUIRED_VARIABLES = {
    "target_sst", "observed_sst", "pred_sst_x10", "pred_sst_x3", "pred_sst_x1",
    "hidden_mask", "visible_mask", "original_valid_mask", "surfmask", "sea_ice_fraction", "dmi_oi",
}


def compare_stats(expected: dict, actual: dict, *, atol: float = 1e-4, rtol: float = 1e-7) -> None:
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if isinstance(expected_value, int):
            if int(actual_value) != expected_value:
                raise RuntimeError(f"Integer metric mismatch for {key}: {actual_value} != {expected_value}")
        elif not np.isclose(float(actual_value), float(expected_value), atol=atol, rtol=rtol, equal_nan=True):
            raise RuntimeError(f"Metric mismatch for {key}: {actual_value} != {expected_value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported daily maps against sufficient statistics")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--mode", choices=("controlled", "rectangles"), required=True)
    parser.add_argument("--coastal-mask", required=True)
    parser.add_argument("--frozen-protocol")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dates = [record["central_date"] for record in manifest["records"]]
    evaluation_root = Path(args.evaluation_root)
    coastal = load_coastal_mask(args.coastal_mask)
    protocol_hash = sha256_file(args.frozen_protocol) if args.frozen_protocol else None
    expected_dates = set(dates)
    exported_dates = {
        "done": {
            path.name.removesuffix(".done.json")
            for path in (evaluation_root / "done" / args.mode).glob("*.done.json")
        },
        "metrics": {
            path.name.removesuffix(".json")
            for path in (evaluation_root / "metrics_daily" / args.mode).glob("*.json")
        },
        "netcdf": {
            path.name.removesuffix(".nc")
            for path in (evaluation_root / "daily" / args.mode).glob("*.nc")
        },
    }
    for artifact, actual_dates in exported_dates.items():
        if actual_dates != expected_dates:
            raise RuntimeError(
                f"{artifact} dates do not match the manifest: "
                f"missing={sorted(expected_dates - actual_dates)[:10]}, "
                f"extra={sorted(actual_dates - expected_dates)[:10]}"
            )
    checkpoint_hashes = set()
    verified_rows = 0
    for date in dates:
        done_path = evaluation_root / "done" / args.mode / f"{date}.done.json"
        metrics_path = evaluation_root / "metrics_daily" / args.mode / f"{date}.json"
        netcdf_path = evaluation_root / "daily" / args.mode / f"{date}.nc"
        if not done_path.exists() or not metrics_path.exists() or not netcdf_path.exists():
            raise RuntimeError(f"Incomplete export for {date}")
        done = json.loads(done_path.read_text())
        if sha256_file(netcdf_path) != done["netcdf"]["sha256"]:
            raise RuntimeError(f"NetCDF hash mismatch for {date}")
        if sha256_file(metrics_path) != done["metrics"]["sha256"]:
            raise RuntimeError(f"Metrics hash mismatch for {date}")
        checkpoint_hashes.add(done["checkpoint_sha256"])
        if done.get("frozen_protocol_sha256") != protocol_hash:
            raise RuntimeError(
                f"Frozen protocol mismatch for {date}: "
                f"{done.get('frozen_protocol_sha256')} != {protocol_hash}"
            )

        metrics = json.loads(metrics_path.read_text())
        if metrics.get("mode") != args.mode or metrics.get("record", {}).get("central_date") != date:
            raise RuntimeError(f"Metrics identity mismatch for {date}")
        for resolution in (10, 3, 1):
            uncovered = metrics["assembly"][f"x{resolution}"]["n_uncovered_pixels"]
            if int(uncovered) != 0:
                raise RuntimeError(f"Uncovered x{resolution} pixels for {date}: {uncovered}")
        sufficient_rows = metrics["sufficient_statistics"]
        expected_row_count = 4 * 2 * 4
        expected_rows = {
            (row["method"], row["support"], row["regime"]): row
            for row in sufficient_rows
        }
        if len(sufficient_rows) != expected_row_count or len(expected_rows) != expected_row_count:
            raise RuntimeError(
                f"Expected {expected_row_count} unique sufficient-statistic rows for {date}, "
                f"found {len(sufficient_rows)} rows and {len(expected_rows)} unique keys"
            )
        with xr.open_dataset(netcdf_path) as dataset:
            missing_vars = REQUIRED_VARIABLES - set(dataset.data_vars)
            if missing_vars:
                raise RuntimeError(f"Missing NetCDF variables for {date}: {missing_vars}")
            if int((dataset["patch_coverage_x1"] == 0).sum()) != 0:
                raise RuntimeError(f"Uncovered x1 pixels for {date}")
            if dataset.attrs.get("frozen_protocol_sha256") != (protocol_hash or "none"):
                raise RuntimeError(f"NetCDF frozen protocol mismatch for {date}")
            if (
                dataset.attrs.get("central_date") != date
                or dataset.attrs.get("evaluation_mode") != args.mode
            ):
                raise RuntimeError(f"NetCDF identity mismatch for {date}")
            target = dataset["target_sst"].values
            hidden = dataset["hidden_mask"].values.astype(bool)
            visible = dataset["visible_mask"].values.astype(bool)
            surfmask = dataset["surfmask"].values
            sea_ice = dataset["sea_ice_fraction"].values
            latitude = dataset["lat"].values
            if np.any(hidden & visible):
                raise RuntimeError(f"Hidden and visible supports overlap for {date}")
            if not np.array_equal(dataset["original_valid_mask"].values.astype(bool), hidden | visible):
                raise RuntimeError(f"original_valid_mask mismatch for {date}")
            if np.any(hidden & (~np.isfinite(target) | (surfmask == 0))):
                raise RuntimeError(f"Invalid hidden support for {date}")
            regimes = regime_masks(surfmask, sea_ice, coastal)
            methods = {
                "croscim_x10": dataset["pred_sst_x10"].values,
                "croscim_x3": dataset["pred_sst_x3"].values,
                "croscim_x1": dataset["pred_sst_x1"].values,
                "dmi_oi": dataset["dmi_oi"].values,
            }
            evaluation_support = hidden | visible
            for method, prediction in methods.items():
                n_nonfinite = int(np.sum(evaluation_support & ~np.isfinite(prediction)))
                if n_nonfinite:
                    raise RuntimeError(
                        f"{method} has {n_nonfinite} non-finite values on the evaluation support for {date}"
                    )
            for method, prediction in methods.items():
                for support_name, support in (("hidden", hidden), ("visible", visible)):
                    for regime_name, regime in regimes.items():
                        actual = weighted_sufficient_stats(target, prediction, support & regime, latitude).as_dict()
                        compare_stats(expected_rows[(method, support_name, regime_name)], actual)
                        verified_rows += 1
    if len(checkpoint_hashes) != 1:
        raise RuntimeError(f"Evaluation used multiple checkpoints: {checkpoint_hashes}")
    report = {
        "schema_version": 1,
        "accepted": True,
        "n_dates": len(dates),
        "n_verified_sufficient_stat_rows": verified_rows,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "frozen_protocol_sha256": protocol_hash,
        "manifest_sha256": sha256_file(args.manifest),
        "mode": args.mode,
    }
    output = Path(args.output)
    atomic_write_json(output, report)
    write_sha256_sidecar(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
