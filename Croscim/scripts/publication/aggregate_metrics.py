#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contrib.SST.evaluation.io import atomic_write_json
from contrib.SST.evaluation.metrics import (
    WeightedSufficientStats,
    circular_block_bootstrap_indices,
)
from contrib.SST.evaluation.protocol import load_manifest


STAT_FIELDS = list(WeightedSufficientStats.__dataclass_fields__)
METRIC_FIELDS = ["rmse_c", "mae_c", "bias_c", "correlation", "target_std_c", "nrmse"]


def season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def metrics_from_totals(totals: np.ndarray) -> dict[str, np.ndarray]:
    index = {name: position for position, name in enumerate(STAT_FIELDS)}
    sum_w = totals[..., index["sum_w"]]
    valid = sum_w > 0
    target_mean = np.divide(totals[..., index["sum_w_target"]], sum_w, out=np.zeros_like(sum_w), where=valid)
    prediction_mean = np.divide(totals[..., index["sum_w_prediction"]], sum_w, out=np.zeros_like(sum_w), where=valid)
    target_var = np.maximum(
        np.divide(totals[..., index["sum_w_target_sq"]], sum_w, out=np.zeros_like(sum_w), where=valid)
        - target_mean**2,
        0.0,
    )
    prediction_var = np.maximum(
        np.divide(totals[..., index["sum_w_prediction_sq"]], sum_w, out=np.zeros_like(sum_w), where=valid)
        - prediction_mean**2,
        0.0,
    )
    covariance = (
        np.divide(totals[..., index["sum_w_cross"]], sum_w, out=np.zeros_like(sum_w), where=valid)
        - target_mean * prediction_mean
    )
    rmse = np.sqrt(np.divide(totals[..., index["sum_w_sq_error"]], sum_w, out=np.full_like(sum_w, np.nan), where=valid))
    target_std = np.sqrt(target_var)
    denominator = np.sqrt(target_var * prediction_var)
    return {
        "rmse_c": rmse,
        "mae_c": np.divide(totals[..., index["sum_w_abs_error"]], sum_w, out=np.full_like(sum_w, np.nan), where=valid),
        "bias_c": np.divide(totals[..., index["sum_w_error"]], sum_w, out=np.full_like(sum_w, np.nan), where=valid),
        "correlation": np.divide(covariance, denominator, out=np.full_like(sum_w, np.nan), where=denominator > 0),
        "target_std_c": target_std,
        "nrmse": np.divide(rmse, target_std, out=np.full_like(sum_w, np.nan), where=target_std > 0),
    }


def aggregate_rows(frame: pd.DataFrame, period_type: str, period_value: str) -> list[dict]:
    rows = []
    for keys, group in frame.groupby(["method", "support", "regime"], sort=True):
        stats = WeightedSufficientStats()
        for _, row in group.iterrows():
            stats.merge(WeightedSufficientStats(**{field: row[field] for field in STAT_FIELDS}))
        rows.append({
            "period_type": period_type,
            "period": period_value,
            "method": keys[0],
            "support": keys[1],
            "regime": keys[2],
            "n_dates": int(group["date"].nunique()),
            **stats.metrics(),
        })
    return rows


def stable_seed(seed: int, token: str) -> int:
    return seed ^ int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:4], "big")


def bootstrap_rows(
    frame: pd.DataFrame,
    annual_period: str,
    seed: int,
    n_bootstrap: int,
    block_days: int,
) -> list[dict]:
    rows = []
    periods = [("annual", annual_period, frame)]
    periods.extend(("season", name, frame[frame["season"] == name]) for name in ("DJF", "MAM", "JJA", "SON"))
    for period_type, period_value, period_frame in periods:
        dates = sorted(period_frame["date"].unique())
        effective_seed = stable_seed(seed, f"{period_type}:{period_value}")
        samples = circular_block_bootstrap_indices(
            len(dates),
            block_days=block_days,
            n_bootstrap=n_bootstrap,
            seed=effective_seed,
        )
        date_position = {date: index for index, date in enumerate(dates)}
        for keys, group in period_frame.groupby(["method", "support", "regime"], sort=True):
            matrix = np.zeros((len(dates), len(STAT_FIELDS)), dtype=np.float64)
            for _, row in group.iterrows():
                matrix[date_position[row["date"]]] = [row[field] for field in STAT_FIELDS]
            totals = matrix[samples].sum(axis=1)
            metrics = metrics_from_totals(totals)
            for metric_name in ("rmse_c", "mae_c", "bias_c", "correlation", "nrmse"):
                values = metrics[metric_name]
                rows.append({
                    "period_type": period_type,
                    "period": period_value,
                    "method": keys[0],
                    "support": keys[1],
                    "regime": keys[2],
                    "metric": metric_name,
                    "lower_95": float(np.nanpercentile(values, 2.5)),
                    "median": float(np.nanpercentile(values, 50.0)),
                    "upper_95": float(np.nanpercentile(values, 97.5)),
                    "n_bootstrap": n_bootstrap,
                    "block_days": block_days,
                    "seed": seed,
                    "effective_seed": effective_seed,
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runtime(rows: list[dict]) -> dict:
    end_to_end = np.asarray(
        [row["date_processing_through_netcdf_seconds"] for row in rows],
        dtype=np.float64,
    )
    inference = np.asarray([row["inference_pipeline_seconds"] for row in rows], dtype=np.float64)
    started = pd.to_datetime([row["started_at_utc"] for row in rows], utc=True)
    finished = pd.to_datetime([row["finished_at_utc"] for row in rows], utc=True)
    wall_span_seconds = float((finished.max() - started.min()).total_seconds())
    return {
        "n_dates": len(rows),
        "mean_seconds_per_date": float(end_to_end.mean()),
        "median_seconds_per_date": float(np.median(end_to_end)),
        "p95_seconds_per_date": float(np.percentile(end_to_end, 95)),
        "mean_inference_seconds_per_date": float(inference.mean()),
        "total_single_gpu_processing_hours": float(end_to_end.sum() / 3600.0),
        "total_single_gpu_inference_hours": float(inference.sum() / 3600.0),
        "dates_per_gpu_hour": float(len(rows) * 3600.0 / end_to_end.sum()),
        "observed_parallel_wall_span_hours": wall_span_seconds / 3600.0,
        "observed_dates_per_wall_hour": (
            float(len(rows) * 3600.0 / wall_span_seconds) if wall_span_seconds > 0 else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Croscim publication metrics")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--mode", choices=("controlled", "rectangles"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    expected_dates = [record["central_date"] for record in manifest["records"]]
    if len(expected_dates) != len(set(expected_dates)):
        raise RuntimeError("Manifest contains duplicate dates")
    metrics_root = Path(args.evaluation_root) / "metrics_daily" / args.mode
    all_rows = []
    assembly_rows = []
    gradient_rows = []
    runtime_rows = []
    missing = []
    for date in expected_dates:
        path = metrics_root / f"{date}.json"
        if not path.exists():
            missing.append(date)
            continue
        payload = json.loads(path.read_text())
        if payload["record"]["central_date"] != date or payload["mode"] != args.mode:
            raise RuntimeError(f"Metrics identity mismatch in {path}")
        all_rows.extend(payload["sufficient_statistics"])
        if "runtime" not in payload:
            raise RuntimeError(f"Runtime measurements are missing from {path}")
        runtime = payload["runtime"]
        resolution_seconds = runtime["resolution_seconds"]
        runtime_rows.append({
            "date": date,
            "started_at_utc": runtime["started_at_utc"],
            "finished_at_utc": runtime["finished_at_utc"],
            "device_name": runtime["device_name"],
            "batch_size": runtime["batch_size"],
            "num_workers": runtime["num_workers"],
            "n_patches_total": runtime["n_patches_total"],
            "setup_seconds": runtime["setup_seconds"],
            "x10_seconds": resolution_seconds["x10"],
            "x3_seconds": resolution_seconds["x3"],
            "x1_seconds": resolution_seconds["x1"],
            "inference_pipeline_seconds": runtime["inference_pipeline_seconds"],
            "postprocessing_seconds": runtime["postprocessing_seconds"],
            "netcdf_write_seconds": runtime["netcdf_write_seconds"],
            "date_processing_through_netcdf_seconds": runtime[
                "date_processing_through_netcdf_seconds"
            ],
            "peak_cuda_allocated_gib": runtime["peak_cuda_allocated_gib"],
            "peak_cuda_reserved_gib": runtime["peak_cuda_reserved_gib"],
        })
        for stage, diagnostics in payload["assembly"].items():
            assembly_rows.append({
                "date": date,
                "stage": stage,
                "n_patches": diagnostics["n_patches"],
                "n_uncovered_pixels": diagnostics["n_uncovered_pixels"],
                "min_geometric_coverage": diagnostics["min_geometric_coverage"],
                "max_geometric_coverage": diagnostics["max_geometric_coverage"],
                "n_nonfinite_assembled": diagnostics["n_nonfinite_assembled"],
                "overlap_std_mean_c": diagnostics["overlap_std_mean_c"],
                "overlap_std_p95_c": diagnostics["overlap_std_p95_c"],
                "seam_margin_pixels": payload["seams"][stage]["margin_pixels"],
                "seam_rmse_c": payload["seams"][stage]["seam"]["rmse_c"],
                "interior_rmse_c": payload["seams"][stage]["interior"]["rmse_c"],
            })
        for method, supports in payload["gradient_rmse_c_per_km"].items():
            for support, value in supports.items():
                gradient_rows.append({
                    "date": date,
                    "method": method,
                    "support": support,
                    "gradient_rmse_c_per_km": value,
                })
    if missing:
        raise RuntimeError(f"Missing {len(missing)} daily metrics, first dates: {missing[:10]}")

    frame = pd.DataFrame(all_rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month
    frame["season"] = frame["month"].map(season)
    years = sorted(int(year) for year in frame["date"].dt.year.unique())
    annual_period = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    summary = aggregate_rows(frame, "annual", annual_period)
    for month in range(1, 13):
        summary.extend(aggregate_rows(frame[frame["month"] == month], "month", f"{month:02d}"))
    for name in ("DJF", "MAM", "JJA", "SON"):
        summary.extend(aggregate_rows(frame[frame["season"] == name], "season", name))
    bootstrap = bootstrap_rows(
        frame,
        annual_period,
        args.seed,
        args.bootstrap,
        args.block_days,
    )

    output_dir = Path(args.output_dir)
    daily_path = output_dir / "metrics_daily_sufficient.csv"
    summary_path = output_dir / "metrics_summary.csv"
    bootstrap_path = output_dir / "bootstrap_intervals.csv"
    assembly_path = output_dir / "assembly_metrics.csv"
    gradient_path = output_dir / "gradient_metrics.csv"
    runtime_daily_path = output_dir / "runtime_daily.csv"
    runtime_summary_path = output_dir / "runtime_summary.csv"
    write_csv(daily_path, frame.drop(columns=["month", "season"]).to_dict("records"))
    write_csv(summary_path, summary)
    write_csv(bootstrap_path, bootstrap)
    write_csv(assembly_path, assembly_rows)
    write_csv(gradient_path, gradient_rows)
    write_csv(runtime_daily_path, runtime_rows)
    write_csv(runtime_summary_path, [summarize_runtime(runtime_rows)])

    annual_global = {
        (row["method"], row["support"]): row
        for row in summary
        if row["period_type"] == "annual"
        and row["regime"] == "global"
    }
    interval_lookup = {
        (row["method"], row["support"], row["metric"]): row
        for row in bootstrap
        if row["period_type"] == "annual"
        and row["period"] == annual_period
        and row["regime"] == "global"
    }
    main_table = []
    for method in ("dmi_oi", "croscim_x1"):
        hidden = annual_global[(method, "hidden")]
        visible = annual_global[(method, "visible")]
        row = {
            "method": method,
            "n_hidden_pixels": hidden["n_pixels"],
            "n_visible_pixels": visible["n_pixels"],
            "hidden_rmse_c": hidden["rmse_c"],
            "hidden_nrmse": hidden["nrmse"],
            "hidden_mae_c": hidden["mae_c"],
            "hidden_bias_c": hidden["bias_c"],
            "hidden_correlation": hidden["correlation"],
            "visible_rmse_c": visible["rmse_c"],
        }
        for support, metric in (
            ("hidden", "rmse_c"),
            ("hidden", "nrmse"),
            ("hidden", "mae_c"),
            ("hidden", "bias_c"),
            ("hidden", "correlation"),
            ("visible", "rmse_c"),
        ):
            interval = interval_lookup[(method, support, metric)]
            prefix = f"{support}_{metric}"
            row[f"{prefix}_lower_95"] = interval["lower_95"]
            row[f"{prefix}_upper_95"] = interval["upper_95"]
        main_table.append(row)
    stages = [
        row for row in summary
        if row["period_type"] == "annual"
        and row["support"] == "hidden"
        and row["regime"] == "global"
        and row["method"] in {"croscim_x10", "croscim_x3", "croscim_x1"}
    ]
    write_csv(output_dir / "table_main_croscim_vs_dmi_oi.csv", main_table)
    write_csv(output_dir / "table_resolution_refinement.csv", stages)
    marker = output_dir / "aggregation_complete.json"
    atomic_write_json(marker, {
        "schema_version": 1,
        "manifest": str(Path(args.manifest).resolve()),
        "mode": args.mode,
        "n_dates": len(expected_dates),
        "bootstrap": {"replicates": args.bootstrap, "block_days": args.block_days, "seed": args.seed},
    })
    print(f"summary={summary_path}")
    print(f"bootstrap={bootstrap_path}")
    print(f"runtime={runtime_summary_path}")


if __name__ == "__main__":
    main()
