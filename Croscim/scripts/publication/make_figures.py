#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


LAND_COLOR = "#8B4513"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def temperature_cmap():
    cmap = mpl.colormaps["turbo"].copy()
    cmap.set_bad("white")
    return cmap


def error_cmap():
    cmap = mpl.colormaps["Reds"].copy()
    cmap.set_bad("white")
    return cmap


def draw_land(axis, land):
    overlay = np.ma.masked_where(~land, np.ones_like(land, dtype=np.float32))
    axis.imshow(
        overlay,
        origin="lower",
        cmap=mpl.colors.ListedColormap([LAND_COLOR]),
        vmin=0,
        vmax=1,
    )


def draw_field(axis, values, land, *, cmap, vmin, vmax, title):
    image = axis.imshow(np.ma.masked_where(land, values), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    draw_land(axis, land)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def figure_b1(dataset: xr.Dataset, case: dict, output_dir: Path) -> None:
    y0, x0 = case["lat_start"], case["lon_start"]
    selection = np.s_[y0:y0 + 256, x0:x0 + 256]
    target = dataset["target_sst"].values[selection]
    land = dataset["surfmask"].values[selection] == 0
    finite = target[np.isfinite(target) & ~land]
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)
    images = []
    for axis, variable, title in zip(
        axes,
        ("pred_sst_x10", "pred_sst_x3", "pred_sst_x1"),
        (r"Coarse $\times 10$", r"Refinement $\times 3$", r"Final $\times 1$"),
    ):
        images.append(draw_field(axis, dataset[variable].values[selection], land, cmap=temperature_cmap(), vmin=vmin, vmax=vmax, title=title))
    fig.colorbar(images[-1], ax=axes, label=r"Surface temperature ($^\circ$C)", fraction=0.025)
    save_figure(fig, output_dir, "figure_b1_final_hierarchy_data")


def figure_b2(dataset: xr.Dataset, output_dir: Path) -> None:
    target = dataset["target_sst"].values
    observed = dataset["observed_sst"].values
    prediction = dataset["pred_sst_x1"].values
    land = dataset["surfmask"].values == 0
    finite = target[np.isfinite(target) & ~land]
    vmin, vmax = np.nanpercentile(finite, [1, 99])
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), constrained_layout=True)
    image_temp = draw_field(axes[0, 0], observed, land, cmap=temperature_cmap(), vmin=vmin, vmax=vmax, title="Withheld fused channel")
    draw_field(axes[0, 1], prediction, land, cmap=temperature_cmap(), vmin=vmin, vmax=vmax, title="Global reconstruction")
    coverage = dataset["patch_coverage_x1"].values
    image_cov = axes[1, 0].imshow(np.ma.masked_where(land, coverage), origin="lower", cmap="viridis")
    draw_land(axes[1, 0], land)
    axes[1, 0].set_title("Patch coverage")
    disagreement = dataset["patch_disagreement_x1"].values
    maximum = max(float(np.nanpercentile(disagreement[~land], 99)), 1e-6)
    image_dis = draw_field(axes[1, 1], disagreement, land, cmap=error_cmap(), vmin=0, vmax=maximum, title="Overlap disagreement")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image_temp, ax=axes[0], label=r"$^\circ$C", fraction=0.02)
    fig.colorbar(image_cov, ax=axes[1, 0], label="Contributions", fraction=0.04)
    fig.colorbar(image_dis, ax=axes[1, 1], label=r"Std. ($^\circ$C)", fraction=0.04)
    save_figure(fig, output_dir, "figure_b2_global_reconstruction")


def figure_b3(evaluation_root: Path, mode: str, cases: list[dict], output_dir: Path) -> None:
    rows = []
    for case in cases:
        with xr.open_dataset(evaluation_root / "daily" / mode / f"{case['date']}.nc") as dataset:
            y0, x0 = case["lat_start"], case["lon_start"]
            selection = np.s_[y0:y0 + 256, x0:x0 + 256]
            rows.append({
                "case": case,
                "fields": [
                    dataset[name].values[selection]
                    for name in ("observed_sst", "target_sst", "pred_sst_x1")
                ],
                "hidden": dataset["hidden_mask"].values[selection].astype(bool),
                "land": dataset["surfmask"].values[selection] == 0,
            })
    target_values = np.concatenate([
        row["fields"][1][np.isfinite(row["fields"][1]) & ~row["land"]]
        for row in rows
    ])
    vmin, vmax = np.nanpercentile(target_values, [2, 98])
    hidden_errors = np.concatenate([
        np.abs(row["fields"][2] - row["fields"][1])[row["hidden"]]
        for row in rows
    ])
    error_max = max(float(np.nanpercentile(hidden_errors, 98)), 1e-6)

    fig, axes = plt.subplots(4, 4, figsize=(7.0, 7.1), constrained_layout=True)
    for row_index, row in enumerate(rows):
        case = row["case"]
        fields = row["fields"]
        land = row["land"]
        for column, (field, title) in enumerate(zip(fields, ("Input", "Revealed target", "Croscim"))):
            temperature_image = draw_field(
                axes[row_index, column], field, land, cmap=temperature_cmap(),
                vmin=vmin, vmax=vmax, title=title if row_index == 0 else "",
            )
        error = np.where(row["hidden"], np.abs(fields[2] - fields[1]), np.nan)
        error_image = draw_field(
            axes[row_index, 3], error, land, cmap=error_cmap(), vmin=0,
            vmax=error_max, title="Hidden-pixel error" if row_index == 0 else "",
        )
        axes[row_index, 0].set_ylabel(
            f"{case['category'].replace('_', ' ').title()}\n{case['missingness']:.0%} withheld",
            labelpad=5,
        )
    fig.colorbar(temperature_image, ax=axes[:, :3], label=r"Surface temperature ($^\circ$C)", fraction=0.012)
    fig.colorbar(error_image, ax=axes[:, 3], label=r"Absolute error ($^\circ$C)", fraction=0.03)
    save_figure(fig, output_dir, "figure_b3_controlled_cases")


def patch_gallery(evaluation_root: Path, mode: str, cases: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, start=1):
        path = evaluation_root / "daily" / mode / f"{case['date']}.nc"
        with xr.open_dataset(path) as dataset:
            y0, x0 = case["lat_start"], case["lon_start"]
            selection = {"lat": slice(y0, y0 + 256), "lon": slice(x0, x0 + 256)}
            fields = [
                dataset[name].isel(selection).values
                for name in (
                    "observed_sst", "target_sst", "pred_sst_x10",
                    "pred_sst_x3", "pred_sst_x1",
                )
            ]
            hidden = dataset["hidden_mask"].isel(selection).values.astype(bool)
            land = dataset["surfmask"].isel(selection).values == 0

        target_values = fields[1][np.isfinite(fields[1]) & ~land]
        vmin, vmax = np.nanpercentile(target_values, [2, 98])
        absolute_error = np.where(hidden, np.abs(fields[4] - fields[1]), np.nan)
        finite_error = absolute_error[np.isfinite(absolute_error)]
        error_max = max(float(np.nanpercentile(finite_error, 98)), 0.25)
        fig, axes = plt.subplots(1, 6, figsize=(16.0, 2.8), constrained_layout=True)
        titles = ("Withheld input", "Revealed target", r"$\times10$", r"$\times3$", r"$\times1$", "Hidden error")
        for axis, field, title in zip(axes[:5], fields, titles[:5]):
            temperature_image = draw_field(
                axis, field, land, cmap=temperature_cmap(),
                vmin=vmin, vmax=vmax, title=title,
            )
        error_image = draw_field(
            axes[5], absolute_error, land, cmap=error_cmap(),
            vmin=0, vmax=error_max, title=titles[5],
        )
        fig.colorbar(temperature_image, ax=axes[:5], label=r"$^\circ$C", fraction=0.012)
        fig.colorbar(error_image, ax=axes[5], label=r"$^\circ$C", fraction=0.04)
        fig.suptitle(
            f"{case['patch_id']} | {case['category'].replace('_', ' ')} | "
            f"withheld={case['missingness']:.0%} | x1 RMSE={case['hidden_rmse_x1_c']:.3f} degC",
            fontsize=9,
        )
        filename = (
            f"{index:03d}_{case['gallery_group']}_{case['category']}_"
            f"rmse_{case['hidden_rmse_x1_c']:.3f}_{case['patch_id']}.png"
        )
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def figure_b4(
    summary_path: Path,
    daily_path: Path,
    bootstrap_path: Path,
    output_dir: Path,
    stem: str = "figure_b4_quantitative_diagnostics",
) -> None:
    summary = pd.read_csv(summary_path)
    daily = pd.read_csv(daily_path)
    intervals = pd.read_csv(bootstrap_path)
    annual = summary[(summary.period_type == "annual") & (summary.support == "hidden") & (summary.regime == "global")]
    stages = annual[annual.method.isin(["croscim_x10", "croscim_x3", "croscim_x1"])].copy()
    stages["order"] = stages.method.map({"croscim_x10": 0, "croscim_x3": 1, "croscim_x1": 2})
    stages = stages.sort_values("order")
    annual_intervals = intervals[
        (intervals.period_type == "annual")
        & (intervals.support == "hidden")
        & (intervals.regime == "global")
        & (intervals.metric == "rmse_c")
    ].set_index("method")
    stage_lower = np.array([
        annual_intervals.loc[method, "lower_95"] for method in stages.method
    ])
    stage_upper = np.array([
        annual_intervals.loc[method, "upper_95"] for method in stages.method
    ])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    axes[0].bar(
        [r"$\times10$", r"$\times3$", r"$\times1$"],
        stages.rmse_c,
        yerr=np.vstack((stages.rmse_c.to_numpy() - stage_lower, stage_upper - stages.rmse_c.to_numpy())),
        capsize=2,
        color=["#3274A1", "#E1812C", "#C44E52"],
    )
    axes[0].set_ylabel(r"Hidden-pixel RMSE ($^\circ$C)")
    axes[0].set_title("Resolution refinement")

    x1_daily = daily[(daily.method == "croscim_x1") & (daily.support == "hidden") & (daily.regime == "global")].copy()
    x1_daily["rmse_c"] = np.sqrt(x1_daily.sum_w_sq_error / x1_daily.sum_w)
    axes[1].scatter(
        x1_daily.missing_fraction,
        x1_daily.rmse_c,
        s=8,
        alpha=0.45,
        color="#3274A1",
        linewidth=0,
    )
    axes[1].set_xlabel("Withheld fraction")
    axes[1].set_ylabel(r"RMSE ($^\circ$C)")
    axes[1].set_title("Missingness sensitivity")

    seasons = ("DJF", "MAM", "JJA", "SON")
    seasonal = summary[
        (summary.period_type == "season")
        & (summary.support == "hidden")
        & (summary.regime == "global")
    ]
    seasonal_intervals = intervals[
        (intervals.period_type == "season")
        & (intervals.support == "hidden")
        & (intervals.regime == "global")
        & (intervals.metric == "rmse_c")
    ]
    for method, label, color in (("croscim_x1", "Croscim", "#3274A1"), ("dmi_oi", "DMI-OI", "#C44E52")):
        values = seasonal[seasonal.method == method].set_index("period").reindex(seasons)
        bounds = seasonal_intervals[seasonal_intervals.method == method].set_index("period").reindex(seasons)
        axes[2].errorbar(
            seasons,
            values.rmse_c,
            yerr=np.vstack((values.rmse_c - bounds.lower_95, bounds.upper_95 - values.rmse_c)),
            marker="o",
            ms=3,
            capsize=2,
            label=label,
            color=color,
        )
    axes[2].set_xlabel("Season")
    axes[2].set_ylabel(r"RMSE ($^\circ$C)")
    axes[2].set_title("Seasonal performance")
    axes[2].legend(frameon=False)
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")
    save_figure(fig, output_dir, stem)


def draw_summary_table(axis, title, columns, rows, widths=None) -> None:
    axis.axis("off")
    axis.set_title(title, loc="left", fontweight="bold", pad=6)
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=widths,
        bbox=[0, 0, 1, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DEE2")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#1F4E5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F5F6")


def evaluation_summary(evaluation_root: Path, output_dir: Path) -> None:
    results = evaluation_root / "results"
    summary = pd.read_csv(results / "metrics_summary.csv")
    main_table = pd.read_csv(results / "table_main_croscim_vs_dmi_oi.csv")
    gradients = pd.read_csv(results / "gradient_metrics.csv")
    assembly = pd.read_csv(results / "assembly_metrics.csv")
    runtime = pd.read_csv(results / "runtime_summary.csv").iloc[0]
    provenance = json.loads((evaluation_root / "provenance" / "run_controlled.json").read_text())
    aggregation = json.loads((results / "aggregation_complete.json").read_text())

    annual = summary[summary.period_type == "annual"]
    period = str(annual.period.iloc[0])
    global_hidden = annual[(annual.support == "hidden") & (annual.regime == "global")]
    global_visible = annual[(annual.support == "visible") & (annual.regime == "global")]

    labels = {"dmi_oi": "DMI-OI", "croscim_x1": "CROSCIM x1"}
    main_rows = []
    for method in ("croscim_x1", "dmi_oi"):
        row = main_table[main_table.method == method].iloc[0]
        main_rows.append([
            labels[method],
            f"{row.hidden_rmse_c:.3f} [{row.hidden_rmse_c_lower_95:.3f}, {row.hidden_rmse_c_upper_95:.3f}]",
            f"{row.hidden_nrmse:.3f}",
            f"{row.hidden_mae_c:.3f}",
            f"{row.hidden_bias_c:+.3f}",
            f"{row.hidden_correlation:.4f}",
            f"{row.visible_rmse_c:.3f}",
        ])

    gradient_means = gradients.groupby(["method", "support"])["gradient_rmse_c_per_km"].mean()
    cascade_rows = []
    previous_rmse = None
    stage_changes = {}
    for method, label in (("croscim_x10", "x10"), ("croscim_x3", "x3"), ("croscim_x1", "x1")):
        hidden = global_hidden[global_hidden.method == method].iloc[0]
        visible = global_visible[global_visible.method == method].iloc[0]
        change = "-" if previous_rmse is None else f"{100 * (hidden.rmse_c / previous_rmse - 1):+.1f}%"
        if previous_rmse is not None:
            stage_changes[label] = 100 * (hidden.rmse_c / previous_rmse - 1)
        previous_rmse = hidden.rmse_c
        cascade_rows.append([
            label,
            f"{hidden.rmse_c:.3f}",
            change,
            f"{visible.rmse_c:.3f}",
            f"{gradient_means.loc[(method, 'hidden')]:.3f}",
            f"{gradient_means.loc[(method, 'visible')]:.3f}",
        ])

    x1_global = global_hidden[global_hidden.method == "croscim_x1"].iloc[0]
    regimes = annual[
        (annual.method == "croscim_x1")
        & (annual.support == "hidden")
        & (annual.regime != "global")
    ].set_index("regime")
    regime_rows = []
    regime_labels = {"open_ocean": "Open ocean", "coastal": "Coastal", "high_ice": "High ice"}
    sse_shares = {}
    for regime in ("open_ocean", "coastal", "high_ice"):
        row = regimes.loc[regime]
        weight_share = row.sum_w / x1_global.sum_w
        sse_share = row.sum_w * row.rmse_c ** 2 / (x1_global.sum_w * x1_global.rmse_c ** 2)
        sse_shares[regime] = sse_share
        regime_rows.append([
            regime_labels[regime],
            f"{100 * weight_share:.1f}%",
            f"{100 * sse_share:.1f}%",
            f"{row.rmse_c:.3f}",
            f"{row.mae_c:.3f}",
            f"{row.bias_c:+.3f}",
        ])

    assembly_rows = []
    for stage in ("x10", "x3", "x1"):
        rows = assembly[assembly.stage == stage]
        assembly_rows.append([
            stage,
            f"{int(rows.n_patches.median())}",
            f"{int(rows.n_uncovered_pixels.max())}",
            f"{rows.overlap_std_mean_c.mean():.3f}",
            f"{rows.seam_rmse_c.mean() / rows.interior_rmse_c.mean():.3f}",
        ])

    diagnostic = plt.imread(results / "diagnostics" / "quantitative_diagnostics.png")
    fig = plt.figure(figsize=(15.5, 11.0))
    grid = fig.add_gridspec(
        4, 2, height_ratios=(1.6, 0.75, 0.9, 0.8),
        left=0.04, right=0.97, top=0.90, bottom=0.05, hspace=0.32, wspace=0.16,
    )
    checkpoint = provenance["checkpoint"]
    fig.suptitle(
        f"CROSCIM controlled evaluation - {period}",
        x=0.04, y=0.965, ha="left", fontsize=18, fontweight="bold",
    )
    fig.text(
        0.04, 0.925,
        f"Checkpoint epoch {checkpoint['epoch']} | {aggregation['n_dates']} dates | "
        "area-weighted metrics on matched pixels",
        ha="left", fontsize=10, color="#40515A",
    )

    axis_diagnostic = fig.add_subplot(grid[0, :])
    axis_diagnostic.imshow(diagnostic)
    axis_diagnostic.axis("off")

    axis_main = fig.add_subplot(grid[1, :])
    draw_summary_table(
        axis_main,
        "Primary matched-support metrics (degC except NRMSE and correlation)",
        ("Method", "Hidden RMSE [95% CI]", "NRMSE", "MAE", "Bias", "Correlation", "Visible RMSE"),
        main_rows,
        widths=(0.13, 0.23, 0.10, 0.10, 0.10, 0.14, 0.15),
    )

    axis_cascade = fig.add_subplot(grid[2, 0])
    draw_summary_table(
        axis_cascade,
        "Resolution cascade",
        ("Stage", "Hidden RMSE", "Change", "Visible RMSE", "Hidden grad.", "Visible grad."),
        cascade_rows,
        widths=(0.10, 0.19, 0.13, 0.19, 0.19, 0.19),
    )
    axis_regimes = fig.add_subplot(grid[2, 1])
    draw_summary_table(
        axis_regimes,
        "CROSCIM x1 hidden errors by regime",
        ("Regime", "Area weight", "Error share", "RMSE", "MAE", "Bias"),
        regime_rows,
        widths=(0.22, 0.16, 0.16, 0.14, 0.14, 0.14),
    )

    axis_assembly = fig.add_subplot(grid[3, 0])
    draw_summary_table(
        axis_assembly,
        "Global patch assembly",
        ("Stage", "Patches/date", "Uncovered max", "Overlap std.", "Seam/interior"),
        assembly_rows,
        widths=(0.12, 0.22, 0.22, 0.20, 0.22),
    )

    axis_run = fig.add_subplot(grid[3, 1])
    axis_run.axis("off")
    axis_run.set_title("Run summary and direct reading", loc="left", fontweight="bold", pad=6)
    lines = [
        f"Checkpoint: {Path(checkpoint['path']).name}",
        f"SHA-256: {checkpoint['sha256'][:16]}... | global step {checkpoint['global_step']}",
        f"Mean inference/date: {runtime.mean_inference_seconds_per_date:.1f} s "
        f"({provenance['num_shards']} concurrent workers)",
        f"Parallel wall time: {runtime.observed_parallel_wall_span_hours * 60:.1f} min "
        f"({runtime.observed_dates_per_wall_hour:.1f} dates/h)",
        f"x10 to x3 hidden RMSE: {stage_changes['x3']:+.1f}% | x3 to x1: {stage_changes['x1']:+.1f}%",
        f"High ice: {100 * sse_shares['high_ice']:.1f}% of weighted squared error",
    ]
    axis_run.text(0.0, 0.86, "\n".join(lines), va="top", fontsize=8.2, linespacing=1.45)
    axis_run.text(
        0.0, 0.02,
        "DMI-OI is an advantaged operational reference: it was not rerun after withholding "
        "and may have assimilated the hidden observations.",
        va="bottom", fontsize=7.5, color="#7A2E2E", wrap=True,
    )
    save_figure(fig, output_dir, "evaluation_summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review galleries or appendix-B figures")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--mode", default="controlled")
    parser.add_argument("--kind", choices=("gallery", "diagnostics", "report", "final"), default="final")
    parser.add_argument("--cases")
    parser.add_argument("--summary")
    parser.add_argument("--daily-metrics")
    parser.add_argument("--bootstrap-intervals")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    configure_style()
    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    if args.kind in {"gallery", "final"} and not args.cases:
        parser.error(f"--cases is required for --kind {args.kind}")
    if args.kind in {"diagnostics", "final"} and (
        not args.summary or not args.daily_metrics or not args.bootstrap_intervals
    ):
        parser.error(
            f"--summary, --daily-metrics and --bootstrap-intervals are required for --kind {args.kind}"
        )

    if args.kind == "gallery":
        cases = json.loads(Path(args.cases).read_text())["cases"]
        patch_gallery(evaluation_root, args.mode, cases, output_dir)
    elif args.kind == "diagnostics":
        figure_b4(
            Path(args.summary), Path(args.daily_metrics),
            Path(args.bootstrap_intervals), output_dir,
            stem="quantitative_diagnostics",
        )
    elif args.kind == "report":
        evaluation_summary(evaluation_root, output_dir)
    else:
        cases = json.loads(Path(args.cases).read_text())["cases"]
        if len(cases) != 4:
            raise RuntimeError("Final appendix figure generation requires exactly four selected cases")
        representative = cases[0]
        with xr.open_dataset(evaluation_root / "daily" / args.mode / f"{representative['date']}.nc") as dataset:
            dataset.load()
            figure_b1(dataset, representative, output_dir)
            figure_b2(dataset, output_dir)
        figure_b3(evaluation_root, args.mode, cases, output_dir)
        figure_b4(
            Path(args.summary), Path(args.daily_metrics),
            Path(args.bootstrap_intervals), output_dir,
        )
    print(f"figures={output_dir}")


if __name__ == "__main__":
    main()
