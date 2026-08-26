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


def figure_b4(summary_path: Path, daily_path: Path, output_dir: Path) -> None:
    summary = pd.read_csv(summary_path)
    daily = pd.read_csv(daily_path)
    annual = summary[(summary.period_type == "annual") & (summary.support == "hidden") & (summary.regime == "global")]
    stages = annual[annual.method.isin(["croscim_x10", "croscim_x3", "croscim_x1"])].copy()
    stages["order"] = stages.method.map({"croscim_x10": 0, "croscim_x3": 1, "croscim_x1": 2})
    stages = stages.sort_values("order")
    monthly = summary[(summary.period_type == "month") & (summary.support == "hidden") & (summary.regime == "global")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    axes[0].bar([r"$\times10$", r"$\times3$", r"$\times1$"], stages.rmse_c, color=["#3274A1", "#E1812C", "#C44E52"])
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

    for method, label, color in (("croscim_x1", "Croscim", "#3274A1"), ("dmi_oi", "DMI-OI", "#C44E52")):
        values = monthly[monthly.method == method].sort_values("period")
        axes[2].plot(values.period.astype(int), values.rmse_c, marker="o", ms=3, label=label, color=color)
    years = sorted(pd.to_datetime(daily["date"]).dt.year.unique())
    period_label = str(years[0]) if len(years) == 1 else "evaluation period"
    axes[2].set_xticks([1, 3, 5, 7, 9, 11])
    axes[2].set_xlabel(f"Month of {period_label}")
    axes[2].set_ylabel(r"RMSE ($^\circ$C)")
    axes[2].set_title("Seasonal stability")
    axes[2].legend(frameon=False)
    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")
    save_figure(fig, output_dir, "figure_b4_quantitative_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate appendix-B figures from frozen exports")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--mode", default="controlled")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--daily-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    configure_style()
    evaluation_root = Path(args.evaluation_root)
    output_dir = Path(args.output_dir)
    cases = json.loads(Path(args.cases).read_text())["cases"]
    representative = cases[0]
    with xr.open_dataset(evaluation_root / "daily" / args.mode / f"{representative['date']}.nc") as dataset:
        dataset.load()
        figure_b1(dataset, representative, output_dir)
        figure_b2(dataset, output_dir)
    figure_b3(evaluation_root, args.mode, cases, output_dir)
    figure_b4(Path(args.summary), Path(args.daily_metrics), output_dir)
    print(f"figures={output_dir}")


if __name__ == "__main__":
    main()
