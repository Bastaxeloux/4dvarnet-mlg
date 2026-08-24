"""Plot the fields used to diagnose one preprocessed SST Zarr store."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import zarr


SATELLITES = ("aasti", "avhrr", "pmw", "slstr")


def _read_2d(group, name):
    values = np.squeeze(np.asarray(group[name][:]))
    if values.ndim != 2:
        raise ValueError(f"{name} has shape {values.shape}; expected a 2-D field")
    return values


def _finite_percent(values):
    return 100.0 * float(np.isfinite(values).mean())


def _plot_field(ax, values, title, cmap, vmin=None, vmax=None):
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.04)


def plot_diagnostic(store_path, output_path):
    group = zarr.open_group(str(store_path), mode="r")
    means = {sat: _read_2d(group, f"{sat}_av") for sat in SATELLITES}
    uncertainties = {sat: _read_2d(group, f"{sat}_std") for sat in SATELLITES}
    sea_ice = _read_2d(group, "sea_ice_fraction")
    surfmask = _read_2d(group, "surfmask")
    fused_target = np.where(sea_ice >= 0.70, means["aasti"], means["slstr"])

    fig, axes = plt.subplots(4, 4, figsize=(24, 18), constrained_layout=True)
    fig.suptitle(str(store_path), fontsize=12)

    for col, satellite in enumerate(SATELLITES):
        mean = means[satellite]
        uncertainty = uncertainties[satellite]
        mean_vmin = -45.0 if satellite == "aasti" else -2.0
        _plot_field(
            axes[0, col],
            mean,
            f"{satellite.upper()} mean\n{_finite_percent(mean):.2f}% finite",
            "RdYlBu_r",
            mean_vmin,
            35.0,
        )
        _plot_field(
            axes[1, col],
            uncertainty,
            f"{satellite.upper()} uncertainty\n{_finite_percent(uncertainty):.2f}% finite",
            "viridis",
            0.0,
            1.0,
        )
        axes[2, col].imshow(np.isfinite(mean), origin="lower", aspect="auto", cmap="gray_r", vmin=0, vmax=1)
        axes[2, col].set_title(f"{satellite.upper()} mean availability", fontsize=11)
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        identical = np.array_equal(mean, uncertainty, equal_nan=True)
        print(
            f"{satellite.upper()}: mean={_finite_percent(mean):.2f}% finite, "
            f"uncertainty={_finite_percent(uncertainty):.2f}% finite, "
            f"mean_equals_uncertainty={identical}"
        )

    _plot_field(
        axes[3, 0],
        fused_target,
        f"Fused target\n{_finite_percent(fused_target):.2f}% finite",
        "RdYlBu_r",
        -45.0,
        35.0,
    )
    axes[3, 1].imshow(
        np.isfinite(fused_target), origin="lower", aspect="auto", cmap="gray_r", vmin=0, vmax=1
    )
    axes[3, 1].set_title("Fused target availability", fontsize=11)
    axes[3, 1].set_xticks([])
    axes[3, 1].set_yticks([])

    _plot_field(axes[3, 2], sea_ice, "Sea-ice fraction", "Blues", 0.0, 1.0)

    surfmask_cmap = ListedColormap(["#8B4513", "#277DA1", "#90BE6D", "#E9C46A"])
    surfmask_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], surfmask_cmap.N)
    surface_image = axes[3, 3].imshow(
        surfmask, origin="lower", aspect="auto", cmap=surfmask_cmap, norm=surfmask_norm
    )
    axes[3, 3].set_title("Surface mask", fontsize=11)
    axes[3, 3].set_xticks([])
    axes[3, 3].set_yticks([])
    colorbar = plt.colorbar(surface_image, ax=axes[3, 3], fraction=0.035, pad=0.04, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(["land", "ocean", "ice-water", "ice"])

    ocean = surfmask != 0
    ocean_coverage = 100.0 * float(np.isfinite(fused_target[ocean]).mean())
    print(f"FUSED TARGET: {_finite_percent(fused_target):.2f}% finite globally")
    print(f"FUSED TARGET: {ocean_coverage:.2f}% finite over non-land pixels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="Daily x1/x3/x10 Zarr store")
    parser.add_argument("--output", type=Path, default=Path("data_diag.png"))
    args = parser.parse_args()
    plot_diagnostic(args.store, args.output)


if __name__ == "__main__":
    main()
