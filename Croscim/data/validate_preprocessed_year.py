"""Validate that preprocessed satellite means were not read as uncertainties."""

import argparse
from pathlib import Path

import numpy as np
import zarr


SATELLITES = ("aasti", "avhrr", "pmw", "slstr")


def finite_ratio(values):
    return float(np.isfinite(values).mean())


def validate_year(year, data_root):
    year_dir = Path(data_root) / f"data_{year}"
    stores = sorted(year_dir.glob(f"{year}*_x10.zarr"))
    expected = 366 if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0) else 365
    if len(stores) != expected:
        raise RuntimeError(
            f"{year}: found {len(stores)} x10 stores, expected {expected}"
        )

    coverage = {satellite: [] for satellite in SATELLITES}
    target_coverage = []
    for store in stores:
        group = zarr.open_group(str(store), mode="r")
        arrays = {}
        for satellite in SATELLITES:
            mean = np.asarray(group[f"{satellite}_av"][:])
            uncertainty = np.asarray(group[f"{satellite}_std"][:])
            if (
                np.isfinite(mean).any()
                and np.isfinite(uncertainty).any()
                and np.array_equal(mean, uncertainty, equal_nan=True)
            ):
                raise RuntimeError(
                    f"{store.name}: {satellite}_av is identical to "
                    f"{satellite}_std"
                )
            arrays[satellite] = mean
            coverage[satellite].append(finite_ratio(mean))

        sea_ice = np.asarray(group["sea_ice_fraction"][:])
        surfmask = np.asarray(group["surfmask"][:])
        target = np.where(sea_ice >= 0.70, arrays["aasti"], arrays["slstr"])
        ocean = surfmask != 0
        ocean_count = int(ocean.sum())
        target_coverage.append(
            float(np.isfinite(target[ocean]).mean()) if ocean_count else 0.0
        )

    if max(coverage["slstr"], default=0.0) == 0.0:
        raise RuntimeError(f"{year}: SLSTR mean is empty for the entire year")
    if max(target_coverage, default=0.0) == 0.0:
        raise RuntimeError(f"{year}: fused target is empty for the entire year")

    print(f"Validated {year}: {len(stores)} x10 stores")
    for satellite in SATELLITES:
        values = np.asarray(coverage[satellite]) * 100.0
        print(
            f"  {satellite.upper()} mean finite: "
            f"min={values.min():.2f}% median={np.median(values):.2f}% "
            f"max={values.max():.2f}%"
        )
    values = np.asarray(target_coverage) * 100.0
    print(
        "  Fused target over non-land pixels: "
        f"min={values.min():.2f}% median={np.median(values):.2f}% "
        f"max={values.max():.2f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    validate_year(args.year, args.data_root)


if __name__ == "__main__":
    main()
