from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from contrib.SST.load_data import COVARIATES, VAR_GROUPS

from .masking import find_daily_store


@dataclass(frozen=True)
class OIVerification:
    units_requested: str
    units_resolved: str
    sample_dates: list[str]
    min_value: float
    max_value: float
    median_value: float
    time_matches_filename: bool
    grid_matches_x1: bool
    excluded_from_model_inputs: bool


def oi_to_celsius(values: np.ndarray, units: str) -> np.ndarray:
    units = units.lower()
    values = np.asarray(values, dtype=np.float32)
    if units == "kelvin":
        return values - np.float32(273.15)
    if units == "celsius":
        return values
    raise ValueError("DMI-OI units must be explicitly set to 'kelvin' or 'celsius'")


def _store_time_matches(store, expected_date: dt.date) -> bool:
    if "time" not in store:
        return False
    values = np.asarray(store["time"][:]).reshape(-1)
    if not values.size:
        return False
    value = values[0]
    try:
        if np.issubdtype(values.dtype, np.datetime64):
            parsed = pd.Timestamp(value)
        else:
            parsed = pd.to_datetime(int(value), unit="ns")
    except (ValueError, OverflowError, TypeError):
        return False
    return parsed == pd.Timestamp(expected_date)


def verify_dmi_oi(
    data_root: str | Path,
    sample_dates: Sequence[dt.date],
    *,
    units: str,
) -> OIVerification:
    if units not in {"kelvin", "celsius"}:
        raise ValueError(
            "Zarr conversion does not preserve analysed_st units; pass --dmi-oi-units kelvin or celsius"
        )
    minima, maxima, medians = [], [], []
    time_matches = []
    grid_matches = []
    reference_lat = reference_lon = None
    for date in sample_dates:
        path = find_daily_store(data_root, date, "x1")
        store = zarr.open(str(path), mode="r")
        if "analysed_st" not in store:
            raise RuntimeError(f"analysed_st is missing from {path}")
        values = np.asarray(store["analysed_st"][:], dtype=np.float32)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise RuntimeError(f"analysed_st has no finite values in {path}")
        minima.append(float(finite.min()))
        maxima.append(float(finite.max()))
        medians.append(float(np.median(finite)))
        lat = np.asarray(store["lat"][:])
        lon = np.asarray(store["lon"][:])
        if reference_lat is None:
            reference_lat, reference_lon = lat, lon
        grid_matches.append(np.array_equal(lat, reference_lat) and np.array_equal(lon, reference_lon))
        time_matches.append(_store_time_matches(store, date))

    model_input_names = {
        f"{group}_{variable}"
        for group, variables in VAR_GROUPS.items()
        for variable in variables
    } | set(COVARIATES) | {"tgt_sst", "lat", "lon", "surfmask", "time"}
    excluded = "analysed_st" not in model_input_names and "oi_data" not in model_input_names
    raw_median = float(np.median(medians))
    if units == "kelvin" and not 200.0 < raw_median < 330.0:
        raise RuntimeError(f"analysed_st median {raw_median:.3f} is inconsistent with Kelvin")
    if units == "celsius" and not -100.0 < raw_median < 100.0:
        raise RuntimeError(f"analysed_st median {raw_median:.3f} is inconsistent with Celsius")
    return OIVerification(
        units_requested=units,
        units_resolved=units,
        sample_dates=[date.isoformat() for date in sample_dates],
        min_value=min(minima),
        max_value=max(maxima),
        median_value=raw_median,
        time_matches_filename=all(time_matches),
        grid_matches_x1=all(grid_matches),
        excluded_from_model_inputs=excluded,
    )


def verification_payload(report: OIVerification) -> dict:
    return {"schema_version": 1, **asdict(report)}


def _canonical_temperature_units(units: str) -> str:
    normalized = units.strip().lower().replace(" ", "_").replace("°", "degree_")
    if normalized in {"k", "kelvin", "degree_kelvin", "degrees_kelvin"}:
        return "kelvin"
    if normalized in {
        "c",
        "degc",
        "celsius",
        "degree_celsius",
        "degrees_celsius",
    }:
        return "celsius"
    raise RuntimeError(f"Unrecognized analysed_st units attribute: {units!r}")


def compare_zarr_with_original_netcdf(
    zarr_path: str | Path,
    netcdf_path: str | Path,
    *,
    expected_units: str,
    expected_date: dt.date,
) -> dict:
    zarr_path = Path(zarr_path)
    netcdf_path = Path(netcdf_path)
    store = zarr.open(str(zarr_path), mode="r")
    with xr.open_dataset(netcdf_path) as dataset:
        if "analysed_st" not in dataset:
            raise RuntimeError(f"analysed_st missing from original NetCDF {netcdf_path}")
        original = np.asarray(dataset["analysed_st"].squeeze().values, dtype=np.float32)
        converted = np.asarray(store["analysed_st"][:], dtype=np.float32)
        if original.shape != converted.shape or not np.array_equal(original, converted, equal_nan=True):
            raise RuntimeError("analysed_st differs between the original NetCDF and x1 Zarr")
        lat_matches = np.array_equal(
            np.asarray(dataset["lat"].values, dtype=np.float32),
            np.asarray(store["lat"][:], dtype=np.float32),
        )
        lon_matches = np.array_equal(
            np.asarray(dataset["lon"].values, dtype=np.float32),
            np.asarray(store["lon"][:], dtype=np.float32),
        )
        if not lat_matches or not lon_matches:
            raise RuntimeError("DMI-OI grid differs between original NetCDF and x1 Zarr")
        units = str(dataset["analysed_st"].attrs.get("units", "")).strip()
        resolved_units = _canonical_temperature_units(units)
        if resolved_units != expected_units:
            raise RuntimeError(
                f"analysed_st units are {resolved_units}, expected {expected_units}"
            )
        valid_timestamp = pd.Timestamp(np.asarray(dataset["time"].values).reshape(-1)[0])
        expected_timestamp = pd.Timestamp(expected_date)
        if valid_timestamp != expected_timestamp:
            raise RuntimeError(
                f"analysed_st valid time is {valid_timestamp}, expected {expected_timestamp}"
            )
        valid_time = valid_timestamp.isoformat()
    return {
        "raw_archive_verified": True,
        "original_netcdf": str(netcdf_path.resolve()),
        "zarr": str(zarr_path.resolve()),
        "analysed_st_units_attribute": units,
        "analysed_st_units_resolved": resolved_units,
        "valid_time": valid_time,
        "valid_date_matches_archive": True,
        "valid_time_matches_archive": True,
        "values_exact_after_float32_conversion": True,
        "grid_exact": True,
    }
