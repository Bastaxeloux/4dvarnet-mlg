#!/usr/bin/env python3
"""Compute deterministic SST normalization statistics from daily x1 stores."""

import argparse
import calendar
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/nwp/sst_malegu"
DEFAULT_N_SAMPLES = 1500
DEFAULT_SEED = 20260821

VAR_GROUPS = {
    "aasti": {"av": "zscore", "std": "zscore"},
    "avhrr": {"av": "zscore", "std": "zscore"},
    "pmw": {"av": "zscore", "std": "zscore"},
    "slstr": {"av": "zscore", "std": "zscore"},
}
COVARIATES = {"sea_ice_fraction": "minmax"}
TEMPERATURE_AV_GROUPS = ("aasti", "avhrr", "pmw", "slstr")


def _complete_zarr_stores(directory, resolution):
    return sorted(
        path
        for path in directory.glob(f"*_{resolution}.zarr")
        if (path / ".zmetadata").is_file()
    )


def _new_accumulator():
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": np.inf,
        "max": -np.inf,
    }


def _update_accumulator(accumulator, values):
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        return

    accumulator["count"] += int(array.size)
    accumulator["sum"] += float(array.sum(dtype=np.float64))
    accumulator["sum_sq"] += float(np.square(array).sum(dtype=np.float64))
    accumulator["min"] = min(accumulator["min"], float(array.min()))
    accumulator["max"] = max(accumulator["max"], float(array.max()))


def _finalize(accumulator, norm_type, variable):
    if accumulator["count"] == 0:
        raise ValueError(f"No finite value found for required variable {variable!r}")

    if norm_type == "minmax":
        return {
            "min": float(accumulator["min"]),
            "max": float(accumulator["max"]),
            "type": "minmax",
        }

    mean = accumulator["sum"] / accumulator["count"]
    variance = accumulator["sum_sq"] / accumulator["count"] - mean**2
    return {
        "mean": float(mean),
        "std": float(np.sqrt(max(variance, 0.0))),
        "type": "zscore",
    }


def discover_x1_files(data_root, years=None, require_complete_years=True):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    counts_by_year = {}
    files = []
    if years:
        for year in years:
            year_dir = root / f"data_{year}"
            if not year_dir.is_dir():
                raise FileNotFoundError(f"Missing requested year directory: {year_dir}")
            year_files = _complete_zarr_stores(year_dir, "x1")
            counts_by_year[str(year)] = len(year_files)
            expected_days = 366 if calendar.isleap(year) else 365
            if require_complete_years and len(year_files) != expected_days:
                raise ValueError(
                    f"Year {year} has {len(year_files)} complete x1 stores; "
                    f"expected {expected_days}"
                )
            files.extend(year_files)
    else:
        year_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        for year_dir in year_dirs:
            year_files = _complete_zarr_stores(year_dir, "x1")
            if year_files:
                counts_by_year[year_dir.name] = len(year_files)
                files.extend(year_files)
        if not files:
            files = sorted(root.glob("*_x1.zarr"))
            files.extend(sorted(root.glob("*.nc")))
            counts_by_year["root"] = len(files)

    if not files:
        raise FileNotFoundError(f"No x1 Zarr or NetCDF file found below {root}")
    return root, files, counts_by_year


def sample_files(files, n_samples, seed):
    if n_samples <= 0 or n_samples >= len(files):
        return list(files)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(files), size=n_samples, replace=False))
    return [files[int(index)] for index in indices]


def compute_statistics(file_list):
    variable_types = {
        f"{satellite}_{variable}": norm_type
        for satellite, variables in VAR_GROUPS.items()
        for variable, norm_type in variables.items()
    }
    accumulated_variable_types = {
        variable: norm_type
        for variable, norm_type in variable_types.items()
        if not variable.endswith("_av")
    }
    required_variables = set(variable_types) | set(COVARIATES)
    accumulators = {
        variable: _new_accumulator()
        for variable in (
            set(accumulated_variable_types) | set(COVARIATES) | {"tgt_sst"}
        )
    }

    for path in tqdm(file_list, desc="Computing normalization", unit="file"):
        dataset = None
        try:
            dataset = xr.open_zarr(path) if str(path).endswith(".zarr") else xr.open_dataset(path)
            missing = sorted(required_variables.difference(dataset.variables))
            if missing:
                raise KeyError(f"missing variables: {', '.join(missing)}")

            for variable in accumulated_variable_types:
                _update_accumulator(accumulators[variable], dataset[variable].values)
            covariate_values = {}
            for variable in COVARIATES:
                values = dataset[variable].values
                covariate_values[variable] = values
                _update_accumulator(accumulators[variable], values)

            sea_ice = covariate_values["sea_ice_fraction"]
            target = np.where(
                sea_ice >= 0.70,
                dataset["aasti_av"].values,
                dataset["slstr_av"].values,
            )
            _update_accumulator(accumulators["tgt_sst"], target)
        except Exception as exc:
            raise RuntimeError(f"Failed while processing {path}: {exc}") from exc
        finally:
            if dataset is not None:
                dataset.close()

    raw_stats = {
        variable: _finalize(accumulators[variable], norm_type, variable)
        for variable, norm_type in accumulated_variable_types.items()
    }
    target_stats = _finalize(accumulators["tgt_sst"], "zscore", "tgt_sst")

    norm_stats = {}
    for satellite, variables in VAR_GROUPS.items():
        norm_stats[satellite] = {}
        for variable in variables:
            if variable == "av" and satellite in TEMPERATURE_AV_GROUPS:
                norm_stats[satellite][variable] = dict(target_stats)
            else:
                norm_stats[satellite][variable] = raw_stats[f"{satellite}_{variable}"]
    norm_stats["tgt_sst"] = dict(target_stats)
    norm_stats["sst_common"] = dict(target_stats)

    norm_stats_covs = {
        variable: _finalize(accumulators[variable], norm_type, variable)
        for variable, norm_type in COVARIATES.items()
    }
    return norm_stats, norm_stats_covs


def build_sample_manifest(root, available_counts, sampled_files, years, seed):
    records = []
    digest = hashlib.sha256()
    for path in sampled_files:
        metadata_path = path / ".zmetadata" if path.is_dir() else path
        stat = metadata_path.stat()
        relative_path = str(path.relative_to(root))
        record = {
            "path": relative_path,
            "metadata_size_bytes": stat.st_size,
            "metadata_mtime_ns": stat.st_mtime_ns,
        }
        records.append(record)
        digest.update(
            f"{relative_path}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8")
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "years": list(years) if years else None,
        "available_files_by_year": available_counts,
        "n_files_available": int(sum(available_counts.values())),
        "n_files_sampled": len(sampled_files),
        "sampling_seed": seed,
        "sample_inventory_sha256": digest.hexdigest(),
        "sampled_files": records,
    }


def _atomic_write_text(path, content):
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


def write_outputs(norm_stats, norm_stats_covs, manifest, output_yaml, output_txt, manifest_path):
    payload = {
        "metadata": {
            key: value for key, value in manifest.items() if key != "sampled_files"
        },
        "norm_stats": norm_stats,
        "norm_stats_covs": norm_stats_covs,
    }
    _atomic_write_text(output_yaml, yaml.safe_dump(payload, sort_keys=False))
    _atomic_write_text(
        output_txt,
        "norm_stats = "
        + repr(norm_stats)
        + "\n\nnorm_stats_covs = "
        + repr(norm_stats_covs)
        + "\n",
    )
    _atomic_write_text(manifest_path, yaml.safe_dump(manifest, sort_keys=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", nargs="?", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-complete-years",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-yaml", default="contrib/SST/norm_stats.yaml")
    parser.add_argument("--output-txt", default="contrib/SST/norm_stats.txt")
    parser.add_argument(
        "--manifest",
        default="contrib/SST/norm_stats_manifest.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root, available_files, counts = discover_x1_files(
        args.data_root,
        years=args.years,
        require_complete_years=args.require_complete_years,
    )
    selected_files = sample_files(available_files, args.n_samples, args.seed)

    print(f"Data root: {root}")
    print(f"Available x1 stores: {len(available_files)} ({counts})")
    print(
        f"Selected stores: {len(selected_files)} "
        f"(seed={args.seed}, requested={args.n_samples})"
    )

    norm_stats, norm_stats_covs = compute_statistics(selected_files)
    manifest = build_sample_manifest(root, counts, selected_files, args.years, args.seed)
    write_outputs(
        norm_stats,
        norm_stats_covs,
        manifest,
        args.output_yaml,
        args.output_txt,
        args.manifest,
    )

    common = norm_stats["sst_common"]
    print(
        "SST common normalization: "
        f"mean={common['mean']:.6f}, std={common['std']:.6f}"
    )
    print(f"YAML: {Path(args.output_yaml).expanduser().resolve()}")
    print(f"Manifest: {Path(args.manifest).expanduser().resolve()}")


if __name__ == "__main__":
    main()
