from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import contrib.SST.models as sst_models
from contrib.SST.data import XrDataset, build_sst_postprocessor
from contrib.SST.load_data import COVARIATES, VAR_GROUPS

from .assembly import PatchAccumulator, patch_starts_payload, seam_mask_from_starts
from .coast import load_coastal_mask
from .io import atomic_write_bytes, atomic_write_json, sha256_file, write_sha256_sidecar
from .masking import (
    MaskBundle,
    build_real_availability_mask,
    build_training_like_mask,
    date_sequence,
    find_daily_store,
)
from .metrics import gradient_rmse, regime_masks, weighted_sufficient_stats
from .oi import oi_to_celsius
from .protocol import EvaluationRecord, load_manifest


MODEL_VARIABLES = [
    f"{group}_{variable}"
    for group, variables in VAR_GROUPS.items()
    for variable in variables
] + list(COVARIATES)


def _atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _evaluation_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible_devices = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    device_index = 0 if visible_devices == 1 else local_rank
    if device_index >= visible_devices:
        raise RuntimeError(
            f"SLURM_LOCALID={local_rank} cannot select among {visible_devices} visible GPUs"
        )
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index)


def _validate_completed_date(
    done_path: Path,
    *,
    date: str,
    mode: str,
    checkpoint_sha256: str,
    protocol_sha256: str | None,
) -> None:
    done = json.loads(done_path.read_text())
    expected = {
        "date": date,
        "mode": mode,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_protocol_sha256": protocol_sha256,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise RuntimeError(
                f"Existing completion marker {done_path} has {key}={done.get(key)!r}, "
                f"expected {value!r}"
            )
    for artifact_name in ("netcdf", "metrics"):
        artifact = Path(done[artifact_name]["path"])
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"Completed {date} is missing its {artifact_name}: {artifact}")


def _compose_runtime(project_root: Path, experiment: str):
    with initialize_config_dir(version_base=None, config_dir=str(project_root / "config")):
        cfg = compose(
            config_name="main",
            overrides=[f"xp={experiment}", "trainer.logger.version=publication_evaluation"],
        )
    OmegaConf.resolve(cfg)
    norm_stats = instantiate(cfg.datamodule.norm_stats)
    norm_stats_covs = instantiate(cfg.datamodule.norm_stats_covs)
    model = instantiate(cfg.model)
    return cfg, norm_stats, norm_stats_covs, model


def _resolved_config_sha256(cfg) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")).hexdigest()


def _normalization_path(cfg) -> Path:
    value = OmegaConf.select(cfg, "datamodule.norm_stats.path")
    if not value:
        raise RuntimeError("Resolved configuration has no datamodule.norm_stats.path")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Normalization file not found: {path}")
    return path


def load_publication_model(
    project_root: Path,
    experiment: str,
    checkpoint_path: Path,
    device: torch.device,
):
    sst_models.device = device
    cfg, norm_stats, norm_stats_covs, model = _compose_runtime(project_root, experiment)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return cfg, norm_stats, norm_stats_covs, model, checkpoint


def _move_batch(batch, device: torch.device):
    return type(batch)(
        **{
            name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for name, value in batch._asdict().items()
        }
    )


def interpolate_regular_global(
    field: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_latitude: np.ndarray,
    target_longitude: np.ndarray,
    device: torch.device,
    *,
    mode: str = "bilinear",
) -> np.ndarray:
    """Interpolate regular grids with bilinear border replication and no edge gaps."""
    field = np.asarray(field, dtype=np.float32)
    if field.ndim == 2:
        field = field[None, :, :]
    source_latitude = np.asarray(source_latitude, dtype=np.float32)
    source_longitude = np.asarray(source_longitude, dtype=np.float32)
    target_latitude = np.asarray(target_latitude, dtype=np.float32)
    target_longitude = np.asarray(target_longitude, dtype=np.float32)
    if source_latitude.size < 2 or source_longitude.size < 2:
        raise ValueError("Source grid must contain at least two points per dimension")

    source = torch.as_tensor(field, device=device).unsqueeze(0)
    target_lat = torch.as_tensor(target_latitude, device=device)
    target_lon = torch.as_tensor(target_longitude, device=device)
    norm_y = 2.0 * (target_lat - float(source_latitude[0])) / float(source_latitude[-1] - source_latitude[0]) - 1.0
    norm_x = 2.0 * (target_lon - float(source_longitude[0])) / float(source_longitude[-1] - source_longitude[0]) - 1.0
    grid = torch.empty(
        (1, target_latitude.size, target_longitude.size, 2),
        dtype=torch.float32,
        device=device,
    )
    grid[0, :, :, 0] = norm_x[None, :]
    grid[0, :, :, 1] = norm_y[:, None]
    grid.clamp_(-1.0, 1.0)
    with torch.no_grad():
        interpolated = F.grid_sample(
            source,
            grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )
    return interpolated.squeeze(0).cpu().numpy()


def _make_dataset(
    data_root: Path,
    context_dates: Sequence[dt.date],
    resolution: int,
    evaluation_mask: np.ndarray,
    postprocessor,
    *,
    min_spatial_overlap: int,
) -> XrDataset:
    paths = [find_daily_store(data_root, date, f"x{resolution}") for date in context_dates]
    times = np.array([np.datetime64(date.isoformat()) for date in context_dates])
    return XrDataset(
        sst_daily_paths=paths,
        tgt_vars=["slstr_av", "aasti_av"],
        mask=None,
        times=times,
        patch_dims={"time": 15, "lat": 256, "lon": 256},
        strides={"time": 15, "lat": 246, "lon": 246},
        postpro_fn=postprocessor,
        resize=resolution,
        res=5.0,
        pad=False,
        precomputed=True,
        enable_patch_filtering=False,
        cover_edges=True,
        min_spatial_overlap=min_spatial_overlap,
        evaluation_inpaint_mask=evaluation_mask,
        load_variable_names=MODEL_VARIABLES,
    )


def _raw_fields(path: Path, oi_units: str) -> dict[str, np.ndarray]:
    store = zarr.open(str(path), mode="r")
    aasti = np.asarray(store["aasti_av"][:], dtype=np.float32)
    slstr = np.asarray(store["slstr_av"][:], dtype=np.float32)
    sea_ice = np.asarray(store["sea_ice_fraction"][:], dtype=np.float32)
    target = np.where(sea_ice >= 0.70, aasti, slstr)
    return {
        "target": target,
        "sea_ice": sea_ice,
        "surfmask": np.asarray(store["surfmask"][:]),
        "dmi_oi": oi_to_celsius(np.asarray(store["analysed_st"][:], dtype=np.float32), oi_units),
        "latitude": np.asarray(store["lat"][:], dtype=np.float32),
        "longitude": np.asarray(store["lon"][:], dtype=np.float32),
    }


def _denormalize(values: np.ndarray, stats: Mapping[str, object]) -> np.ndarray:
    if stats["type"] == "zscore":
        return values * float(stats["std"]) + float(stats["mean"])
    if stats["type"] == "minmax":
        return values * (float(stats["max"]) - float(stats["min"])) + float(stats["min"])
    if stats["type"] is None:
        return values
    raise ValueError(f"Unsupported target normalization {stats['type']}")


def _mask_bundle(
    mode: str,
    record: EvaluationRecord,
    data_root: Path,
    context_dates: Sequence[dt.date],
) -> MaskBundle:
    if mode == "natural":
        masks = {}
        for resolution in (1, 3, 10):
            store = zarr.open(str(find_daily_store(data_root, context_dates[0], f"x{resolution}")), mode="r")
            masks[f"x{resolution}"] = np.zeros((15, store["lat"].shape[0], store["lon"].shape[0]), dtype=bool)
        return MaskBundle(masks, {"schema_version": 1, "family": "natural", "mask_id": "natural"})
    if mode == "controlled":
        return build_real_availability_mask(record, data_root)
    if mode == "rectangles":
        valid = []
        for date in context_dates:
            raw = _raw_fields(find_daily_store(data_root, date, "x1"), "celsius")
            valid.append(np.isfinite(raw["target"]) & (raw["surfmask"] != 0))
        return build_training_like_mask(
            np.stack(valid), seed=20260821, date_token=record.central_date
        )
    raise ValueError(f"Unknown evaluation mode {mode}")


def _coarse_patches(global_field: np.ndarray, dataset, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    patches = []
    for index in indices:
        slices = dataset._slices_from_flat_index(int(index))
        lat_slice = slices.get("lat", slices.get("yc"))
        lon_slice = slices.get("lon", slices.get("xc"))
        patches.append(global_field[:, lat_slice, lon_slice])
    return torch.as_tensor(np.stack(patches), dtype=torch.float32, device=device)


def _run_resolution(
    model,
    dataset: XrDataset,
    resolution: int,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    coarse_on_target_grid: np.ndarray | None,
) -> tuple[object, dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    n_timesteps = model.len_daw[resolution]
    accumulator = PatchAccumulator(dataset, n_timesteps, central_only=(resolution == 1))
    sample_offset = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        batch = _move_batch(batch, device)
        batch = model.modify_batch(batch, resolution)
        current_batch_size = int(batch.tgt_sst.shape[0])
        indices = list(range(sample_offset, sample_offset + current_batch_size))
        sample_offset += current_batch_size

        interpolated_coarse = None
        if coarse_on_target_grid is not None:
            interpolated_coarse = {
                "tgt_sst": _coarse_patches(coarse_on_target_grid, dataset, indices, device)
            }
            batch = model.update_batch_as_anomaly(batch, interpolated_coarse)

        solver_batch = model.format_batch_for_solver(batch)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast:
            prediction = model.split_tensor_to_dict(model(batch=solver_batch, res=resolution))["tgt_sst"]
            if interpolated_coarse is not None:
                prediction = prediction + interpolated_coarse["tgt_sst"]
        accumulator.add(prediction.detach().float().cpu().numpy(), indices)
        if batch_index % 25 == 0:
            print(
                f"[x{resolution}] batch {batch_index + 1}/{len(loader)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if sample_offset != len(dataset):
        raise RuntimeError(f"Processed {sample_offset} patches but dataset contains {len(dataset)}")
    assembled = accumulator.finalize()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    diagnostics = {
        "resolution": resolution,
        "n_patches": len(dataset),
        "n_uncovered_pixels": int(np.sum(assembled.geometric_coverage == 0)),
        "min_geometric_coverage": int(assembled.geometric_coverage.min()),
        "max_geometric_coverage": int(assembled.geometric_coverage.max()),
        "n_nonfinite_assembled": int(np.sum(~np.isfinite(assembled.mean))),
        "overlap_std_mean_normalized": float(np.nanmean(assembled.overlap_std_central)),
        "overlap_std_p95_normalized": float(np.nanpercentile(assembled.overlap_std_central, 95)),
        "patch_starts": patch_starts_payload(dataset),
        "elapsed_seconds": elapsed_seconds,
    }
    return assembled, diagnostics


def _metric_rows(
    date: str,
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    dmi_oi: np.ndarray,
    hidden: np.ndarray,
    visible: np.ndarray,
    surfmask: np.ndarray,
    sea_ice: np.ndarray,
    coastal_mask: np.ndarray,
    latitude: np.ndarray,
    mask_id: str,
) -> list[dict]:
    rows = []
    regimes = regime_masks(surfmask, sea_ice, coastal_mask)
    methods = {**predictions, "dmi_oi": dmi_oi}
    for method, prediction in methods.items():
        for support_name, support in (("hidden", hidden), ("visible", visible)):
            for regime_name, regime in regimes.items():
                selected_support = support & regime
                stats = weighted_sufficient_stats(target, prediction, selected_support, latitude)
                eligible = (hidden | visible) & regime
                rows.append({
                    "date": date,
                    "method": method,
                    "support": support_name,
                    "regime": regime_name,
                    "mask_id": mask_id,
                    "missing_fraction": float((hidden & regime).sum() / max(int(eligible.sum()), 1)),
                    "n_dates": 1,
                    **stats.as_dict(),
                })
    return rows


def _write_daily_netcdf(path: Path, payload: Mapping[str, np.ndarray], attrs: Mapping[str, object]) -> None:
    latitude = payload["latitude"]
    longitude = payload["longitude"]
    data_vars = {}
    for name, values in payload.items():
        if name in {"latitude", "longitude"}:
            continue
        data_vars[name] = (("lat", "lon"), values)
    dataset = xr.Dataset(data_vars=data_vars, coords={"lat": latitude, "lon": longitude}, attrs=dict(attrs))
    encoding = {}
    for name, variable in dataset.data_vars.items():
        encoding[name] = {
            "zlib": True,
            "complevel": 2,
            "shuffle": True,
            "chunksizes": (min(256, variable.shape[0]), min(256, variable.shape[1])),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
    os.replace(temporary, path)


def evaluate_date(
    record: EvaluationRecord,
    *,
    model,
    norm_stats: Mapping[str, object],
    norm_stats_covs: Mapping[str, object],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    data_root: Path,
    output_root: Path,
    coastal_mask: np.ndarray,
    mode: str,
    oi_units: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    min_spatial_overlap: int,
    protocol_sha256: str | None,
) -> None:
    date = record.central_date
    done_path = output_root / "done" / mode / f"{date}.done.json"
    if done_path.exists():
        _validate_completed_date(
            done_path,
            date=date,
            mode=mode,
            checkpoint_sha256=checkpoint_sha256,
            protocol_sha256=protocol_sha256,
        )
        print(f"[SKIP] {date} already complete", flush=True)
        return
    lock_path = output_root / "locks" / mode / f"{date}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise RuntimeError(f"Date is already locked by another evaluator: {date}") from exc
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(f"host={socket.gethostname()} pid={os.getpid()} date={date}\n")
    lock_stream.flush()

    try:
        started_at = dt.datetime.now(dt.timezone.utc)
        date_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        context_dates = date_sequence(record.context_start, record.context_end)
        mask_bundle = _mask_bundle(mode, record, data_root, context_dates)
        postprocessor = build_sst_postprocessor(norm_stats, norm_stats_covs, rand_obs=False)
        datasets = {
            resolution: _make_dataset(
                data_root,
                context_dates,
                resolution,
                mask_bundle.masks[f"x{resolution}"],
                postprocessor,
                min_spatial_overlap=min_spatial_overlap,
            )
            for resolution in (10, 3, 1)
        }
        setup_seconds = time.perf_counter() - date_started

        assembled = {}
        diagnostics = {}
        coarse_target_grid = None
        inference_started = time.perf_counter()
        for resolution in (10, 3, 1):
            result, resolution_diagnostics = _run_resolution(
                model,
                datasets[resolution],
                resolution,
                device,
                batch_size=batch_size,
                num_workers=num_workers,
                coarse_on_target_grid=coarse_target_grid,
            )
            assembled[resolution] = result
            diagnostics[f"x{resolution}"] = resolution_diagnostics
            if resolution != 1:
                next_resolution = 3 if resolution == 10 else 1
                next_length = model.len_daw[next_resolution]
                start = (model.len_daw[resolution] - next_length) // 2
                cropped = result.mean[start:start + next_length]
                coarse_target_grid = interpolate_regular_global(
                    cropped,
                    datasets[resolution].lat_1d,
                    datasets[resolution].lon_1d,
                    datasets[next_resolution].lat_1d,
                    datasets[next_resolution].lon_1d,
                    device,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_pipeline_seconds = time.perf_counter() - inference_started

        postprocessing_started = time.perf_counter()
        central_date = dt.date.fromisoformat(date)
        raw = _raw_fields(find_daily_store(data_root, central_date, "x1"), oi_units)
        target_stats = norm_stats["tgt_sst"]
        if target_stats["type"] == "zscore":
            target_scale = float(target_stats["std"])
        elif target_stats["type"] == "minmax":
            target_scale = float(target_stats["max"]) - float(target_stats["min"])
        else:
            target_scale = 1.0
        for resolution_diagnostics in diagnostics.values():
            resolution_diagnostics["overlap_std_mean_c"] = (
                resolution_diagnostics["overlap_std_mean_normalized"] * target_scale
            )
            resolution_diagnostics["overlap_std_p95_c"] = (
                resolution_diagnostics["overlap_std_p95_normalized"] * target_scale
            )
        prediction_x1 = _denormalize(assembled[1].mean[0], target_stats)
        prediction_x3 = _denormalize(
            interpolate_regular_global(
                assembled[3].mean[model.len_daw[3] // 2],
                datasets[3].lat_1d,
                datasets[3].lon_1d,
                raw["latitude"],
                raw["longitude"],
                device,
            )[0],
            target_stats,
        )
        prediction_x10 = _denormalize(
            interpolate_regular_global(
                assembled[10].mean[model.len_daw[10] // 2],
                datasets[10].lat_1d,
                datasets[10].lon_1d,
                raw["latitude"],
                raw["longitude"],
                device,
            )[0],
            target_stats,
        )
        hidden = (
            mask_bundle.masks["x1"][7]
            & np.isfinite(raw["target"])
            & (raw["surfmask"] != 0)
        )
        visible = np.isfinite(raw["target"]) & (raw["surfmask"] != 0) & ~hidden
        observed = np.where(hidden, np.nan, raw["target"])
        prediction_payload = {
            "croscim_x10": prediction_x10,
            "croscim_x3": prediction_x3,
            "croscim_x1": prediction_x1,
        }

        metrics_rows = []
        if mode != "natural":
            metrics_rows = _metric_rows(
                date,
                prediction_payload,
                raw["target"],
                raw["dmi_oi"],
                hidden,
                visible,
                raw["surfmask"],
                raw["sea_ice"],
                coastal_mask,
                raw["latitude"],
                str(mask_bundle.metadata["mask_id"]),
            )
        gradient = {
            method: {
                support: gradient_rmse(
                    raw["target"], prediction, hidden if support == "hidden" else visible,
                    raw["latitude"], raw["longitude"],
                )
                for support in ("hidden", "visible")
            }
            for method, prediction in {**prediction_payload, "dmi_oi": raw["dmi_oi"]}.items()
        } if mode != "natural" else {}

        seam_diagnostics = {}
        seam_margin_pixels = 5
        for resolution in (10, 3, 1):
            dataset = datasets[resolution]
            seam = seam_mask_from_starts(
                (len(dataset.lat_1d), len(dataset.lon_1d)),
                list(dataset.patch_starts["lat"]),
                list(dataset.patch_starts["lon"]),
                256,
                margin=seam_margin_pixels,
            )
            native_raw = _raw_fields(find_daily_store(data_root, central_date, f"x{resolution}"), oi_units)
            native_prediction = _denormalize(
                assembled[resolution].mean[0 if resolution == 1 else model.len_daw[resolution] // 2],
                target_stats,
            )
            native_valid = np.isfinite(native_raw["target"]) & (native_raw["surfmask"] != 0)
            native_hidden = mask_bundle.masks[f"x{resolution}"][7] & native_valid
            seam_diagnostics[f"x{resolution}"] = {
                "margin_pixels": seam_margin_pixels,
                "seam": weighted_sufficient_stats(
                    native_raw["target"], native_prediction, native_hidden & seam, native_raw["latitude"]
                ).metrics(),
                "interior": weighted_sufficient_stats(
                    native_raw["target"], native_prediction, native_hidden & ~seam, native_raw["latitude"]
                ).metrics(),
            }

        netcdf_path = output_root / "daily" / mode / f"{date}.nc"
        metrics_path = output_root / "metrics_daily" / mode / f"{date}.json"
        netcdf_payload = {
            "latitude": raw["latitude"],
            "longitude": raw["longitude"],
            "target_sst": raw["target"].astype(np.float32),
            "observed_sst": observed.astype(np.float32),
            "pred_sst_x10": prediction_x10.astype(np.float32),
            "pred_sst_x3": prediction_x3.astype(np.float32),
            "pred_sst_x1": prediction_x1.astype(np.float32),
            "hidden_mask": hidden.astype(np.uint8),
            "visible_mask": visible.astype(np.uint8),
            "original_valid_mask": (hidden | visible).astype(np.uint8),
            "surfmask": raw["surfmask"].astype(np.int8),
            "sea_ice_fraction": raw["sea_ice"].astype(np.float32),
            "dmi_oi": raw["dmi_oi"].astype(np.float32),
            "patch_coverage_x1": assembled[1].geometric_coverage.astype(np.uint16),
            "patch_disagreement_x1": _denormalize(
                assembled[1].overlap_std_central, {**target_stats, "mean": 0.0}
            ).astype(np.float32),
        }
        attrs = {
            "central_date": date,
            "evaluation_mode": mode,
            "checkpoint_sha256": checkpoint_sha256,
            "frozen_protocol_sha256": protocol_sha256 or "none",
            "mask_id": mask_bundle.metadata["mask_id"],
            "temperature_units": "degree_Celsius",
            "dmi_oi_comparison": "operational_reference_not_rerun_after_withholding",
            "patch_assembly": "equal_weight_mean",
            "patch_size_pixels": 256,
            "minimum_patch_overlap_pixels": min_spatial_overlap,
            "coarse_to_fine_interpolation": "bilinear_align_corners_true_border_replication",
        }
        postprocessing_seconds = time.perf_counter() - postprocessing_started
        netcdf_started = time.perf_counter()
        _write_daily_netcdf(netcdf_path, netcdf_payload, attrs)
        netcdf_write_seconds = time.perf_counter() - netcdf_started
        finished_at = dt.datetime.now(dt.timezone.utc)
        runtime = {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "n_patches_total": sum(item["n_patches"] for item in diagnostics.values()),
            "setup_seconds": setup_seconds,
            "resolution_seconds": {
                stage: item["elapsed_seconds"] for stage, item in diagnostics.items()
            },
            "inference_pipeline_seconds": inference_pipeline_seconds,
            "postprocessing_seconds": postprocessing_seconds,
            "netcdf_write_seconds": netcdf_write_seconds,
            "date_processing_through_netcdf_seconds": time.perf_counter() - date_started,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            ),
            "peak_cuda_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            ),
        }
        metrics_payload = {
            "schema_version": 1,
            "record": asdict(record),
            "mode": mode,
            "mask": dict(mask_bundle.metadata),
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
            "frozen_protocol_sha256": protocol_sha256,
            "assembly": diagnostics,
            "seams": seam_diagnostics,
            "gradient_rmse_c_per_km": gradient,
            "sufficient_statistics": metrics_rows,
            "runtime": runtime,
        }
        atomic_write_json(metrics_path, metrics_payload)
        done_payload = {
            "schema_version": 1,
            "date": date,
            "mode": mode,
            "netcdf": {"path": str(netcdf_path), "sha256": sha256_file(netcdf_path)},
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "checkpoint_sha256": checkpoint_sha256,
            "frozen_protocol_sha256": protocol_sha256,
        }
        atomic_write_json(done_path, done_payload)
        print(
            f"[TIMING] {date} total={runtime['date_processing_through_netcdf_seconds']:.1f}s "
            f"inference={runtime['inference_pipeline_seconds']:.1f}s "
            f"x10={runtime['resolution_seconds']['x10']:.1f}s "
            f"x3={runtime['resolution_seconds']['x3']:.1f}s "
            f"x1={runtime['resolution_seconds']['x1']:.1f}s",
            flush=True,
        )
        print(f"[DONE] {date} -> {netcdf_path}", flush=True)
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()
        lock_path.unlink(missing_ok=True)


def _write_run_provenance(
    output_root: Path,
    cfg,
    manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_data: Mapping[str, object],
    mode: str,
    coastal_mask_path: Path,
    oi_verification_path: Path,
    frozen_protocol_path: Path | None,
    shard_index: int,
    num_shards: int,
) -> None:
    provenance_dir = output_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = provenance_dir / "resolved_config.yaml"
    resolved_config_text = OmegaConf.to_yaml(cfg, resolve=True)
    if resolved_config.exists():
        if resolved_config.read_text() != resolved_config_text:
            raise RuntimeError("Evaluation output already contains a different resolved configuration")
    else:
        _atomic_write_text(resolved_config, resolved_config_text)
        write_sha256_sidecar(resolved_config)
    archived_checkpoint = provenance_dir / "publication_best.ckpt"
    if not archived_checkpoint.exists():
        try:
            os.link(checkpoint_path, archived_checkpoint)
        except FileExistsError:
            pass
    if sha256_file(archived_checkpoint) != sha256_file(checkpoint_path):
        raise RuntimeError("Archived evaluation checkpoint hash mismatch")
    write_sha256_sidecar(archived_checkpoint)
    package_versions = {}
    for package in ("pytorch-lightning", "hydra-core", "omegaconf", "pandas", "scipy", "zarr", "netCDF4"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    normalization_path = _normalization_path(cfg)
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "packages": package_versions,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint_data.get("epoch", -1)),
            "global_step": int(checkpoint_data.get("global_step", -1)),
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "resolved_config": {"path": str(resolved_config), "sha256": sha256_file(resolved_config)},
        "normalization": {
            "path": str(normalization_path),
            "sha256": sha256_file(normalization_path),
        },
        "coastal_mask": {"path": str(coastal_mask_path), "sha256": sha256_file(coastal_mask_path)},
        "dmi_oi_verification": {
            "path": str(oi_verification_path),
            "sha256": sha256_file(oi_verification_path),
        },
        "frozen_protocol": (
            {"path": str(frozen_protocol_path), "sha256": sha256_file(frozen_protocol_path)}
            if frozen_protocol_path is not None
            else None
        ),
        "mode": mode,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    path = provenance_dir / f"runtime_{mode}_rank_{shard_index:02d}.json"
    if not path.exists():
        atomic_write_json(path, payload)
        write_sha256_sidecar(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming Croscim publication evaluator")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--experiment", default="SST/multires_jeanzay_resunet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--coastal-mask", required=True)
    parser.add_argument("--oi-verification", required=True)
    parser.add_argument("--frozen-protocol")
    parser.add_argument("--mode", choices=("controlled", "natural", "rectangles"), required=True)
    parser.add_argument("--dmi-oi-units", choices=("kelvin", "celsius"), required=True)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-spatial-overlap", type=int, default=10)
    parser.add_argument("--limit-dates", type=int)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    project_root = Path(args.project_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    data_root = Path(args.data_root).resolve()
    coastal_mask_path = Path(args.coastal_mask).resolve()
    oi_verification_path = Path(args.oi_verification).resolve()
    frozen_protocol_path = Path(args.frozen_protocol).resolve() if args.frozen_protocol else None
    device = _evaluation_device()

    manifest = load_manifest(manifest_path)
    records = [EvaluationRecord(**record) for record in manifest["records"]]
    record_splits = {record.split for record in records}
    if len(record_splits) != 1:
        raise RuntimeError(f"Evaluation manifest mixes splits: {sorted(record_splits)}")
    manifest_sha256 = sha256_file(manifest_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    oi_verification = json.loads(oi_verification_path.read_text())
    if oi_verification.get("units_resolved") != args.dmi_oi_units:
        raise RuntimeError("DMI-OI units disagree with the verified protocol")
    protocol_sha256 = None
    if frozen_protocol_path is not None:
        frozen_protocol = json.loads(frozen_protocol_path.read_text())
        protocol_sha256 = sha256_file(frozen_protocol_path)
        if frozen_protocol.get("status") != "frozen_after_2023_pilot_before_2024_test":
            raise RuntimeError("Invalid frozen protocol status")
        if frozen_protocol["checkpoint"]["sha256"] != checkpoint_sha256:
            raise RuntimeError("Checkpoint does not match the frozen protocol")
        static_artifacts = frozen_protocol.get("static_artifacts", {})
        for name, path in (
            ("coastal_mask", coastal_mask_path),
            ("dmi_oi_verification", oi_verification_path),
        ):
            if static_artifacts.get(name, {}).get("sha256") != sha256_file(path):
                raise RuntimeError(f"{name} does not match the frozen protocol")
        for relative_path, expected_hash in frozen_protocol.get("evaluation_sources", {}).items():
            source_path = project_root / relative_path
            if not source_path.is_file() or sha256_file(source_path) != expected_hash:
                raise RuntimeError(f"Evaluation source changed after protocol freeze: {relative_path}")
        frozen_manifests = frozen_protocol.get("manifests", {})
        if "test" in record_splits:
            expected_manifest_hash = frozen_manifests.get("test", {}).get("sha256")
        else:
            expected_manifest_hash = manifest_sha256 if manifest_sha256 in {
                entry["sha256"] for entry in frozen_manifests.values()
            } else None
        if manifest_sha256 != expected_manifest_hash:
            raise RuntimeError("Evaluation manifest is not part of the frozen protocol")
    if any(record.split == "test" for record in records) and frozen_protocol_path is None:
        raise RuntimeError("The 2024 test split requires --frozen-protocol")
    if args.limit_dates is not None:
        records = records[:args.limit_dates]
    records = records[args.shard_index::args.num_shards]
    if not records:
        print(f"Shard {args.shard_index} has no dates")
        return

    cfg, norm_stats, norm_stats_covs, model, checkpoint_data = load_publication_model(
        project_root, args.experiment, checkpoint_path, device
    )
    if frozen_protocol_path is not None:
        expected_normalization = frozen_protocol["static_artifacts"]["normalization"]
        normalization_path = _normalization_path(cfg)
        if sha256_file(normalization_path) != expected_normalization["sha256"]:
            raise RuntimeError("Normalization statistics do not match the frozen protocol")
        expected_config_hash = frozen_protocol["runtime_contract"]["resolved_config_sha256"]
        if _resolved_config_sha256(cfg) != expected_config_hash:
            raise RuntimeError("Resolved evaluation configuration changed after protocol freeze")
    _write_run_provenance(
        output_root,
        cfg,
        manifest_path,
        checkpoint_path,
        checkpoint_data,
        args.mode,
        coastal_mask_path,
        oi_verification_path,
        frozen_protocol_path,
        args.shard_index,
        args.num_shards,
    )
    coastal_mask = load_coastal_mask(coastal_mask_path)
    for record in records:
        evaluate_date(
            record,
            model=model,
            norm_stats=norm_stats,
            norm_stats_covs=norm_stats_covs,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            data_root=data_root,
            output_root=output_root,
            coastal_mask=coastal_mask,
            mode=args.mode,
            oi_units=args.dmi_oi_units,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            min_spatial_overlap=args.min_spatial_overlap,
            protocol_sha256=protocol_sha256,
        )


if __name__ == "__main__":
    main()
