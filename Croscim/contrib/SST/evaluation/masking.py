from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .protocol import EvaluationRecord


SHIFT_PIXELS_PER_UNIT = {"x1": 30, "x3": 10, "x10": 3}


@dataclass(frozen=True)
class MaskBundle:
    masks: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]


def date_sequence(start: str | dt.date, end: str | dt.date) -> list[dt.date]:
    start_date = dt.date.fromisoformat(start) if isinstance(start, str) else start
    end_date = dt.date.fromisoformat(end) if isinstance(end, str) else end
    return [start_date + dt.timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def find_daily_store(data_root: str | Path, date: dt.date, resolution: str) -> Path:
    root = Path(data_root) / f"data_{date.year}"
    matches = sorted(root.glob(f"{date:%Y%m%d}*_{resolution}.zarr"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {resolution} store for {date.isoformat()} under {root}; found {matches}"
        )
    return matches[0]


def _fused_target_from_store(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import zarr

    store = zarr.open(str(path), mode="r")
    missing = [name for name in ("aasti_av", "slstr_av", "sea_ice_fraction") if name not in store]
    if missing:
        raise RuntimeError(f"Missing arrays {missing} in {path}")
    aasti = np.asarray(store["aasti_av"][:], dtype=np.float32)
    slstr = np.asarray(store["slstr_av"][:], dtype=np.float32)
    sea_ice = np.asarray(store["sea_ice_fraction"][:], dtype=np.float32)
    fused = np.where(sea_ice >= 0.70, aasti, slstr)
    return fused, sea_ice, np.asarray(store["surfmask"][:])


def _block_fraction(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask
    time, height, width = mask.shape
    if height % factor or width % factor:
        raise ValueError(f"Mask shape {mask.shape} is not divisible by factor {factor}")
    return mask.reshape(time, height // factor, factor, width // factor, factor).mean(axis=(2, 4))


def downsample_mask(mask_x1: np.ndarray, factor: int) -> np.ndarray:
    """Mask a coarse cell whenever its footprint contains a withheld x1 pixel."""
    return _block_fraction(np.asarray(mask_x1, dtype=np.float32), factor) > 0.0


def build_real_availability_mask(
    record: EvaluationRecord | Mapping[str, object],
    data_root: str | Path,
) -> MaskBundle:
    if not isinstance(record, EvaluationRecord):
        record = EvaluationRecord(**record)
    donor_dates = date_sequence(record.donor_context_start, record.donor_context_end)
    masks = []
    for donor_date in donor_dates:
        fused, _, surfmask = _fused_target_from_store(
            find_daily_store(data_root, donor_date, "x1")
        )
        masks.append(~np.isfinite(fused) & (surfmask != 0))
    mask_x1 = np.stack(masks, axis=0)
    shift_x1 = record.longitude_shift_units * SHIFT_PIXELS_PER_UNIT["x1"]
    mask_x1 = np.roll(mask_x1, shift=shift_x1, axis=-1)

    result = {
        "x1": mask_x1,
        "x3": downsample_mask(mask_x1, 3),
        "x10": downsample_mask(mask_x1, 10),
    }
    metadata = {
        "schema_version": 1,
        "family": "displaced_real_fused_availability",
        "mask_id": record.mask_id,
        "donor_context_start": record.donor_context_start,
        "donor_context_end": record.donor_context_end,
        "longitude_shift_units": record.longitude_shift_units,
        "longitude_shift_degrees": record.longitude_shift_degrees,
        "downsample_rule": "any_x1_withheld_pixel",
        "shapes": {key: list(value.shape) for key, value in result.items()},
        "missing_fraction": {key: float(value.mean()) for key, value in result.items()},
    }
    return MaskBundle(result, metadata)


def _rng_for(seed: int, token: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{token}".encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def build_training_like_mask(
    valid_target_x1: np.ndarray,
    *,
    seed: int,
    date_token: str,
    removal_fraction: float = 0.5,
    tile_size: int = 256,
) -> MaskBundle:
    """Build a deterministic global analogue of the training rectangle masks."""
    valid = np.asarray(valid_target_x1, dtype=bool)
    if valid.ndim != 3:
        raise ValueError(f"Expected (time, lat, lon), got {valid.shape}")
    removed = np.zeros_like(valid)
    _, height, width = valid.shape
    for time_index in range(valid.shape[0]):
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                tile_valid = valid[
                    time_index,
                    y0:min(y0 + tile_size, height),
                    x0:min(x0 + tile_size, width),
                ]
                initial_count = int(tile_valid.sum())
                if initial_count <= int(0.02 * tile_valid.size):
                    continue
                tile_removed = np.zeros_like(tile_valid)
                rng = _rng_for(seed, f"{date_token}:{time_index}:{y0}:{x0}")
                target_count = int(np.ceil(removal_fraction * initial_count))
                attempts = 0
                while int((tile_removed & tile_valid).sum()) < target_count and attempts < 4096:
                    half_h = int(rng.integers(2, 10))
                    half_w = int(rng.integers(2, 10))
                    yc = int(rng.integers(0, tile_valid.shape[0]))
                    xc = int(rng.integers(0, tile_valid.shape[1]))
                    tile_removed[
                        max(0, yc - half_h):min(tile_valid.shape[0], yc + half_h + 1),
                        max(0, xc - half_w):min(tile_valid.shape[1], xc + half_w + 1),
                    ] = True
                    attempts += 1
                removed[
                    time_index,
                    y0:min(y0 + tile_size, height),
                    x0:min(x0 + tile_size, width),
                ] = tile_removed & tile_valid

    result = {
        "x1": removed,
        "x3": downsample_mask(removed, 3),
        "x10": downsample_mask(removed, 10),
    }
    metadata = {
        "schema_version": 1,
        "family": "training_like_rectangles",
        "mask_id": hashlib.sha256(
            f"training-like-v1:{seed}:{date_token}".encode("ascii")
        ).hexdigest()[:16],
        "seed": seed,
        "date_token": date_token,
        "removal_fraction": removal_fraction,
        "tile_size": tile_size,
        "missing_fraction": {key: float(value.mean()) for key, value in result.items()},
    }
    return MaskBundle(result, metadata)
