from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class WeightedSufficientStats:
    n_pixels: int = 0
    sum_w: float = 0.0
    sum_w_target: float = 0.0
    sum_w_target_sq: float = 0.0
    sum_w_prediction: float = 0.0
    sum_w_prediction_sq: float = 0.0
    sum_w_cross: float = 0.0
    sum_w_error: float = 0.0
    sum_w_abs_error: float = 0.0
    sum_w_sq_error: float = 0.0

    def merge(self, other: "WeightedSufficientStats") -> "WeightedSufficientStats":
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))
        return self

    def metrics(self) -> dict[str, float | int]:
        if self.sum_w <= 0:
            return {
                "n_pixels": self.n_pixels,
                "sum_w": self.sum_w,
                "rmse_c": np.nan,
                "mae_c": np.nan,
                "bias_c": np.nan,
                "correlation": np.nan,
                "target_std_c": np.nan,
                "nrmse": np.nan,
            }
        target_mean = self.sum_w_target / self.sum_w
        prediction_mean = self.sum_w_prediction / self.sum_w
        target_var = max(self.sum_w_target_sq / self.sum_w - target_mean**2, 0.0)
        prediction_var = max(self.sum_w_prediction_sq / self.sum_w - prediction_mean**2, 0.0)
        covariance = self.sum_w_cross / self.sum_w - target_mean * prediction_mean
        denominator = np.sqrt(target_var * prediction_var)
        rmse = np.sqrt(self.sum_w_sq_error / self.sum_w)
        target_std = np.sqrt(target_var)
        return {
            "n_pixels": self.n_pixels,
            "sum_w": self.sum_w,
            "rmse_c": float(rmse),
            "mae_c": float(self.sum_w_abs_error / self.sum_w),
            "bias_c": float(self.sum_w_error / self.sum_w),
            "correlation": float(covariance / denominator) if denominator > 0 else np.nan,
            "target_std_c": float(target_std),
            "nrmse": float(rmse / target_std) if target_std > 0 else np.nan,
        }

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "WeightedSufficientStats":
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


def latitude_weights(latitude: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    latitude = np.asarray(latitude)
    if latitude.ndim == 1:
        weights = np.cos(np.deg2rad(latitude))[:, None]
    elif latitude.ndim == 2:
        weights = np.cos(np.deg2rad(latitude))
    else:
        raise ValueError(f"Latitude must be 1D or 2D, got {latitude.shape}")
    return np.broadcast_to(weights, target_shape)


def weighted_sufficient_stats(
    target: np.ndarray,
    prediction: np.ndarray,
    support: np.ndarray,
    latitude: np.ndarray,
) -> WeightedSufficientStats:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if target.shape != prediction.shape or target.shape != support.shape:
        raise ValueError(f"Shape mismatch: target={target.shape}, prediction={prediction.shape}, support={support.shape}")
    weights = latitude_weights(latitude, target.shape)
    valid = support & np.isfinite(target) & np.isfinite(prediction) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return WeightedSufficientStats()
    y = target[valid]
    p = prediction[valid]
    w = weights[valid]
    error = p - y
    return WeightedSufficientStats(
        n_pixels=int(valid.sum()),
        sum_w=float(w.sum()),
        sum_w_target=float(np.dot(w, y)),
        sum_w_target_sq=float(np.dot(w, y * y)),
        sum_w_prediction=float(np.dot(w, p)),
        sum_w_prediction_sq=float(np.dot(w, p * p)),
        sum_w_cross=float(np.dot(w, y * p)),
        sum_w_error=float(np.dot(w, error)),
        sum_w_abs_error=float(np.dot(w, np.abs(error))),
        sum_w_sq_error=float(np.dot(w, error * error)),
    )


def merge_stats(rows: Iterable[WeightedSufficientStats]) -> WeightedSufficientStats:
    merged = WeightedSufficientStats()
    for row in rows:
        merged.merge(row)
    return merged


def regime_masks(
    surfmask: np.ndarray,
    sea_ice_fraction: np.ndarray,
    coastal_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    ocean = np.asarray(surfmask) != 0
    high_ice = ocean & (np.asarray(sea_ice_fraction) >= 0.70)
    coast = ocean & ~high_ice & np.asarray(coastal_mask, dtype=bool)
    open_ocean = ocean & ~high_ice & ~np.asarray(coastal_mask, dtype=bool)
    return {
        "global": ocean,
        "high_ice": high_ice,
        "coastal": coast,
        "open_ocean": open_ocean,
    }


def gradient_rmse(
    target: np.ndarray,
    prediction: np.ndarray,
    support: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if target.ndim != 2:
        raise ValueError("gradient_rmse expects 2D central-day fields")

    lat_step_km = 111.195 * np.abs(np.diff(latitude))[:, None]
    lon_delta = np.abs(np.diff(longitude))
    lon_step_km = 111.195 * np.cos(np.deg2rad(latitude))[:, None] * lon_delta[None, :]
    error = prediction - target

    valid_y = support[1:, :] & support[:-1, :] & np.isfinite(error[1:, :]) & np.isfinite(error[:-1, :])
    valid_x = support[:, 1:] & support[:, :-1] & np.isfinite(error[:, 1:]) & np.isfinite(error[:, :-1])
    grad_y = np.divide(np.diff(error, axis=0), lat_step_km, out=np.full_like(error[1:, :], np.nan), where=lat_step_km > 0)
    grad_x = np.divide(np.diff(error, axis=1), lon_step_km, out=np.full_like(error[:, 1:], np.nan), where=lon_step_km > 0)
    values = np.concatenate((grad_y[valid_y], grad_x[valid_x]))
    return float(np.sqrt(np.mean(values * values))) if values.size else np.nan


def circular_block_bootstrap_indices(
    n_days: int,
    *,
    block_days: int = 30,
    n_bootstrap: int = 2000,
    seed: int = 20260821,
) -> np.ndarray:
    if n_days <= 0 or block_days <= 0:
        raise ValueError("n_days and block_days must be positive")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_days))
    starts = rng.integers(0, n_days, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_days)
    samples = (starts[..., None] + offsets) % n_days
    return samples.reshape(n_bootstrap, -1)[:, :n_days]
