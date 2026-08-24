from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AssemblyResult:
    mean: np.ndarray
    geometric_coverage: np.ndarray
    finite_coverage: np.ndarray
    overlap_std_central: np.ndarray


class PatchAccumulator:
    def __init__(self, dataset, n_timesteps: int, *, central_only: bool = False):
        self.dataset = dataset
        self.n_timesteps = n_timesteps
        self.central_only = central_only
        nlat = int(dataset.da_dims["lat"])
        nlon = int(dataset.da_dims["lon"])
        stored_timesteps = 1 if central_only else n_timesteps
        self.sum = np.zeros((stored_timesteps, nlat, nlon), dtype=np.float32)
        self.finite_count = np.zeros((stored_timesteps, nlat, nlon), dtype=np.uint16)
        self.geometric_count = np.zeros((nlat, nlon), dtype=np.uint16)
        self.central_sum_sq = np.zeros((nlat, nlon), dtype=np.float32)

    def add(self, predictions: np.ndarray, dataset_indices: Sequence[int]) -> None:
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.ndim != 4 or predictions.shape[0] != len(dataset_indices):
            raise ValueError(f"Unexpected prediction batch shape {predictions.shape}")
        if predictions.shape[1] != self.n_timesteps:
            raise ValueError(f"Expected {self.n_timesteps} timesteps, got {predictions.shape[1]}")

        central_index = self.n_timesteps // 2
        for prediction, dataset_index in zip(predictions, dataset_indices):
            slices = self.dataset._slices_from_flat_index(int(dataset_index))
            lat_slice = slices.get("lat", slices.get("yc"))
            lon_slice = slices.get("lon", slices.get("xc"))
            expected_shape = (lat_slice.stop - lat_slice.start, lon_slice.stop - lon_slice.start)
            if tuple(prediction.shape[-2:]) != expected_shape:
                raise ValueError(f"Patch shape {prediction.shape[-2:]} does not match {expected_shape}")

            self.geometric_count[lat_slice, lon_slice] += 1
            selected = prediction[central_index:central_index + 1] if self.central_only else prediction
            finite = np.isfinite(selected)
            self.sum[:, lat_slice, lon_slice] += np.where(finite, selected, 0.0)
            self.finite_count[:, lat_slice, lon_slice] += finite.astype(np.uint16)

            central = prediction[central_index]
            self.central_sum_sq[lat_slice, lon_slice] += np.where(
                np.isfinite(central), central * central, 0.0
            )

    def finalize(self) -> AssemblyResult:
        if np.any(self.geometric_count == 0):
            raise RuntimeError(
                f"Incomplete patch geometry: {int(np.sum(self.geometric_count == 0))} uncovered pixels"
            )
        mean = np.divide(
            self.sum,
            self.finite_count,
            out=np.full_like(self.sum, np.nan),
            where=self.finite_count > 0,
        )
        central_position = 0 if self.central_only else self.n_timesteps // 2
        central_count = self.finite_count[central_position]
        central_mean = mean[central_position]
        variance = np.divide(
            self.central_sum_sq,
            central_count,
            out=np.full_like(self.central_sum_sq, np.nan),
            where=central_count > 0,
        ) - central_mean * central_mean
        overlap_std = np.sqrt(np.maximum(variance, 0.0), dtype=np.float32)
        return AssemblyResult(mean, self.geometric_count, self.finite_count, overlap_std)


def seam_mask_from_starts(
    shape: tuple[int, int],
    lat_starts: Sequence[int],
    lon_starts: Sequence[int],
    patch_size: int,
    margin: int = 5,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    boundaries_y = set(lat_starts[1:]) | {start + patch_size for start in lat_starts[:-1]}
    boundaries_x = set(lon_starts[1:]) | {start + patch_size for start in lon_starts[:-1]}
    for boundary in boundaries_y:
        mask[max(0, boundary - margin):min(shape[0], boundary + margin + 1), :] = True
    for boundary in boundaries_x:
        mask[:, max(0, boundary - margin):min(shape[1], boundary + margin + 1)] = True
    return mask


def patch_starts_payload(dataset) -> dict[str, list[int]]:
    return {
        dim: [int(value) for value in starts]
        for dim, starts in dataset.patch_starts.items()
    }
