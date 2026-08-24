from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088


def _unit_sphere(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(latitude)
    lon_rad = np.deg2rad(longitude)
    cos_lat = np.cos(lat_rad)
    return np.column_stack((cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)))


def build_coastal_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    surfmask: np.ndarray,
    *,
    threshold_km: float = 50.0,
    query_chunk_size: int = 1_000_000,
) -> np.ndarray:
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)
    land = np.asarray(surfmask) == 0
    ocean = ~land
    padded_ocean = np.pad(ocean, ((1, 1), (1, 1)), mode="constant")
    padded_ocean[:, 0] = padded_ocean[:, -2]
    padded_ocean[:, -1] = padded_ocean[:, 1]
    ocean_neighbours = binary_dilation(
        padded_ocean, structure=np.ones((3, 3), dtype=bool)
    )[1:-1, 1:-1]
    coastal_land = land & ocean_neighbours
    coast_y, coast_x = np.where(coastal_land)
    if not coast_y.size:
        raise RuntimeError("No coastline pixels found in surfmask")
    tree = cKDTree(_unit_sphere(latitude[coast_y], longitude[coast_x]))

    result = np.zeros_like(ocean)
    ocean_flat = np.flatnonzero(ocean)
    nlon = longitude.size
    chord_threshold = 2.0 * np.sin((threshold_km / EARTH_RADIUS_KM) / 2.0)
    for offset in range(0, ocean_flat.size, query_chunk_size):
        flat = ocean_flat[offset:offset + query_chunk_size]
        y = flat // nlon
        x = flat % nlon
        distance, _ = tree.query(_unit_sphere(latitude[y], longitude[x]), k=1, workers=-1)
        result.flat[flat] = distance <= chord_threshold
    return result


def save_coastal_mask(path: str | Path, mask: np.ndarray, *, threshold_km: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, coastal_mask=np.asarray(mask, dtype=np.uint8), threshold_km=threshold_km)
    temporary.replace(path)


def load_coastal_mask(path: str | Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["coastal_mask"], dtype=bool)
