#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from contrib.SST.evaluation.io import atomic_write_json, write_sha256_sidecar
from contrib.SST.evaluation.protocol import load_manifest


TILE_SIZE = 256
GRID_KM = 5.0
STRUCTURE_SCALES_PIXELS = np.array([1, 2, 4, 8, 16, 32])


def valid_tiles(target: np.ndarray, prediction: np.ndarray, surfmask: np.ndarray):
    for y0 in range(0, target.shape[0] - TILE_SIZE + 1, TILE_SIZE):
        for x0 in range(0, target.shape[1] - TILE_SIZE + 1, TILE_SIZE):
            target_tile = target[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
            prediction_tile = prediction[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE]
            ocean_tile = surfmask[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] != 0
            valid = ocean_tile & np.isfinite(target_tile) & np.isfinite(prediction_tile)
            if valid.mean() >= 0.90:
                yield target_tile, prediction_tile, valid


def radial_spectral_diagnostic(tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    window_1d = np.hanning(TILE_SIZE)
    window = window_1d[:, None] * window_1d[None, :]
    frequencies = np.fft.fftshift(np.fft.fftfreq(TILE_SIZE, d=GRID_KM))
    fx, fy = np.meshgrid(frequencies, frequencies)
    radial = np.sqrt(fx * fx + fy * fy)
    edges = np.linspace(0.0, radial.max(), 65)
    bin_index = np.digitize(radial.ravel(), edges) - 1
    ptt = np.zeros(64)
    ppp = np.zeros(64)
    cross = np.zeros(64, dtype=np.complex128)
    counts = np.zeros(64, dtype=np.int64)
    for target, prediction, valid in tiles:
        target_anomaly = np.where(valid, target - np.nanmean(target[valid]), 0.0) * window
        prediction_anomaly = np.where(valid, prediction - np.nanmean(prediction[valid]), 0.0) * window
        ft = np.fft.fftshift(np.fft.fft2(target_anomaly))
        fp = np.fft.fftshift(np.fft.fft2(prediction_anomaly))
        arrays = (np.abs(ft) ** 2, np.abs(fp) ** 2, ft * np.conj(fp))
        for bin_number in range(64):
            selected = bin_index == bin_number
            if not np.any(selected):
                continue
            ptt[bin_number] += arrays[0].ravel()[selected].sum()
            ppp[bin_number] += arrays[1].ravel()[selected].sum()
            cross[bin_number] += arrays[2].ravel()[selected].sum()
            counts[bin_number] += int(selected.sum())
    valid_bins = counts > 0
    ptt = np.divide(ptt, counts, out=np.full_like(ptt, np.nan), where=valid_bins)
    ppp = np.divide(ppp, counts, out=np.full_like(ppp, np.nan), where=valid_bins)
    cross = np.divide(cross, counts, out=np.full_like(cross, np.nan), where=valid_bins)
    coherence = np.divide(
        np.abs(cross) ** 2,
        ptt * ppp,
        out=np.full_like(ptt, np.nan),
        where=(ptt * ppp) > 0,
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "diagnostic": "radially_averaged_psd_and_coherence",
        "tile_count": len(tiles),
        "wavenumber_cycles_per_km": centers.tolist(),
        "target_psd": ptt.tolist(),
        "prediction_psd": ppp.tolist(),
        "coherence": coherence.tolist(),
    }


def structure_function(datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    sums = {"target": np.zeros(len(STRUCTURE_SCALES_PIXELS)), "prediction": np.zeros(len(STRUCTURE_SCALES_PIXELS))}
    counts = np.zeros(len(STRUCTURE_SCALES_PIXELS), dtype=np.int64)
    for target, prediction, valid in datasets:
        for scale_index, shift in enumerate(STRUCTURE_SCALES_PIXELS):
            target_differences = []
            prediction_differences = []
            valid_pairs = []
            valid_pairs.append(valid[:, shift:] & valid[:, :-shift])
            target_differences.append(target[:, shift:] - target[:, :-shift])
            prediction_differences.append(prediction[:, shift:] - prediction[:, :-shift])
            valid_pairs.append(valid[shift:, :] & valid[:-shift, :])
            target_differences.append(target[shift:, :] - target[:-shift, :])
            prediction_differences.append(prediction[shift:, :] - prediction[:-shift, :])
            for pair_mask, target_delta, prediction_delta in zip(valid_pairs, target_differences, prediction_differences):
                sums["target"][scale_index] += float(np.sum(target_delta[pair_mask] ** 2, dtype=np.float64))
                sums["prediction"][scale_index] += float(np.sum(prediction_delta[pair_mask] ** 2, dtype=np.float64))
                counts[scale_index] += int(pair_mask.sum())
    return {
        "diagnostic": "second_order_structure_function",
        "reason": "fewer_than_100_tiles_with_90_percent_valid_target",
        "scales_km": (STRUCTURE_SCALES_PIXELS * GRID_KM).tolist(),
        "target": np.divide(sums["target"], counts, out=np.full(len(counts), np.nan), where=counts > 0).tolist(),
        "prediction": np.divide(sums["prediction"], counts, out=np.full(len(counts), np.nan), where=counts > 0).tolist(),
        "valid_pairs": counts.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 2023 spatial-scale diagnostic")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--mode", default="controlled")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if len(manifest["records"]) != 24 or any(record["split"] != "pilot" for record in manifest["records"]):
        raise RuntimeError("Spatial diagnostic must use the frozen 24-date 2023 pilot")

    datasets = []
    tiles = []
    for record in manifest["records"]:
        path = Path(args.evaluation_root) / "daily" / args.mode / f"{record['central_date']}.nc"
        with xr.open_dataset(path) as dataset:
            target = dataset["target_sst"].values
            prediction = dataset["pred_sst_x1"].values
            surfmask = dataset["surfmask"].values
        valid = (surfmask != 0) & np.isfinite(target) & np.isfinite(prediction)
        datasets.append((target, prediction, valid))
        tiles.extend(valid_tiles(target, prediction, surfmask))
    result = radial_spectral_diagnostic(tiles) if len(tiles) >= 100 else structure_function(datasets)
    result.update({"schema_version": 1, "pilot_dates": 24, "valid_tile_count": len(tiles)})
    output = Path(args.output)
    atomic_write_json(output, result)
    write_sha256_sidecar(output)
    print(f"diagnostic={result['diagnostic']}")
    print(f"valid_tiles={len(tiles)}")


if __name__ == "__main__":
    main()
