from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .io import atomic_write_json, sha256_payload, write_sha256_sidecar


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260821
CONTEXT_RADIUS_DAYS = 7
DONOR_YEARS = tuple(range(2017, 2023))
COMMON_LONGITUDE_STEP_DEGREES = 1.5


@dataclass(frozen=True)
class EvaluationRecord:
    split: str
    index: int
    central_date: str
    context_start: str
    context_end: str
    donor_year: int
    donor_central_date: str
    donor_context_start: str
    donor_context_end: str
    longitude_shift_units: int
    longitude_shift_degrees: float
    mask_id: str


def _date_at_index(year: int, index: int) -> dt.date:
    return dt.date(year, 1, 1) + dt.timedelta(days=index)


def eligible_indices(year: int, radius: int = CONTEXT_RADIUS_DAYS) -> list[int]:
    n_days = 366 if calendar.isleap(year) else 365
    return list(range(radius, n_days - radius))


def _stable_integer(seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{seed}:{token}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _donor_candidates(target: dt.date, donor_years: Sequence[int]) -> list[dt.date]:
    candidates = []
    for year in donor_years:
        try:
            candidate = target.replace(year=year)
        except ValueError:
            continue
        if candidate - dt.timedelta(days=CONTEXT_RADIUS_DAYS) < dt.date(year, 1, 1):
            continue
        if candidate + dt.timedelta(days=CONTEXT_RADIUS_DAYS) > dt.date(year, 12, 31):
            continue
        candidates.append(candidate)
    if not candidates:
        raise RuntimeError(f"No valid mask donor for {target.isoformat()}")
    return candidates


def _longitude_shift_units(seed: int, target: dt.date) -> int:
    n_units = int(round(360.0 / COMMON_LONGITUDE_STEP_DEGREES))
    allowed = [unit for unit in range(1, n_units) if min(unit, n_units - unit) >= 20]
    return allowed[_stable_integer(seed, f"shift:{target.isoformat()}") % len(allowed)]


def make_record(
    split: str,
    index: int,
    target: dt.date,
    *,
    seed: int = DEFAULT_SEED,
    donor_years: Sequence[int] = DONOR_YEARS,
) -> EvaluationRecord:
    candidates = _donor_candidates(target, donor_years)
    donor = candidates[_stable_integer(seed, f"donor:{target.isoformat()}") % len(candidates)]
    shift_units = _longitude_shift_units(seed, target)
    mask_id = hashlib.sha256(
        f"real-availability-v1:{donor.isoformat()}:{shift_units}".encode("ascii")
    ).hexdigest()[:16]
    radius = dt.timedelta(days=CONTEXT_RADIUS_DAYS)
    return EvaluationRecord(
        split=split,
        index=index,
        central_date=target.isoformat(),
        context_start=(target - radius).isoformat(),
        context_end=(target + radius).isoformat(),
        donor_year=donor.year,
        donor_central_date=donor.isoformat(),
        donor_context_start=(donor - radius).isoformat(),
        donor_context_end=(donor + radius).isoformat(),
        longitude_shift_units=shift_units,
        longitude_shift_degrees=shift_units * COMMON_LONGITUDE_STEP_DEGREES,
        mask_id=mask_id,
    )


def _records_for_indices(split: str, year: int, indices: Iterable[int], seed: int) -> list[EvaluationRecord]:
    return [make_record(split, index, _date_at_index(year, index), seed=seed) for index in indices]


def pilot_indices(year: int = 2023, n_dates: int = 24) -> list[int]:
    eligible = eligible_indices(year)
    positions = np.rint(np.linspace(0, len(eligible) - 1, n_dates)).astype(int)
    indices = [eligible[position] for position in positions]
    if len(set(indices)) != n_dates:
        raise RuntimeError("Pilot date selection produced duplicate indices")
    return indices


def _manifest(name: str, records: list[EvaluationRecord], seed: int) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "seed": seed,
        "context_days": 2 * CONTEXT_RADIUS_DAYS + 1,
        "records": [asdict(record) for record in records],
    }
    payload["content_sha256"] = sha256_payload(payload)
    return payload


def build_publication_manifests(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    donor_records = []
    for year in DONOR_YEARS:
        donor_records.extend(_records_for_indices("mask_donor", year, eligible_indices(year), seed))
    pilot_records = _records_for_indices("pilot", 2023, pilot_indices(), seed)
    final_indices = eligible_indices(2024)
    if final_indices != list(range(7, 359)) or len(final_indices) != 352:
        raise RuntimeError("The 2024 protocol must contain exactly indices 7..358")
    final_records = _records_for_indices("test", 2024, final_indices, seed)

    manifests = {
        "donors": _manifest("mask_donors_2017_2022", donor_records, seed),
        "pilot": _manifest("pilot_2023_24_dates", pilot_records, seed),
        "test": _manifest("test_2024_352_dates", final_records, seed),
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "train_years": list(DONOR_YEARS),
        "validation_year": 2023,
        "test_year": 2024,
        "pilot_dates": 24,
        "test_indices_inclusive": [7, 358],
        "test_dates": 352,
        "mask_family": "displaced_real_fused_availability",
        "coarse_mask_rule": "any_x1_withheld_pixel",
        "longitude_alignment": {
            "step_degrees": COMMON_LONGITUDE_STEP_DEGREES,
            "pixels": {"x1": 30, "x3": 10, "x10": 3},
        },
        "manifests": {
            key: {
                "filename": f"{key}.json",
                "content_sha256": value["content_sha256"],
            }
            for key, value in manifests.items()
        },
    }
    protocol["content_sha256"] = sha256_payload(protocol)
    manifests["protocol"] = protocol

    paths = {}
    for key, payload in manifests.items():
        path = output_dir / f"{key}.json"
        atomic_write_json(path, payload)
        write_sha256_sidecar(path)
        paths[key] = path
    return paths


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text())
    expected = payload.pop("content_sha256", None)
    if expected is None or sha256_payload(payload) != expected:
        raise RuntimeError(f"Manifest content hash mismatch: {path}")
    payload["content_sha256"] = expected
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Croscim publication evaluation manifests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    paths = build_publication_manifests(args.output_dir, args.seed)
    for key, path in paths.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
