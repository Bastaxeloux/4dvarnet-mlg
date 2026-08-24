"""Resolve satellite ASCII files without confusing means and uncertainties."""

from pathlib import Path


SATELLITES = ("aasti", "avhrr", "pmw", "slstr")
STATISTICS = ("av", "std")
DERIVED_AV_SUFFIXES = ("_std_av.asc", "_min_av.asc", "_max_av.asc")


def satellite_ascii_candidates(directory, day, satellite, statistic):
    directory = Path(directory)
    if satellite not in SATELLITES:
        raise ValueError(f"Unsupported satellite: {satellite}")
    if statistic not in STATISTICS:
        raise ValueError(f"Unsupported statistic: {statistic}")

    candidates = sorted(directory.glob(f"{day}_{satellite}_*av.asc"))
    if statistic == "std":
        return [path for path in candidates if path.name.endswith("_std_av.asc")]
    return [
        path
        for path in candidates
        if not path.name.endswith(DERIVED_AV_SUFFIXES)
    ]


def resolve_satellite_ascii(directory, day, satellite, statistic):
    candidates = satellite_ascii_candidates(
        directory, day, satellite, statistic
    )
    logical_name = f"{satellite}_{statistic}"
    if not candidates:
        raise FileNotFoundError(
            f"Missing {logical_name} ASCII file in {directory} for {day}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Ambiguous {logical_name} ASCII files in {directory} for {day}: "
            f"{names}"
        )
    return candidates[0]
