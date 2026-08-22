"""Hydra helpers for loading generated SST normalization statistics."""

from pathlib import Path

import yaml


def load_stats_section(path, section):
    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or section not in payload:
        raise KeyError(f"Section {section!r} not found in {stats_path}")
    if not isinstance(payload[section], dict):
        raise TypeError(f"Section {section!r} in {stats_path} must be a mapping")
    return payload[section]
