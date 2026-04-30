"""Construction d'un set de validation figé en deux sous-ensembles.

Avant le training, on scanne `val_ds` pour sélectionner :

- ``viz`` (n_viz patchs) : filtre dur, qualité visuelle (pour figures par epoch).
- ``loss`` (n_loss patchs) : filtre bas, représentativité statistique
  (pour ``val/loss``).

Pendant le scan, les stats (valid_ratio, mean SST, ocean_pct) de tous les
patchs candidats sont accumulées puis tracées en histogrammes avec les
seuils annotés. Les indices retenus sont sérialisés en JSON pour rejouer
exactement le même set entre runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


@dataclass
class FilterThresholds:
    min_valid_ratio: float
    min_variance: float
    min_ocean_ratio: float

    def as_kwargs(self) -> dict:
        return {
            "min_valid_ratio": self.min_valid_ratio,
            "min_variance": self.min_variance,
            "min_ocean_ratio": self.min_ocean_ratio,
        }


@dataclass
class _ScanAccumulator:
    valid_ratio: list = field(default_factory=list)
    mean_sst: list = field(default_factory=list)
    ocean_pct: list = field(default_factory=list)
    accepted_viz_idx: list = field(default_factory=list)
    accepted_loss_idx: list = field(default_factory=list)


def _extract_patch_dict(sample: Any) -> dict:
    """Renvoie un dict {tgt_sst, surfmask} en numpy à partir d'un sample dataset.

    Le dataset multi-résolution renvoie soit un dict {patch_x1, patch_x3, ...},
    soit directement un objet TrainingItem. On extrait toujours le patch x1.
    """
    if isinstance(sample, dict) and "patch_x1" in sample:
        sample = sample["patch_x1"]
    if isinstance(sample, dict):
        return sample
    out = {}
    for attr in ("tgt_sst", "surfmask"):
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            if hasattr(value, "cpu"):
                value = value.cpu().numpy()
            out[attr] = value
    return out


def _patch_stats(patch_dict: dict) -> dict | None:
    """Calcule (valid_ratio, mean, ocean_pct) sans appliquer de seuil."""
    if "tgt_sst" not in patch_dict:
        return None
    data = patch_dict["tgt_sst"]
    valid_ratio = float(np.sum(~np.isnan(data)) / data.size)
    mean = float(np.nanmean(data)) if valid_ratio > 0 else float("nan")

    ocean_pct = float("nan")
    mask = patch_dict.get("surfmask")
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[0]
        ocean_pixels = int(np.sum((mask == 1) | (mask == 2) | (mask == 3)))
        ocean_pct = 100.0 * ocean_pixels / mask.size
    return {"valid_ratio": valid_ratio, "mean": mean, "ocean_pct": ocean_pct}


def _plot_histogram(
    out_path: Path,
    title: str,
    xlabel: str,
    all_values: Sequence[float],
    viz_values: Sequence[float],
    loss_values: Sequence[float],
    thresholds: dict[str, float],
) -> None:
    arr_all = np.array(all_values, dtype=float)
    arr_all = arr_all[~np.isnan(arr_all)]
    if arr_all.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(arr_all, bins=40, color="lightgray", alpha=0.8, label=f"scannés (n={len(arr_all)})")
    if viz_values:
        ax.hist(viz_values, bins=40, color="tab:blue", alpha=0.7, label=f"viz (n={len(viz_values)})")
    if loss_values:
        ax.hist(loss_values, bins=40, color="tab:orange", alpha=0.5, label=f"loss (n={len(loss_values)})")
    for label, value in thresholds.items():
        ax.axvline(value, linestyle="--", linewidth=1.2, label=f"{label}={value:g}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("nombre de patchs")
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _passes(stats: dict, thresholds: FilterThresholds, std_for_var: float) -> bool:
    """Filtre simplifié sur les stats déjà calculées.

    Reproduit la logique de `is_valid_patch` mais à partir des stats
    pré-calculées. La variance est approximée via std² (cohérent avec
    `is_valid_patch` qui utilise `np.nanvar`).
    """
    if stats["valid_ratio"] < thresholds.min_valid_ratio:
        return False
    if (std_for_var * std_for_var) < thresholds.min_variance:
        return False
    if stats["ocean_pct"] / 100.0 < thresholds.min_ocean_ratio:
        return False
    return True


def build_validation_set(
    val_ds,
    output_dir: Path,
    n_viz: int = 16,
    n_loss: int = 48,
    filter_viz: FilterThresholds | None = None,
    filter_loss: FilterThresholds | None = None,
    max_scan: int = 2000,
    seed: int = 42,
) -> list[int]:
    """Sélectionne deux sous-ensembles de patchs val et écrit JSON + histogrammes.

    Returns
    -------
    list[int]
        64 indices ordonnés : les n_viz premiers sont les patchs viz, les
        n_loss suivants sont les patchs loss. Cet ordre est exploité par
        ``on_validation_epoch_end`` qui prend les premiers patchs pour les
        figures.
    """
    if filter_viz is None:
        filter_viz = FilterThresholds(0.50, 0.30, 0.50)
    if filter_loss is None:
        filter_loss = FilterThresholds(0.02, 0.05, 0.05)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    accumulator = _ScanAccumulator()
    viz_indices: list[int] = []
    loss_indices: list[int] = []
    visited: set[int] = set()

    pbar = tqdm(
        total=n_viz + n_loss,
        desc="Construction set val",
        bar_format="{l_bar}{bar} | {postfix} [{elapsed}<{remaining}]",
        leave=True,
    )
    pbar.set_postfix_str(f"viz 0/{n_viz} | loss 0/{n_loss}")

    n_dataset = len(val_ds)
    if n_dataset == 0:
        pbar.close()
        raise RuntimeError("val_ds is empty, cannot build validation set")

    n_attempts = 0
    while n_attempts < max_scan:
        if len(viz_indices) >= n_viz and len(loss_indices) >= n_loss:
            break

        if len(visited) >= n_dataset:
            break
        idx = int(rng.integers(0, n_dataset))
        if idx in visited:
            continue
        visited.add(idx)
        n_attempts += 1

        try:
            sample = val_ds[idx]
        except Exception as exc:  # noqa: BLE001 - patch loading can fail in many ways
            tqdm.write(f"[VAL SET] idx={idx} load failed: {exc}")
            continue

        patch_dict = _extract_patch_dict(sample)
        stats = _patch_stats(patch_dict)
        if stats is None:
            continue

        std_val = float(np.nanstd(patch_dict["tgt_sst"])) if "tgt_sst" in patch_dict else 0.0

        accumulator.valid_ratio.append(stats["valid_ratio"])
        accumulator.mean_sst.append(stats["mean"])
        accumulator.ocean_pct.append(stats["ocean_pct"])

        if len(viz_indices) < n_viz and _passes(stats, filter_viz, std_val):
            viz_indices.append(idx)
            accumulator.accepted_viz_idx.append(len(accumulator.valid_ratio) - 1)
        elif len(loss_indices) < n_loss and _passes(stats, filter_loss, std_val):
            loss_indices.append(idx)
            accumulator.accepted_loss_idx.append(len(accumulator.valid_ratio) - 1)

        pbar.set_postfix_str(
            f"viz {len(viz_indices)}/{n_viz} | loss {len(loss_indices)}/{n_loss}"
        )
        pbar.n = len(viz_indices) + len(loss_indices)
        pbar.refresh()

    pbar.close()

    partial_viz = len(viz_indices) < n_viz
    partial_loss = len(loss_indices) < n_loss
    if partial_viz:
        print(
            f"[VAL SET] WARNING: only {len(viz_indices)}/{n_viz} viz patches found "
            f"after {n_attempts} scans (max_scan={max_scan})"
        )
    if partial_loss:
        print(
            f"[VAL SET] WARNING: only {len(loss_indices)}/{n_loss} loss patches found "
            f"after {n_attempts} scans (max_scan={max_scan})"
        )
    if partial_viz or partial_loss:
        expected_total = n_viz + n_loss
        actual_total = len(viz_indices) + len(loss_indices)
        print(
            "\n"
            + "!" * 80 + "\n"
            "[VAL SET] WARNING: validation set is incomplete.\n"
            f"  Expected: {n_viz} viz + {n_loss} loss = {expected_total} patches\n"
            f"  Found:    {len(viz_indices)} viz + {len(loss_indices)} loss = {actual_total} patches\n"
            "  This run can still continue, but validation plots/loss will use fewer patches.\n"
            "  Suggested fixes: increase datamodule.val_set_max_scan, lower patch_filter\n"
            "  thresholds, or set rebuild_val_set=true after changing config/date/stride.\n"
            + "!" * 80 + "\n"
        )

    # Histograms — three figures with thresholds annotated.
    viz_set = set(accumulator.accepted_viz_idx)
    loss_set = set(accumulator.accepted_loss_idx)

    def _split(values):
        viz_vals = [values[i] for i in range(len(values)) if i in viz_set]
        loss_vals = [values[i] for i in range(len(values)) if i in loss_set]
        return viz_vals, loss_vals

    viz_vr, loss_vr = _split(accumulator.valid_ratio)
    viz_mean, loss_mean = _split(accumulator.mean_sst)
    viz_op, loss_op = _split(accumulator.ocean_pct)

    _plot_histogram(
        output_dir / "hist_valid_ratio.png",
        title="Distribution valid_ratio (set val)",
        xlabel="valid_ratio",
        all_values=accumulator.valid_ratio,
        viz_values=viz_vr,
        loss_values=loss_vr,
        thresholds={
            "viz min": filter_viz.min_valid_ratio,
            "loss min": filter_loss.min_valid_ratio,
        },
    )
    _plot_histogram(
        output_dir / "hist_mean_sst.png",
        title="Distribution mean SST (set val)",
        xlabel="mean SST (°C)",
        all_values=accumulator.mean_sst,
        viz_values=viz_mean,
        loss_values=loss_mean,
        thresholds={},
    )
    _plot_histogram(
        output_dir / "hist_ocean_pct.png",
        title="Distribution ocean_pct (set val)",
        xlabel="ocean_pct (%)",
        all_values=accumulator.ocean_pct,
        viz_values=viz_op,
        loss_values=loss_op,
        thresholds={
            "viz min": 100 * filter_viz.min_ocean_ratio,
            "loss min": 100 * filter_loss.min_ocean_ratio,
        },
    )

    indices = viz_indices + loss_indices
    payload = {
        "viz": [
            {
                "idx": int(idx),
                "valid_ratio": accumulator.valid_ratio[accumulator.accepted_viz_idx[i]],
                "mean_sst": accumulator.mean_sst[accumulator.accepted_viz_idx[i]],
                "ocean_pct": accumulator.ocean_pct[accumulator.accepted_viz_idx[i]],
            }
            for i, idx in enumerate(viz_indices)
        ],
        "loss": [
            {
                "idx": int(idx),
                "valid_ratio": accumulator.valid_ratio[accumulator.accepted_loss_idx[i]],
                "mean_sst": accumulator.mean_sst[accumulator.accepted_loss_idx[i]],
                "ocean_pct": accumulator.ocean_pct[accumulator.accepted_loss_idx[i]],
            }
            for i, idx in enumerate(loss_indices)
        ],
        "filter_viz": filter_viz.as_kwargs(),
        "filter_loss": filter_loss.as_kwargs(),
        "n_scanned": len(accumulator.valid_ratio),
        "seed": seed,
        "n_viz_target": n_viz,
        "n_loss_target": n_loss,
        "dataset_len": n_dataset,
    }
    with open(output_dir / "val_indices.json", "w") as fp:
        json.dump(payload, fp, indent=2)

    print(
        f"[VAL SET] {len(viz_indices)} viz + {len(loss_indices)} loss patches "
        f"(scanned {len(accumulator.valid_ratio)}). "
        f"Indices and histograms saved to {output_dir}"
    )
    return indices


def load_validation_indices(json_path: Path) -> list[int]:
    """Recharge la liste d'indices depuis un fichier val_indices.json."""
    with open(json_path, "r") as fp:
        payload = json.load(fp)
    return [entry["idx"] for entry in payload["viz"]] + [
        entry["idx"] for entry in payload["loss"]
    ]


def check_val_cache_compatible(
    json_path: Path,
    n_viz: int,
    n_loss: int,
    dataset_len: int,
) -> tuple[bool, str]:
    """Vérifie que le cache JSON correspond aux paramètres actuels.

    Returns (compatible, reason). Si incompatible, le caller doit reconstruire.
    """
    with open(json_path, "r") as fp:
        payload = json.load(fp)

    mismatches = []
    for key, current, label in [
        ("n_viz_target", n_viz, "n_viz"),
        ("n_loss_target", n_loss, "n_loss"),
        ("dataset_len", dataset_len, "dataset_len"),
    ]:
        cached = payload.get(key)
        if cached is None:
            mismatches.append(f"{label} absent du cache (ancien format)")
        elif cached != current:
            mismatches.append(f"{label}: cache={cached} vs config={current}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, ""
