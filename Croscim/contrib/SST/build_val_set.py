"""Construction d'un set de validation figé en deux sous-ensembles.

Le builder scanne un budget fixe de candidats, puis sélectionne :

- ``viz`` (n_viz patchs) : meilleurs candidats selon un score continu pour les
  figures par epoch.
- ``loss`` (n_loss patchs) : candidats tirés aléatoirement parmi les patchs qui
  passent le filtre bas, pour ``val/loss``.

Les indices retenus sont sérialisés en JSON pour rejouer exactement le même set
entre runs.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
    std_sst: list = field(default_factory=list)
    score: list = field(default_factory=list)


@dataclass
class _CandidateStats:
    idx: int
    valid_ratio: float
    mean_sst: float
    ocean_pct: float
    std_sst: float
    score: float
    passes_loss: bool


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
    std = float(np.nanstd(data)) if valid_ratio > 0 else 0.0

    ocean_pct = float("nan")
    mask = patch_dict.get("surfmask")
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[0]
        ocean_pixels = int(np.sum((mask == 1) | (mask == 2) | (mask == 3)))
        ocean_pct = 100.0 * ocean_pixels / mask.size
    return {
        "valid_ratio": valid_ratio,
        "mean": mean,
        "ocean_pct": ocean_pct,
        "std": std,
    }


def _plot_histogram(
    out_path: Path,
    title: str,
    xlabel: str,
    all_values: Sequence[float],
    viz_values: Sequence[float],
    loss_values: Sequence[float],
    thresholds: dict[str, float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[VAL SET] matplotlib unavailable, skipping histogram {out_path.name}")
        return

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
    valid_ratio = stats["valid_ratio"]
    ocean_pct = stats["ocean_pct"]
    if not np.isfinite(valid_ratio) or valid_ratio < thresholds.min_valid_ratio:
        return False
    if (std_for_var * std_for_var) < thresholds.min_variance:
        return False
    if not np.isfinite(ocean_pct) or ocean_pct / 100.0 < thresholds.min_ocean_ratio:
        return False
    return True


def _viz_score(stats: dict, filter_viz: FilterThresholds) -> float:
    """Score continu pour classer les meilleurs patchs de visualisation."""
    valid_ratio = stats["valid_ratio"]
    ocean_ratio = stats["ocean_pct"] / 100.0
    variance = stats["std"] * stats["std"]
    if not np.isfinite(valid_ratio):
        valid_ratio = 0.0
    if not np.isfinite(ocean_ratio):
        ocean_ratio = 0.0
    if not np.isfinite(variance):
        variance = 0.0

    var_ref = max(filter_viz.min_variance, 1e-12)
    return float(
        0.45 * min(valid_ratio / max(filter_viz.min_valid_ratio, 1e-12), 1.0)
        + 0.35 * min(variance / var_ref, 1.0)
        + 0.20 * min(ocean_ratio / max(filter_viz.min_ocean_ratio, 1e-12), 1.0)
    )


def _scan_candidate(
    val_ds,
    idx: int,
    filter_viz: FilterThresholds,
    filter_loss: FilterThresholds,
) -> _CandidateStats | None:
    """Charge un candidat et renvoie ses stats utiles à la sélection."""
    try:
        try:
            import dask
            dask.config.set(scheduler="synchronous")
        except Exception:
            pass
        sample = val_ds[idx]
    except Exception as exc:  # noqa: BLE001 - patch loading can fail in many ways
        tqdm.write(f"[VAL SET] idx={idx} load failed: {exc}")
        return None

    patch_dict = _extract_patch_dict(sample)
    stats = _patch_stats(patch_dict)
    if stats is None:
        return None

    score = _viz_score(stats, filter_viz)
    passes_loss = _passes(stats, filter_loss, stats["std"])
    return _CandidateStats(
        idx=int(idx),
        valid_ratio=stats["valid_ratio"],
        mean_sst=stats["mean"],
        ocean_pct=stats["ocean_pct"],
        std_sst=stats["std"],
        score=score,
        passes_loss=passes_loss,
    )


def _candidate_to_payload(candidate: _CandidateStats) -> dict:
    return {
        "idx": int(candidate.idx),
        "valid_ratio": candidate.valid_ratio,
        "mean_sst": candidate.mean_sst,
        "ocean_pct": candidate.ocean_pct,
        "std_sst": candidate.std_sst,
        "score": candidate.score,
    }


def build_validation_set(
    val_ds,
    output_dir: Path,
    n_viz: int = 16,
    n_loss: int = 48,
    filter_viz: FilterThresholds | None = None,
    filter_loss: FilterThresholds | None = None,
    max_scan: int = 2000,
    candidate_budget: int | None = None,
    num_workers: int = 0,
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

    n_dataset = len(val_ds)
    if n_dataset == 0:
        raise RuntimeError("val_ds is empty, cannot build validation set")

    if candidate_budget is None:
        candidate_budget = max_scan
    candidate_budget = max(0, min(int(candidate_budget), n_dataset))
    num_workers = max(0, int(num_workers or 0))
    candidate_indices = rng.choice(n_dataset, size=candidate_budget, replace=False)
    candidate_indices = [int(idx) for idx in candidate_indices]

    print(
        f"[VAL SET] Scanning {candidate_budget}/{n_dataset} candidate patches "
        f"(workers={num_workers}, seed={seed})"
    )

    candidates: list[_CandidateStats] = []
    if num_workers <= 1:
        iterator = (
            _scan_candidate(val_ds, idx, filter_viz, filter_loss)
            for idx in candidate_indices
        )
        for candidate in tqdm(iterator, total=candidate_budget, desc="Scan set val", leave=True):
            if candidate is not None:
                candidates.append(candidate)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_scan_candidate, val_ds, idx, filter_viz, filter_loss)
                for idx in candidate_indices
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scan set val", leave=True):
                candidate = future.result()
                if candidate is not None:
                    candidates.append(candidate)

    # Tri stable pour rendre le résultat indépendant de l'ordre de fin des threads.
    candidates.sort(key=lambda item: item.idx)
    for candidate in candidates:
        accumulator.valid_ratio.append(candidate.valid_ratio)
        accumulator.mean_sst.append(candidate.mean_sst)
        accumulator.ocean_pct.append(candidate.ocean_pct)
        accumulator.std_sst.append(candidate.std_sst)
        accumulator.score.append(candidate.score)

    loss_pool = [candidate for candidate in candidates if candidate.passes_loss]
    viz_candidates = sorted(loss_pool, key=lambda item: (-item.score, item.idx))
    selected_viz = viz_candidates[:n_viz]
    viz_idx_set = {candidate.idx for candidate in selected_viz}

    remaining_loss_pool = [candidate for candidate in loss_pool if candidate.idx not in viz_idx_set]
    if len(remaining_loss_pool) > n_loss:
        chosen_pos = rng.choice(len(remaining_loss_pool), size=n_loss, replace=False)
        selected_loss = [remaining_loss_pool[int(pos)] for pos in chosen_pos]
        selected_loss.sort(key=lambda item: item.idx)
    else:
        selected_loss = remaining_loss_pool

    viz_indices = [candidate.idx for candidate in selected_viz]
    loss_indices = [candidate.idx for candidate in selected_loss]

    partial_viz = len(viz_indices) < n_viz
    partial_loss = len(loss_indices) < n_loss
    if partial_viz:
        print(
            f"[VAL SET] WARNING: only {len(viz_indices)}/{n_viz} viz patches selected "
            f"from {len(loss_pool)} loss-eligible candidates "
            f"(budget={candidate_budget})"
        )
    if partial_loss:
        print(
            f"[VAL SET] WARNING: only {len(loss_indices)}/{n_loss} loss patches selected "
            f"after reserving viz patches (budget={candidate_budget})"
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
            "  Suggested fixes: increase datamodule.val_candidate_budget, lower patch_filter\n"
            "  thresholds, change the validation date window, or set rebuild_val_set=true.\n"
            + "!" * 80 + "\n"
        )

    # Histograms — three figures with thresholds annotated.
    selected_viz_idx = {candidate.idx for candidate in selected_viz}
    selected_loss_idx = {candidate.idx for candidate in selected_loss}

    def _split(values):
        viz_vals = [value for value, candidate in zip(values, candidates) if candidate.idx in selected_viz_idx]
        loss_vals = [value for value, candidate in zip(values, candidates) if candidate.idx in selected_loss_idx]
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
        "selection_method": "ranked_budget_v2",
        "viz": [_candidate_to_payload(candidate) for candidate in selected_viz],
        "loss": [_candidate_to_payload(candidate) for candidate in selected_loss],
        "filter_viz": filter_viz.as_kwargs(),
        "filter_loss": filter_loss.as_kwargs(),
        "n_scanned": len(accumulator.valid_ratio),
        "n_candidates_requested": candidate_budget,
        "n_loss_eligible": len(loss_pool),
        "num_workers": num_workers,
        "seed": seed,
        "n_viz_target": n_viz,
        "n_loss_target": n_loss,
        "dataset_len": n_dataset,
    }
    tmp_path = output_dir / f"val_indices.json.tmp.{os.getpid()}"
    final_path = output_dir / "val_indices.json"
    with open(tmp_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    tmp_path.replace(final_path)

    print(
        f"[VAL SET] {len(viz_indices)} viz + {len(loss_indices)} loss patches "
        f"(scanned {len(accumulator.valid_ratio)}, loss-eligible {len(loss_pool)}). "
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
    candidate_budget: int | None = None,
    seed: int | None = None,
    filter_viz: FilterThresholds | None = None,
    filter_loss: FilterThresholds | None = None,
) -> tuple[bool, str]:
    """Vérifie que le cache JSON correspond aux paramètres actuels.

    Returns (compatible, reason). Si incompatible, le caller doit reconstruire.
    """
    with open(json_path, "r") as fp:
        payload = json.load(fp)

    mismatches = []
    for key, current, label in [
        ("selection_method", "ranked_budget_v2", "selection_method"),
        ("n_viz_target", n_viz, "n_viz"),
        ("n_loss_target", n_loss, "n_loss"),
        ("dataset_len", dataset_len, "dataset_len"),
    ]:
        cached = payload.get(key)
        if cached is None:
            mismatches.append(f"{label} absent du cache (ancien format)")
        elif cached != current:
            mismatches.append(f"{label}: cache={cached} vs config={current}")

    optional_checks = []
    if candidate_budget is not None:
        optional_checks.append(("n_candidates_requested", int(candidate_budget), "val_candidate_budget"))
    if seed is not None:
        optional_checks.append(("seed", int(seed), "val_set_seed"))
    if filter_viz is not None:
        optional_checks.append(("filter_viz", filter_viz.as_kwargs(), "filter_viz"))
    if filter_loss is not None:
        optional_checks.append(("filter_loss", filter_loss.as_kwargs(), "filter_loss"))

    for key, current, label in optional_checks:
        cached = payload.get(key)
        if cached is None:
            mismatches.append(f"{label} absent du cache (ancien format)")
        elif cached != current:
            mismatches.append(f"{label}: cache={cached} vs config={current}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, ""
