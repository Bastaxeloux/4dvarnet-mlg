from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json, sha256_file, write_sha256_sidecar


@dataclass(frozen=True)
class BestCheckpoint:
    path: str
    epoch: int
    global_step: int
    score: float
    monitor: str
    source_state_checkpoint: str
    sha256: str


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect Lightning checkpoints") from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Invalid Lightning checkpoint: {path}")
    return checkpoint


def _resolve_recorded_path(recorded: str, checkpoint_dir: Path) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path.resolve()

    local_candidate = checkpoint_dir / path.name
    if local_candidate.is_file():
        return local_candidate.resolve()

    matches = list(checkpoint_dir.rglob(path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Checkpoint recorded by Lightning is missing: {recorded}. "
            f"Searched under {checkpoint_dir}."
        )
    raise RuntimeError(f"Ambiguous checkpoint basename {path.name}: {matches}")


def _find_monitor_state(
    callbacks: Mapping[Any, Any], monitor: str
) -> tuple[str, Mapping[str, Any]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for callback_key, callback_state in callbacks.items():
        if not isinstance(callback_state, Mapping):
            continue
        key_text = str(callback_key)
        paths = callback_state.get("best_k_models", {})
        monitors_metric = monitor in key_text or callback_state.get("monitor") == monitor
        if monitors_metric and isinstance(paths, Mapping) and paths:
            candidates.append((key_text, callback_state))

    if len(candidates) != 1:
        found = [key for key, _ in candidates]
        raise RuntimeError(
            f"Expected exactly one ModelCheckpoint callback for {monitor!r}; found {found or 'none'}"
        )
    return candidates[0]


def select_best_checkpoint(
    checkpoint_dir: str | Path,
    state_checkpoint: str | Path | None = None,
    *,
    monitor: str = "val/x1/loss",
    cycle_epochs: int = 24,
) -> BestCheckpoint:
    """Select the native Lightning best checkpoint, never a posterior reranking."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    state_path = Path(state_checkpoint or checkpoint_dir / "last.ckpt").expanduser().resolve()
    if not state_path.is_file():
        raise FileNotFoundError(f"Lightning state checkpoint not found: {state_path}")

    state_checkpoint_data = _load_checkpoint(state_path)
    callbacks = state_checkpoint_data.get("callbacks")
    if not isinstance(callbacks, Mapping):
        raise RuntimeError(f"No Lightning callback state found in {state_path}")

    _, callback_state = _find_monitor_state(callbacks, monitor)
    best_k_models = callback_state["best_k_models"]
    scored_paths = [
        (_as_float(score), str(path))
        for path, score in best_k_models.items()
    ]
    score, recorded_path = min(scored_paths, key=lambda item: item[0])
    selected_path = _resolve_recorded_path(recorded_path, checkpoint_dir)

    selected = _load_checkpoint(selected_path)
    epoch = int(selected.get("epoch", -1))
    global_step = int(selected.get("global_step", -1))
    if epoch < 0 or (epoch + 1) % cycle_epochs != 0:
        raise RuntimeError(
            f"Selected checkpoint epoch {epoch} is not a {cycle_epochs}-epoch cycle boundary"
        )

    if selected_path.name == "last.ckpt":
        raise RuntimeError("The native best callback unexpectedly points to last.ckpt")

    return BestCheckpoint(
        path=str(selected_path),
        epoch=epoch,
        global_step=global_step,
        score=score,
        monitor=monitor,
        source_state_checkpoint=str(state_path),
        sha256=sha256_file(selected_path),
    )


def snapshot_best_checkpoint(
    selected: BestCheckpoint,
    destination: str | Path,
    manifest_path: str | Path | None = None,
) -> BestCheckpoint:
    source = Path(selected.path)
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if sha256_file(destination) != selected.sha256:
            raise RuntimeError(f"Immutable snapshot already exists with different content: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != selected.sha256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Checkpoint snapshot hash mismatch after copy")
        os.replace(temporary, destination)
        destination.chmod(0o440)

    snapshot = BestCheckpoint(
        **{
            **asdict(selected),
            "path": str(destination),
            "sha256": sha256_file(destination),
        }
    )
    write_sha256_sidecar(destination, snapshot.sha256)
    if manifest_path is not None:
        atomic_write_json(manifest_path, {"schema_version": 1, **asdict(snapshot)})
        write_sha256_sidecar(manifest_path)
    return snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the native Lightning best cycle checkpoint")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--state-checkpoint")
    parser.add_argument("--monitor", default="val/x1/loss")
    parser.add_argument("--cycle-epochs", type=int, default=24)
    parser.add_argument("--snapshot")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--field",
        choices=("json", "path", "epoch", "score", "sha256"),
        default="json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    selected = select_best_checkpoint(
        args.checkpoint_dir,
        args.state_checkpoint,
        monitor=args.monitor,
        cycle_epochs=args.cycle_epochs,
    )
    if args.snapshot:
        selected = snapshot_best_checkpoint(selected, args.snapshot, args.manifest)
    elif args.manifest:
        atomic_write_json(args.manifest, {"schema_version": 1, **asdict(selected)})
        write_sha256_sidecar(args.manifest)

    payload = asdict(selected)
    print(json.dumps(payload, sort_keys=True) if args.field == "json" else payload[args.field])


if __name__ == "__main__":
    main()
