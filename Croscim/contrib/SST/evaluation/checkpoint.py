from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json, sha256_file, write_sha256_sidecar


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect Lightning checkpoints") from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Invalid Lightning checkpoint: {path}")
    return checkpoint


def snapshot_checkpoint(
    checkpoint: str | Path,
    destination: str | Path,
    manifest_path: str | Path,
    *,
    cycle_epochs: int = 24,
) -> dict:
    source = Path(checkpoint).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")

    payload = _load_checkpoint(source)
    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", -1))
    if epoch < 0 or (epoch + 1) % cycle_epochs:
        raise RuntimeError(
            f"Checkpoint epoch {epoch} is not a {cycle_epochs}-epoch cycle boundary"
        )

    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise RuntimeError(
                f"Snapshot already exists with different content: {destination}"
            )
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != source_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Checkpoint snapshot hash mismatch")
        os.replace(temporary, destination)
        destination.chmod(0o440)

    manifest = {
        "schema_version": 1,
        "selection": "explicit_checkpoint_path",
        "selection_criterion": "val/x1/loss",
        "path": str(destination),
        "sha256": sha256_file(destination),
        "epoch": epoch,
        "global_step": global_step,
        "source_checkpoint": str(source),
    }
    atomic_write_json(manifest_path, manifest)
    write_sha256_sidecar(destination, manifest["sha256"])
    write_sha256_sidecar(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot an explicit cycle checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cycle-epochs", type=int, default=24)
    args = parser.parse_args()
    manifest = snapshot_checkpoint(
        args.checkpoint,
        args.snapshot,
        args.manifest,
        cycle_epochs=args.cycle_epochs,
    )
    print(f"checkpoint={manifest['path']}")
    print(f"epoch={manifest['epoch']}")
    print(f"sha256={manifest['sha256']}")


if __name__ == "__main__":
    main()
