from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, payload: Any) -> str:
    content = canonical_json_bytes(payload)
    atomic_write_bytes(path, content)
    return hashlib.sha256(content).hexdigest()


def write_sha256_sidecar(path: str | Path, digest: str | None = None) -> Path:
    artifact = Path(path)
    digest = digest or sha256_file(artifact)
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    atomic_write_bytes(sidecar, f"{digest}  {artifact.name}\n".encode("ascii"))
    return sidecar
