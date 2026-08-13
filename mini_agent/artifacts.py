from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    bytes: int
    media_type: str


class ArtifactStore(Protocol):
    def put(self, content: bytes, media_type: str) -> ArtifactRef:
        """Persist bytes by content hash and return an integrity reference."""

    def get(self, ref: ArtifactRef) -> bytes:
        """Read and verify an artifact reference."""


class FileArtifactStore:
    """Content-addressed local artifact store for snapshots and large evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        if not media_type or "/" not in media_type:
            raise ValueError("media_type must be a non-empty MIME type")
        digest = hashlib.sha256(content).hexdigest()
        artifact_id = digest
        target = self.root / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = target.read_bytes()
            if hashlib.sha256(existing).hexdigest() != digest:
                raise IOError(f"artifact hash collision or corruption: {target}")
        else:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.",
                dir=target.parent,
            )
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, target)
            except Exception:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise
        return ArtifactRef(artifact_id, digest, len(content), media_type)

    def get(self, ref: ArtifactRef) -> bytes:
        if ref.artifact_id != ref.sha256:
            raise ValueError("artifact_id must equal sha256 in the file store")
        if len(ref.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in ref.sha256.lower()
        ):
            raise ValueError("invalid artifact sha256")
        target = (self.root / ref.sha256[:2] / ref.sha256).resolve()
        if target != self.root / ref.sha256[:2] / ref.sha256:
            raise ValueError("artifact path escapes store")
        content = target.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != ref.sha256:
            raise IOError(
                f"artifact integrity check failed: expected {ref.sha256}, got {actual}"
            )
        if len(content) != ref.bytes:
            raise IOError("artifact byte count does not match reference")
        return content


__all__ = ["ArtifactRef", "ArtifactStore", "FileArtifactStore"]
