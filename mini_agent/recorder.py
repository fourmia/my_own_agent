from __future__ import annotations

from typing import Any

from mini_agent.artifacts import ArtifactRef, ArtifactStore
from mini_agent.journal import EventStore, EventType, content_summary
from mini_agent.snapshots import RunSnapshot, SnapshotStore


class RunRecorder:
    """集中管理 Journal 与 Snapshot 的写入顺序。"""

    def __init__(
        self,
        event_store: EventStore | None = None,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        self.event_store = event_store
        self.snapshot_store = snapshot_store
        self._last_sequences: dict[str, int] = {}

    def record(
        self,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> int | None:
        if self.event_store is None:
            return None
        event = self.event_store.append(run_id, event_type, payload)
        self._last_sequences[run_id] = event.sequence
        return event.sequence

    def current_sequence(self, run_id: str) -> int:
        return self._last_sequences.get(run_id, 0)

    def claim_resume(
        self,
        run_id: str,
        expected_sequence: int,
        payload: dict[str, Any],
    ) -> int:
        if self.event_store is None:
            raise RuntimeError("resume requires an EventStore")
        event = self.event_store.claim_resume(run_id, expected_sequence, payload)
        self._last_sequences[run_id] = event.sequence
        return event.sequence

    def save_snapshot(self, snapshot: RunSnapshot) -> ArtifactRef | None:
        if self.snapshot_store is None:
            return None
        ref = self.snapshot_store.save(snapshot)
        snapshot_id = f"{snapshot.run_id}:snapshot:{snapshot.last_event_sequence}"
        self.record(
            snapshot.run_id,
            "snapshot_saved",
            {
                "snapshot_id": snapshot_id,
                "artifact_id": ref.artifact_id,
                "last_event_sequence": snapshot.last_event_sequence,
                "sha256": ref.sha256,
                "bytes": ref.bytes,
                "media_type": ref.media_type,
            },
        )
        return ref

    @staticmethod
    def content_summary(content: str) -> dict[str, Any]:
        return content_summary(content)


__all__ = ["RunRecorder"]
