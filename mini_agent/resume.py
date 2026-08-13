from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mini_agent.artifacts import ArtifactRef
from mini_agent.journal import EventStore, RunEvent, inspect_run
from mini_agent.snapshots import (
    ArtifactSnapshotStore,
    RunSnapshot,
    tool_schema_fingerprint,
    workspace_identity,
)
from mini_agent.tools import ToolRegistry


ResumeDecision = Literal[
    "resume_allowed",
    "already_completed",
    "manual_reconciliation_required",
    "resume_blocked",
]


@dataclass(frozen=True)
class ResumeAnalysis:
    run_id: str
    decision: ResumeDecision
    resume_from_step: int | None
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    snapshot_sequence: int | None
    journal_sequence: int


class ResumeAnalyzer:
    """只读恢复分析器；不会请求模型，也不会执行任何工具。"""

    def __init__(
        self,
        event_store: EventStore,
        snapshot_store: ArtifactSnapshotStore,
        tools: ToolRegistry,
        *,
        llm_identity: str,
    ) -> None:
        self.event_store = event_store
        self.snapshot_store = snapshot_store
        self.tools = tools
        self.llm_identity = llm_identity

    def analyze(self, run_id: str) -> ResumeAnalysis:
        events = self.event_store.load(run_id)
        if not events:
            return ResumeAnalysis(
                run_id=run_id,
                decision="resume_blocked",
                resume_from_step=None,
                blockers=("run_not_found",),
                warnings=(),
                snapshot_sequence=None,
                journal_sequence=0,
            )

        inspection = inspect_run(events)
        latest_sequence = events[-1].sequence
        if inspection.status == "completed":
            return ResumeAnalysis(
                run_id=run_id,
                decision="already_completed",
                resume_from_step=None,
                blockers=(),
                warnings=(),
                snapshot_sequence=self._latest_snapshot_sequence(events),
                journal_sequence=latest_sequence,
            )

        blockers: list[str] = []
        warnings: list[str] = []
        if [event.sequence for event in events] != list(
            range(1, latest_sequence + 1)
        ):
            blockers.append("journal_sequence_gap")
        snapshot_event = self._latest_snapshot_event(events)
        if snapshot_event is None:
            blockers.append("missing_snapshot")
            return self._result(
                run_id,
                blockers,
                warnings,
                None,
                latest_sequence,
            )

        snapshot_sequence = int(snapshot_event.payload["last_event_sequence"])
        if snapshot_event.sequence != snapshot_sequence + 1:
            blockers.append("snapshot_sequence_mismatch")

        try:
            snapshot = self.snapshot_store.load(
                ArtifactRef(
                    artifact_id=str(snapshot_event.payload["artifact_id"]),
                    sha256=str(snapshot_event.payload["sha256"]),
                    bytes=int(snapshot_event.payload["bytes"]),
                    media_type=str(snapshot_event.payload["media_type"]),
                )
            )
        except (OSError, ValueError, IOError, KeyError, TypeError) as error:
            blockers.append(f"snapshot_unavailable:{type(error).__name__}")
            return self._result(
                run_id,
                blockers,
                warnings,
                snapshot_sequence,
                latest_sequence,
            )

        if snapshot.run_id != run_id:
            blockers.append("snapshot_run_id_mismatch")
        if snapshot.last_event_sequence != snapshot_sequence:
            blockers.append("snapshot_sequence_mismatch")
        if snapshot.resume_point == "run_completion":
            blockers.append("completion_commit_incomplete")
        if snapshot.llm_identity != self.llm_identity:
            blockers.append("llm_identity_changed")
        if snapshot.tool_schema_sha256 != tool_schema_fingerprint(
            self.tools.list_tools()
        ):
            blockers.append("tool_schema_changed")

        current_workspace = workspace_identity(self.tools.root)
        if snapshot.workspace.root != current_workspace.root:
            blockers.append("workspace_root_changed")
        if snapshot.workspace.git_head != current_workspace.git_head:
            blockers.append("git_head_changed")
        if snapshot.workspace.state_sha256 != current_workspace.state_sha256:
            blockers.append("workspace_state_changed")

        if inspection.incomplete_tools:
            if any(
                item.disposition == "unknown_outcome"
                for item in inspection.incomplete_tools
            ):
                blockers.append("manual_reconciliation_required")
            else:
                warnings.append("started_only_read_tool")

        tail_events = [
            event
            for event in events
            if event.sequence > snapshot_event.sequence
        ]
        if any(
            event.event_type == "model_request_completed"
            for event in tail_events
        ):
            blockers.append("unsnapshotted_model_turn")
        if any(
            event.event_type == "tool_execution_completed"
            for event in tail_events
        ):
            blockers.append("unsnapshotted_tool_result")

        if blockers:
            decision: ResumeDecision = (
                "manual_reconciliation_required"
                if "manual_reconciliation_required" in blockers
                else "resume_blocked"
            )
        else:
            decision = "resume_allowed"
        return ResumeAnalysis(
            run_id=run_id,
            decision=decision,
            resume_from_step=snapshot.next_step,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            snapshot_sequence=snapshot_sequence,
            journal_sequence=latest_sequence,
        )

    @staticmethod
    def _latest_snapshot_event(events: list[RunEvent]) -> RunEvent | None:
        for event in reversed(events):
            if event.event_type == "snapshot_saved":
                return event
        return None

    def _latest_snapshot_sequence(self, events: list[RunEvent]) -> int | None:
        event = self._latest_snapshot_event(events)
        if event is None:
            return None
        return int(event.payload["last_event_sequence"])

    @staticmethod
    def _result(
        run_id: str,
        blockers: list[str],
        warnings: list[str],
        snapshot_sequence: int | None,
        journal_sequence: int,
    ) -> ResumeAnalysis:
        return ResumeAnalysis(
            run_id=run_id,
            decision="resume_blocked",
            resume_from_step=None,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            snapshot_sequence=snapshot_sequence,
            journal_sequence=journal_sequence,
        )


__all__ = ["ResumeAnalysis", "ResumeAnalyzer", "ResumeDecision"]
