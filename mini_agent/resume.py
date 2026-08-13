from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mini_agent.agent import Agent, AgentResult
from mini_agent.artifacts import ArtifactRef
from mini_agent.journal import EventStore, RunEvent, inspect_run
from mini_agent.snapshots import (
    RunSnapshot,
    SnapshotStore,
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


class ResumeRejectedError(RuntimeError):
    """恢复前置检查未通过；异常中保留机器可读分析结果。"""

    def __init__(self, analysis: ResumeAnalysis) -> None:
        super().__init__(f"resume rejected: {analysis.decision}")
        self.analysis = analysis


class ResumeAnalyzer:
    """只读恢复分析器；不会请求模型，也不会执行任何工具。"""

    def __init__(
        self,
        event_store: EventStore,
        snapshot_store: SnapshotStore,
        tools: ToolRegistry,
        *,
        llm_identity: str,
        permission_policy_identity: str | None = None,
        max_steps: int | None = None,
        require_verification: bool | None = None,
    ) -> None:
        self.event_store = event_store
        self.snapshot_store = snapshot_store
        self.tools = tools
        self.llm_identity = llm_identity
        self.permission_policy_identity = permission_policy_identity
        self.max_steps = max_steps
        self.require_verification = require_verification

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
        if snapshot.resume_point == "tool_batch":
            if not snapshot.messages or snapshot.messages[-1].role != "assistant":
                blockers.append("pending_tool_batch_message_invalid")
            pending_calls = (
                snapshot.messages[-1].tool_calls if snapshot.messages else ()
            )
            if not pending_calls:
                blockers.append("pending_tool_batch_missing")
            pending_call_ids = {call.call_id for call in pending_calls}
            persisted_result_ids = {
                message.tool_call_id
                for message in snapshot.messages
                if message.role == "tool" and message.tool_call_id is not None
            }
            if pending_call_ids.intersection(persisted_result_ids):
                blockers.append("pending_tool_result_in_snapshot")
            for call in pending_calls:
                tool = self.tools.get(call.name)
                if tool is None:
                    blockers.append("pending_tool_unavailable")
                elif tool.permission != "read" or tool.effect == "workspace_mutation":
                    blockers.append("pending_non_read_tool")
        if snapshot.llm_identity != self.llm_identity:
            blockers.append("llm_identity_changed")
        if snapshot.permission_policy_identity is None:
            blockers.append("permission_policy_unknown")
        elif (
            self.permission_policy_identity is not None
            and snapshot.permission_policy_identity != self.permission_policy_identity
        ):
            blockers.append("permission_policy_changed")
        if self.max_steps is not None and snapshot.max_steps != self.max_steps:
            blockers.append("max_steps_changed")
        if (
            self.require_verification is not None
            and snapshot.require_verification != self.require_verification
        ):
            blockers.append("verification_policy_changed")
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
            event.event_type in {
                "tool_execution_completed",
                "tool_execution_skipped",
            }
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

    def load_latest_snapshot(self, run_id: str) -> RunSnapshot:
        """加载最新 Snapshot；调用方必须先检查 analyze() 的决策。"""

        events = self.event_store.load(run_id)
        snapshot_event = self._latest_snapshot_event(events)
        if snapshot_event is None:
            raise ValueError("run has no snapshot")
        return self.snapshot_store.load(
            ArtifactRef(
                artifact_id=str(snapshot_event.payload["artifact_id"]),
                sha256=str(snapshot_event.payload["sha256"]),
                bytes=int(snapshot_event.payload["bytes"]),
                media_type=str(snapshot_event.payload["media_type"]),
            )
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


class ResumeExecutor:
    """先执行保守分析，再把已验证 Snapshot 交回同一个 Agent Runtime。"""

    def __init__(self, agent: Agent) -> None:
        event_store = agent.recorder.event_store
        snapshot_store = agent.recorder.snapshot_store
        if event_store is None or snapshot_store is None:
            raise ValueError("resume execution requires EventStore and SnapshotStore")
        self.agent = agent
        self.analyzer = ResumeAnalyzer(
            event_store,
            snapshot_store,
            agent.tools,
            llm_identity=agent.llm_identity(),
            permission_policy_identity=agent.permission_policy_identity(),
            max_steps=agent.max_steps,
            require_verification=agent.require_verification,
        )

    def run(self, run_id: str, *, trace: bool = True) -> AgentResult:
        analysis = self.analyzer.analyze(run_id)
        if analysis.decision != "resume_allowed":
            raise ResumeRejectedError(analysis)
        snapshot = self.analyzer.load_latest_snapshot(run_id)
        return self.agent._resume_from_snapshot(
            snapshot,
            expected_journal_sequence=analysis.journal_sequence,
            trace=trace,
        )


__all__ = [
    "ResumeAnalysis",
    "ResumeAnalyzer",
    "ResumeDecision",
    "ResumeExecutor",
    "ResumeRejectedError",
]
