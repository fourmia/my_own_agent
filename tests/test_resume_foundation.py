from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from pathlib import Path

import pytest

from mini_agent.agent import Agent
from mini_agent.artifacts import ArtifactRef, FileArtifactStore
from mini_agent.journal import SQLiteEventStore
from mini_agent.llm import LLM
from mini_agent.resume import ResumeAnalyzer
from mini_agent.schema import Message, ModelRequest, ModelTurn, ToolCall
from mini_agent.snapshots import (
    ArtifactSnapshotStore,
    RunSnapshot,
    SNAPSHOT_SCHEMA_VERSION,
    WorkspaceIdentity,
    message_from_dict,
    message_to_dict,
    snapshot_from_bytes,
    snapshot_to_bytes,
    tool_schema_fingerprint,
    workspace_identity,
)
from mini_agent.tools import Tool, create_default_registry


class CountingProvider(LLM):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: ModelRequest) -> ModelTurn:
        self.calls += 1
        raise AssertionError("resume analysis must not call the Provider")


def test_file_artifact_store_deduplicates_and_verifies_content(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    content = b"artifact payload"
    first = store.put(content, "text/plain")
    second = store.put(content, "text/plain")

    assert first == second
    assert store.get(first) == content
    assert list((tmp_path / "artifacts").rglob("*"))[-1].is_file()

    target = tmp_path / "artifacts" / first.sha256[:2] / first.sha256
    target.write_bytes(b"corrupted")
    with pytest.raises(IOError, match="integrity check failed"):
        store.get(first)


def test_file_artifact_store_rejects_invalid_media_type(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="MIME type"):
        store.put(b"data", "plain-text")


def test_message_and_snapshot_round_trip() -> None:
    messages = [
        Message("system", "rules"),
        Message(
            "assistant",
            "inspect",
            tool_calls=(
                ToolCall(
                    "call-1",
                    "read_file",
                    {"path": "README.md"},
                    raw_arguments='{"path":"README.md"}',
                ),
            ),
        ),
        Message(
            "tool",
            "content",
            tool_call_id="call-1",
            tool_name="read_file",
        ),
    ]
    snapshot = RunSnapshot(
        run_id="run-1",
        snapshot_version=SNAPSHOT_SCHEMA_VERSION,
        last_event_sequence=12,
        next_step=3,
        resume_point="model_request",
        messages=messages,
        seen_call_ids=["call-1"],
        verification={
            "mutation_version": 0,
            "diff_version": 0,
            "tests_version": 0,
        },
        max_steps=6,
        require_verification=True,
        llm_identity="fake.provider",
        tool_schema_sha256="tools-hash",
        workspace=WorkspaceIdentity("/workspace", "abc", "state"),
    )

    restored = snapshot_from_bytes(snapshot_to_bytes(snapshot))

    assert restored == snapshot
    assert message_from_dict(message_to_dict(messages[1])) == messages[1]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("resume_point", "unsafe_retry", "unsupported resume_point"),
        ("next_step", 0, "outside the run step range"),
        ("seen_call_ids", ["call-1", "call-1"], "must be unique"),
    ],
)
def test_snapshot_parser_rejects_invalid_recovery_state(
    field: str,
    value: object,
    error: str,
) -> None:
    snapshot = RunSnapshot(
        run_id="run-1",
        snapshot_version=SNAPSHOT_SCHEMA_VERSION,
        last_event_sequence=1,
        next_step=1,
        resume_point="model_request",
        messages=[Message("user", "task")],
        seen_call_ids=[],
        verification={
            "mutation_version": 0,
            "diff_version": 0,
            "tests_version": 0,
        },
        max_steps=6,
        require_verification=True,
        llm_identity="fake.provider",
        tool_schema_sha256="tools-hash",
        workspace=WorkspaceIdentity("/workspace", None, "state"),
    )
    payload = json.loads(snapshot_to_bytes(snapshot))
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        snapshot_from_bytes(json.dumps(payload).encode("utf-8"))


def test_snapshot_store_uses_content_addressed_artifact(tmp_path: Path) -> None:
    artifact_store = FileArtifactStore(tmp_path / "artifacts")
    snapshots = ArtifactSnapshotStore(artifact_store)
    snapshot = RunSnapshot(
        run_id="run-1",
        snapshot_version=1,
        last_event_sequence=1,
        next_step=1,
        resume_point="model_request",
        messages=[Message("user", "task")],
        seen_call_ids=[],
        verification={
            "mutation_version": 0,
            "diff_version": 0,
            "tests_version": 0,
        },
        max_steps=6,
        require_verification=True,
        llm_identity="fake.provider",
        tool_schema_sha256="tools-hash",
        workspace=WorkspaceIdentity("/workspace", None, "state"),
    )

    ref = snapshots.save(snapshot)

    assert ref.media_type == "application/json"
    assert snapshots.load(ref) == snapshot

    with pytest.raises(ValueError, match="application/json"):
        snapshots.load(
            ArtifactRef(ref.artifact_id, ref.sha256, ref.bytes, "text/plain")
        )


def test_agent_saves_snapshots_after_initialization_and_tool_batch(
    tmp_path: Path,
) -> None:
    class Provider(LLM):
        def __init__(self) -> None:
            self.turns = deque(
                [
                    ModelTurn(
                        content="list",
                        tool_calls=(ToolCall("call-1", "list_files", {}),),
                        finish_reason="tool_calls",
                    ),
                    ModelTurn(content="done", finish_reason="completed"),
                ]
            )

        def generate(self, request: ModelRequest) -> ModelTurn:
            return self.turns.popleft()

    event_store = SQLiteEventStore(tmp_path / "runs.db")
    snapshot_store = ArtifactSnapshotStore(
        FileArtifactStore(tmp_path / "artifacts")
    )
    try:
        result = Agent(
            Provider(),
            create_default_registry(tmp_path),
            event_store=event_store,
            snapshot_store=snapshot_store,
        ).run("list files", trace=False)
        assert result.run_id is not None
        events = event_store.load(result.run_id)
    finally:
        event_store.close()

    snapshot_events = [
        event for event in events if event.event_type == "snapshot_saved"
    ]
    assert len(snapshot_events) >= 3
    latest = snapshot_events[-1]
    assert latest.payload["last_event_sequence"] == latest.sequence - 1
    assert list((tmp_path / "artifacts").rglob("*"))


def _make_interrupted_run(
    tmp_path: Path,
    *,
    include_started_write: bool = False,
    complete: bool = False,
) -> tuple[SQLiteEventStore, ArtifactSnapshotStore, str]:
    event_store = SQLiteEventStore(tmp_path / "runs.db")
    artifact_store = ArtifactSnapshotStore(
        FileArtifactStore(tmp_path / "artifacts")
    )
    registry = create_default_registry(tmp_path)
    run_id = "run-analysis"
    event_store.append(
        run_id,
        "run_started",
        {"task": {"sha256": "task"}, "max_steps": 6},
    )
    snapshot = RunSnapshot(
        run_id=run_id,
        snapshot_version=1,
        last_event_sequence=1,
        next_step=2,
        resume_point="model_request",
        messages=[Message("user", "task")],
        seen_call_ids=[],
        verification={
            "mutation_version": 0,
            "diff_version": 0,
            "tests_version": 0,
        },
        max_steps=6,
        require_verification=True,
        llm_identity="fake.provider",
        tool_schema_sha256=tool_schema_fingerprint(registry.list_tools()),
        workspace=workspace_identity(tmp_path),
    )
    ref = artifact_store.save(snapshot)
    event_store.append(
        run_id,
        "snapshot_saved",
        {
            "snapshot_id": "run-analysis:snapshot:1",
            "artifact_id": ref.artifact_id,
            "last_event_sequence": 1,
            "sha256": ref.sha256,
            "bytes": ref.bytes,
            "media_type": ref.media_type,
        },
    )
    if include_started_write:
        event_store.append(
            run_id,
            "tool_execution_started",
            {
                "call_id": "patch-1",
                "tool": "apply_patch",
                "permission": "write",
                "effect": "workspace_mutation",
                "arguments": {},
            },
        )
    if complete:
        event_store.append(
            run_id,
            "run_completed",
            {"steps": 2, "finish_reason": "completed"},
        )
    return event_store, artifact_store, run_id


def test_resume_analyzer_allows_clean_interrupted_run_without_provider_call(
    tmp_path: Path,
) -> None:
    event_store, snapshot_store, run_id = _make_interrupted_run(tmp_path)
    try:
        registry = create_default_registry(tmp_path)
        analyzer = ResumeAnalyzer(
            event_store,
            snapshot_store,
            registry,
            llm_identity="fake.provider",
        )
        result = analyzer.analyze(run_id)
    finally:
        event_store.close()

    assert result.decision == "resume_allowed"
    assert result.resume_from_step == 2
    assert result.blockers == ()


def test_resume_analyzer_blocks_started_write_as_manual_reconciliation(
    tmp_path: Path,
) -> None:
    event_store, snapshot_store, run_id = _make_interrupted_run(
        tmp_path,
        include_started_write=True,
    )
    try:
        result = ResumeAnalyzer(
            event_store,
            snapshot_store,
            create_default_registry(tmp_path),
            llm_identity="fake.provider",
        ).analyze(run_id)
    finally:
        event_store.close()

    assert result.decision == "manual_reconciliation_required"
    assert "manual_reconciliation_required" in result.blockers


def test_resume_analyzer_reports_completed_run_and_schema_change(
    tmp_path: Path,
) -> None:
    event_store, snapshot_store, run_id = _make_interrupted_run(tmp_path, complete=True)
    try:
        completed = ResumeAnalyzer(
            event_store,
            snapshot_store,
            create_default_registry(tmp_path),
            llm_identity="fake.provider",
        ).analyze(run_id)
        assert completed.decision == "already_completed"

        registry = create_default_registry(tmp_path)
        tool = registry.get("list_files")
        assert tool is not None
        registry.register(
            Tool(
                name=tool.name,
                description="changed schema description",
                parameters=tool.parameters,
                fn=tool.fn,
                permission=tool.permission,
                effect=tool.effect,
            )
        )
        changed = ResumeAnalyzer(
            event_store,
            snapshot_store,
            registry,
            llm_identity="fake.provider",
        ).analyze(run_id)
    finally:
        event_store.close()

    assert changed.decision == "already_completed"


def test_resume_analyzer_blocks_schema_change_for_interrupted_run(
    tmp_path: Path,
) -> None:
    event_store, snapshot_store, run_id = _make_interrupted_run(tmp_path)
    try:
        registry = create_default_registry(tmp_path)
        tool = registry.get("list_files")
        assert tool is not None
        registry.register(
            Tool(
                name=tool.name,
                description="changed schema description",
                parameters=tool.parameters,
                fn=tool.fn,
                permission=tool.permission,
                effect=tool.effect,
            )
        )
        result = ResumeAnalyzer(
            event_store,
            snapshot_store,
            registry,
            llm_identity="fake.provider",
        ).analyze(run_id)
    finally:
        event_store.close()

    assert result.decision == "resume_blocked"
    assert "tool_schema_changed" in result.blockers


def test_resume_analyzer_blocks_journal_sequence_gap(tmp_path: Path) -> None:
    event_store, snapshot_store, run_id = _make_interrupted_run(tmp_path)
    try:
        event_store._connection.execute(
            "UPDATE run_events SET sequence = 4 WHERE run_id = ? AND sequence = 2",
            (run_id,),
        )
        event_store._connection.commit()
        result = ResumeAnalyzer(
            event_store,
            snapshot_store,
            create_default_registry(tmp_path),
            llm_identity="fake.provider",
        ).analyze(run_id)
    finally:
        event_store.close()

    assert result.decision == "resume_blocked"
    assert "journal_sequence_gap" in result.blockers
