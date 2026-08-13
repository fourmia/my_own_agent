from __future__ import annotations

from collections import deque
from pathlib import Path
import re
import sys

import pytest

from mini_agent.agent import Agent
from mini_agent.__main__ import main
from mini_agent.artifacts import FileArtifactStore
from mini_agent.journal import SQLiteEventStore
from mini_agent.llm import LLM, MockLLM
from mini_agent.permissions import FullAccessPermissionPolicy
from mini_agent.resume import ResumeExecutor, ResumeRejectedError
from mini_agent.schema import Message, ModelRequest, ModelTurn, ToolCall
from mini_agent.snapshots import (
    ArtifactSnapshotStore,
    RunSnapshot,
    SNAPSHOT_SCHEMA_VERSION,
    tool_schema_fingerprint,
    workspace_identity,
)
from mini_agent.tools import create_default_registry


class ScriptedProvider(LLM):
    """记录恢复后的请求，确保测试能证明历史来自 Snapshot。"""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = deque(turns)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.turns:
            raise AssertionError("unexpected Provider request")
        return self.turns.popleft()


def _recovery_runtime(
    tmp_path: Path,
    provider: LLM,
) -> tuple[SQLiteEventStore, ArtifactSnapshotStore, Agent]:
    event_store = SQLiteEventStore(tmp_path / "runs.db")
    snapshot_store = ArtifactSnapshotStore(
        FileArtifactStore(tmp_path / "artifacts")
    )
    agent = Agent(
        provider,
        create_default_registry(tmp_path),
        event_store=event_store,
        snapshot_store=snapshot_store,
    )
    return event_store, snapshot_store, agent


def _persist_interrupted_snapshot(
    event_store: SQLiteEventStore,
    snapshot_store: ArtifactSnapshotStore,
    agent: Agent,
    *,
    resume_point: str,
    messages: list[Message],
    seen_call_ids: list[str] | None = None,
) -> str:
    run_id = "run-resume-execution"
    started = event_store.append(
        run_id,
        "run_started",
        {"task": {"sha256": "task"}, "max_steps": agent.max_steps},
    )
    snapshot = RunSnapshot(
        run_id=run_id,
        snapshot_version=SNAPSHOT_SCHEMA_VERSION,
        last_event_sequence=started.sequence,
        next_step=2,
        resume_point=resume_point,
        messages=messages,
        seen_call_ids=seen_call_ids or [],
        verification={
            "mutation_version": 0,
            "diff_version": 0,
            "tests_version": 0,
        },
        max_steps=agent.max_steps,
        require_verification=agent.require_verification,
        llm_identity=agent.llm_identity(),
        tool_schema_sha256=tool_schema_fingerprint(agent.tools.list_tools()),
        workspace=workspace_identity(agent.tools.root),
        permission_policy_identity=agent.permission_policy_identity(),
    )
    ref = snapshot_store.save(snapshot)
    event_store.append(
        run_id,
        "snapshot_saved",
        {
            "snapshot_id": f"{run_id}:snapshot:{started.sequence}",
            "artifact_id": ref.artifact_id,
            "last_event_sequence": started.sequence,
            "sha256": ref.sha256,
            "bytes": ref.bytes,
            "media_type": ref.media_type,
        },
    )
    return run_id


def test_resume_model_request_reuses_messages_and_original_run_id(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [ModelTurn(content="recovered answer", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(tmp_path, provider)
    try:
        messages = [Message("user", "persisted task"), Message("tool", "evidence")]
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="model_request",
            messages=messages,
        )

        result = ResumeExecutor(agent).run(run_id, trace=False)
        events = event_store.load(run_id)
    finally:
        event_store.close()

    assert result.run_id == run_id
    assert result.answer == "recovered answer"
    assert provider.requests[0].messages == messages
    assert [event.event_type for event in events].count("run_started") == 1
    assert [event.event_type for event in events].count("run_resumed") == 1
    assert events[-1].event_type == "run_completed"


def test_resume_tool_batch_rechecks_permission_and_executes_only_read_call(
    tmp_path: Path,
) -> None:
    (tmp_path / "note.txt").write_text("persisted evidence", encoding="utf-8")
    provider = ScriptedProvider(
        [ModelTurn(content="read completed", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(tmp_path, provider)
    call = ToolCall("read-1", "read_file", {"path": "note.txt"})
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="tool_batch",
            messages=[
                Message("user", "read note"),
                Message("assistant", "reading", tool_calls=(call,)),
            ],
            seen_call_ids=[call.call_id],
        )

        result = ResumeExecutor(agent).run(run_id, trace=False)
        events = event_store.load(run_id)
    finally:
        event_store.close()

    permission = next(
        event for event in events if event.event_type == "permission_decided"
    )
    assert permission.payload["call_id"] == "read-1"
    assert permission.payload["outcome"] == "allow"
    assert provider.requests[0].messages[-1].content == "persisted evidence"
    assert result.answer == "read completed"


def test_resume_retries_started_only_read_but_never_pending_write(
    tmp_path: Path,
) -> None:
    (tmp_path / "note.txt").write_text("safe read", encoding="utf-8")
    provider = ScriptedProvider(
        [ModelTurn(content="done", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(tmp_path, provider)
    read_call = ToolCall("read-1", "read_file", {"path": "note.txt"})
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="tool_batch",
            messages=[
                Message("user", "read note"),
                Message("assistant", "reading", tool_calls=(read_call,)),
            ],
            seen_call_ids=[read_call.call_id],
        )
        event_store.append(
            run_id,
            "tool_execution_started",
            {
                "call_id": read_call.call_id,
                "tool": read_call.name,
                "permission": "read",
                "effect": "none",
                "arguments": {},
            },
        )

        result = ResumeExecutor(agent).run(run_id, trace=False)
        events = event_store.load(run_id)
    finally:
        event_store.close()

    assert result.answer == "done"
    assert [event.event_type for event in events].count("tool_execution_started") == 2
    assert [event.event_type for event in events].count("tool_execution_completed") == 1

    write_provider = ScriptedProvider(
        [ModelTurn(content="must not be requested", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(
        tmp_path / "write-case",
        write_provider,
    )
    write_call = ToolCall(
        "write-1",
        "write_file",
        {"path": "unsafe.txt", "content": "do not write"},
    )
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="tool_batch",
            messages=[
                Message("user", "write"),
                Message("assistant", "writing", tool_calls=(write_call,)),
            ],
            seen_call_ids=[write_call.call_id],
        )
        with pytest.raises(ResumeRejectedError) as captured:
            ResumeExecutor(agent).run(run_id, trace=False)
    finally:
        event_store.close()

    assert "pending_non_read_tool" in captured.value.analysis.blockers
    assert not (tmp_path / "write-case" / "unsafe.txt").exists()
    assert write_provider.requests == []


def test_resume_blocks_unsnapshotted_result_and_completed_run(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [ModelTurn(content="must not run", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(tmp_path, provider)
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="model_request",
            messages=[Message("user", "task")],
        )
        event_store.append(
            run_id,
            "tool_execution_completed",
            {"call_id": "read-1", "tool": "read_file", "ok": True},
        )
        with pytest.raises(ResumeRejectedError) as captured:
            ResumeExecutor(agent).run(run_id, trace=False)
        assert "unsnapshotted_tool_result" in captured.value.analysis.blockers

        event_store.append(
            run_id,
            "run_completed",
            {"steps": 2, "finish_reason": "completed"},
        )
        with pytest.raises(ResumeRejectedError) as completed:
            ResumeExecutor(agent).run(run_id, trace=False)
    finally:
        event_store.close()

    assert completed.value.analysis.decision == "already_completed"
    assert provider.requests == []


def test_resume_blocks_unsnapshotted_skipped_result_and_policy_change(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [ModelTurn(content="must not run", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(tmp_path, provider)
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="model_request",
            messages=[Message("user", "task")],
        )
        event_store.append(
            run_id,
            "tool_execution_skipped",
            {
                "call_id": "read-1",
                "tool": "read_file",
                "error_code": "permission_denied",
            },
        )
        with pytest.raises(ResumeRejectedError) as skipped:
            ResumeExecutor(agent).run(run_id, trace=False)
        assert "unsnapshotted_tool_result" in skipped.value.analysis.blockers
    finally:
        event_store.close()

    policy_root = tmp_path / "policy-case"
    provider = ScriptedProvider(
        [ModelTurn(content="must not run", finish_reason="completed")]
    )
    event_store, snapshot_store, original_agent = _recovery_runtime(
        policy_root,
        provider,
    )
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            original_agent,
            resume_point="model_request",
            messages=[Message("user", "task")],
        )
        changed_agent = Agent(
            provider,
            create_default_registry(policy_root),
            permission_policy=FullAccessPermissionPolicy(),
            event_store=event_store,
            snapshot_store=snapshot_store,
        )
        with pytest.raises(ResumeRejectedError) as changed:
            ResumeExecutor(changed_agent).run(run_id, trace=False)
    finally:
        event_store.close()

    assert "permission_policy_changed" in changed.value.analysis.blockers
    assert provider.requests == []


def test_resume_rechecks_workspace_after_claim_before_provider_call(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider(
        [ModelTurn(content="must not run", finish_reason="completed")]
    )
    event_store, snapshot_store, agent = _recovery_runtime(tmp_path, provider)
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="model_request",
            messages=[Message("user", "task")],
        )
        (tmp_path / "changed.txt").write_text("changed", encoding="utf-8")
        with pytest.raises(ResumeRejectedError) as rejected:
            ResumeExecutor(agent).run(run_id, trace=False)
    finally:
        event_store.close()

    assert provider.requests == []
    assert "workspace_state_changed" in rejected.value.analysis.blockers


def test_resume_claim_is_atomic(tmp_path: Path) -> None:
    with SQLiteEventStore(tmp_path / "runs.db") as event_store:
        event_store.append(
            "run-claim",
            "run_started",
            {"task": {"sha256": "task"}, "max_steps": 6},
        )
        payload = {
            "from_snapshot_sequence": 1,
            "previous_journal_sequence": 1,
            "resume_point": "model_request",
            "next_step": 2,
        }
        claimed = event_store.claim_resume("run-claim", 1, payload)
        with pytest.raises(RuntimeError, match="claim conflict"):
            event_store.claim_resume("run-claim", 1, payload)

    assert claimed.sequence == 2


def test_cli_resume_run_completes_a_persisted_mock_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime_root = tmp_path / ".mini_agent"
    event_store = SQLiteEventStore(runtime_root / "runs.db")
    snapshot_store = ArtifactSnapshotStore(
        FileArtifactStore(runtime_root / "artifacts")
    )
    agent = Agent(
        MockLLM(),
        create_default_registry(tmp_path),
        permission_policy=FullAccessPermissionPolicy(),
        event_store=event_store,
        snapshot_store=snapshot_store,
    )
    try:
        run_id = _persist_interrupted_snapshot(
            event_store,
            snapshot_store,
            agent,
            resume_point="model_request",
            messages=[Message("user", "list files")],
        )
    finally:
        event_store.close()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mini-agent", "--resume-run", run_id, "--no-trace"],
    )
    main()

    output = capsys.readouterr().out
    assert re.search(rf"^{re.escape(run_id)}$", output, flags=re.MULTILINE)
    assert "[final]" in output
    with SQLiteEventStore(runtime_root / "runs.db") as reopened:
        events = reopened.load(run_id)
    assert any(event.event_type == "run_resumed" for event in events)
    assert events[-1].event_type == "run_completed"
