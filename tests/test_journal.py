from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from mini_agent.agent import Agent
from mini_agent.__main__ import main
from mini_agent.journal import (
    EventType,
    RunEvent,
    SQLiteEventStore,
    inspect_run,
)
from mini_agent.llm import LLM
from mini_agent.provider_errors import ProviderTimeoutError
from mini_agent.retry import ProviderRetryPolicy
from mini_agent.schema import ModelRequest, ModelTurn, ToolCall
from mini_agent.tools import create_default_registry


class ScriptedProvider(LLM):
    def __init__(self, outcomes: list[ModelTurn | Exception]) -> None:
        self.outcomes = deque(outcomes)

    def generate(self, request: ModelRequest) -> ModelTurn:
        if not self.outcomes:
            raise AssertionError("the Agent requested more turns than scripted")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FailOnToolStartStore:
    """Simulate durable storage failure at the last boundary before a side effect."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def append(
        self,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> RunEvent:
        if event_type == "tool_execution_started":
            raise RuntimeError("journal unavailable")
        event = RunEvent(
            event_id=f"event-{len(self.events) + 1}",
            run_id=run_id,
            sequence=len(self.events) + 1,
            event_type=event_type,
            schema_version=1,
            created_at="2026-01-01T00:00:00+00:00",
            payload=payload,
        )
        self.events.append(event)
        return event

    def load(self, run_id: str) -> list[RunEvent]:
        return [event for event in self.events if event.run_id == run_id]


def test_sqlite_event_store_persists_order_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with SQLiteEventStore(path) as store:
        first = store.append(
            "run-1",
            "run_started",
            {"task": {"sha256": "abc"}, "max_steps": 6},
        )
        second = store.append(
            "run-1",
            "run_completed",
            {"steps": 1, "finish_reason": "completed"},
        )

    with SQLiteEventStore(path) as reopened:
        events = reopened.load("run-1")

    assert first.sequence == 1
    assert second.sequence == 2
    assert [event.sequence for event in events] == [1, 2]
    assert [event.event_type for event in events] == [
        "run_started",
        "run_completed",
    ]


def test_event_store_rejects_unknown_or_incomplete_event_schema(tmp_path: Path) -> None:
    with SQLiteEventStore(tmp_path / "runs.db") as store:
        with pytest.raises(ValueError, match="unsupported event_type"):
            store.append("run-1", "future_event", {})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="missing required payload keys"):
            store.append("run-1", "run_completed", {"steps": 1})


def test_event_store_revalidates_persisted_rows_on_load(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with SQLiteEventStore(path) as store:
        store.append(
            "run-1",
            "run_started",
            {"task": {"sha256": "abc"}, "max_steps": 6},
        )
        store.append(
            "run-2",
            "run_started",
            {"task": {"sha256": "def"}, "max_steps": 6},
        )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE run_events SET event_type = 'corrupted_event' WHERE run_id = ?",
            ("run-1",),
        )
        connection.execute(
            "UPDATE run_events SET schema_version = 99 WHERE run_id = ?",
            ("run-2",),
        )

    with SQLiteEventStore(path) as reopened:
        with pytest.raises(ValueError, match="unsupported event_type"):
            reopened.load("run-1")
        with pytest.raises(ValueError, match="unsupported event schema_version"):
            reopened.load("run-2")


def test_event_store_redacts_sensitive_keys_and_credential_shapes(
    tmp_path: Path,
) -> None:
    with SQLiteEventStore(tmp_path / "runs.db") as store:
        store.append(
            "run-1",
            "run_started",
            {
                "task": {"sha256": "abc"},
                "max_steps": 6,
                "headers": {"Authorization": "Bearer top-secret"},
                "api-key": "secret-value",
                "usage": {"input_tokens": 12},
                "diagnostic": "provider returned sk-abcdefgh12345678",
            },
        )
        payload = store.load("run-1")[0].payload

    assert payload["headers"]["Authorization"] == "[REDACTED]"
    assert payload["api-key"] == "[REDACTED]"
    assert payload["usage"]["input_tokens"] == 12
    assert payload["diagnostic"] == "provider returned [REDACTED]"


def test_agent_records_a_complete_tool_timeline_without_raw_content(
    tmp_path: Path,
) -> None:
    secret_file_content = "private-file-contents"
    secret_task = "read note.txt and remember private-task-marker"
    (tmp_path / "note.txt").write_text(secret_file_content, encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="I will read it.",
                tool_calls=(
                    ToolCall("read-1", "read_file", {"path": "note.txt"}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="finished", finish_reason="completed"),
        ]
    )
    with SQLiteEventStore(tmp_path / "runs.db") as store:
        result = Agent(
            provider,
            create_default_registry(tmp_path),
            event_store=store,
        ).run(secret_task, trace=False)
        assert result.run_id is not None
        events = store.load(result.run_id)

    assert [event.event_type for event in events] == [
        "run_started",
        "model_request_started",
        "model_request_completed",
        "permission_decided",
        "tool_execution_started",
        "tool_execution_completed",
        "model_request_started",
        "model_request_completed",
        "run_completed",
    ]
    serialized = json.dumps([event.payload for event in events], ensure_ascii=False)
    assert secret_file_content not in serialized
    assert "private-task-marker" not in serialized
    assert inspect_run(events).status == "completed"


def test_agent_journal_records_provider_attempts(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ProviderTimeoutError("temporary"),
            ModelTurn(content="done", finish_reason="completed"),
        ]
    )
    with SQLiteEventStore(tmp_path / "runs.db") as store:
        result = Agent(
            provider,
            create_default_registry(tmp_path),
            retry_policy=ProviderRetryPolicy(
                max_attempts=2,
                sleeper=lambda _: None,
            ),
            event_store=store,
        ).run("finish", trace=False)
        assert result.run_id is not None
        events = store.load(result.run_id)

    failed = next(
        event for event in events if event.event_type == "model_request_failed"
    )
    completed = next(
        event for event in events if event.event_type == "model_request_completed"
    )
    assert failed.payload["attempt"] == 1
    assert failed.payload["retrying"] is True
    assert completed.payload["attempt"] == 2


def test_journal_failure_before_tool_start_prevents_side_effect(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="list",
                tool_calls=(ToolCall("list-1", "list_files", {}),),
                finish_reason="tool_calls",
            )
        ]
    )
    registry = create_default_registry(tmp_path)
    tool = registry.get("list_files")
    assert tool is not None
    executions = 0

    def counted(args: dict[str, object]) -> str:
        nonlocal executions
        executions += 1
        return tool.fn(args)

    registry.register(
        type(tool)(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            fn=counted,
            permission=tool.permission,
            effect=tool.effect,
        )
    )

    with pytest.raises(RuntimeError, match="journal unavailable"):
        Agent(
            provider,
            registry,
            event_store=FailOnToolStartStore(),
        ).run("list files", trace=False)

    assert executions == 0


def test_inspection_classifies_incomplete_tool_side_effect_risk(tmp_path: Path) -> None:
    with SQLiteEventStore(tmp_path / "runs.db") as store:
        store.append(
            "run-1",
            "run_started",
            {"task": {"sha256": "abc"}, "max_steps": 6},
        )
        store.append(
            "run-1",
            "tool_execution_started",
            {
                "call_id": "read-1",
                "tool": "read_file",
                "permission": "read",
                "effect": "none",
                "arguments": {},
            },
        )
        store.append(
            "run-1",
            "tool_execution_started",
            {
                "call_id": "patch-1",
                "tool": "apply_patch",
                "permission": "write",
                "effect": "workspace_mutation",
                "arguments": {},
            },
        )
        inspection = inspect_run(store.load("run-1"))

    assert inspection.status == "interrupted"
    assert [item.disposition for item in inspection.incomplete_tools] == [
        "safe_to_retry",
        "unknown_outcome",
    ]


def test_cli_creates_and_inspects_a_journaled_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["mini-agent", "list files", "--no-trace"])

    main()

    first_output = capsys.readouterr().out
    matched = re.search(r"^run-[0-9a-f]+$", first_output, flags=re.MULTILINE)
    assert matched is not None
    run_id = matched.group(0)
    assert ".mini_agent" not in first_output
    assert (tmp_path / ".mini_agent" / "runs.db").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["mini-agent", "--inspect-run", run_id],
    )
    main()

    inspection_output = capsys.readouterr().out
    assert '"status": "completed"' in inspection_output
    assert "tool_execution_started" in inspection_output
    assert "tool_execution_completed" in inspection_output
