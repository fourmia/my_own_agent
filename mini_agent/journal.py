from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, cast


EventType = Literal[
    "run_started",
    "model_request_started",
    "model_request_failed",
    "model_request_completed",
    "permission_decided",
    "tool_execution_started",
    "tool_execution_completed",
    "tool_execution_skipped",
    "tool_execution_reconciled",
    "verification_updated",
    "completion_blocked",
    "snapshot_saved",
    "run_resumed",
    "run_completed",
    "run_failed",
]

RecoveryDisposition = Literal["safe_to_retry", "unknown_outcome"]

_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "cookie",
    "password",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
    "token",
}
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_REQUIRED_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "run_started": frozenset({"task", "max_steps"}),
    "model_request_started": frozenset({"request_id", "step"}),
    "model_request_failed": frozenset(
        {"request_id", "step", "attempt", "error_code", "retrying"}
    ),
    "model_request_completed": frozenset(
        {"request_id", "step", "attempt", "finish_reason"}
    ),
    "permission_decided": frozenset(
        {"call_id", "tool", "permission", "outcome"}
    ),
    "tool_execution_started": frozenset(
        {"call_id", "tool", "permission", "effect", "arguments"}
    ),
    "tool_execution_completed": frozenset({"call_id", "tool", "ok"}),
    "tool_execution_skipped": frozenset({"call_id", "tool", "error_code"}),
    "tool_execution_reconciled": frozenset({"call_id", "tool", "status"}),
    "verification_updated": frozenset({"call_id", "tool", "state"}),
    "completion_blocked": frozenset({"step", "required_tools", "verification"}),
    "snapshot_saved": frozenset(
        {"snapshot_id", "artifact_id", "last_event_sequence", "sha256"}
    ),
    "run_resumed": frozenset(
        {
            "from_snapshot_sequence",
            "previous_journal_sequence",
            "resume_point",
            "next_step",
        }
    ),
    "run_completed": frozenset({"steps", "finish_reason"}),
    "run_failed": frozenset({"exception_type"}),
}


@dataclass(frozen=True)
class RunEvent:
    """一条已经持久化、可按 run_id 和 sequence 确定重放顺序的事件。"""

    event_id: str
    run_id: str
    sequence: int
    event_type: EventType
    schema_version: int
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class IncompleteToolExecution:
    """进程中断后只有 started、没有 completed 的工具执行。"""

    call_id: str
    tool_name: str
    permission: str
    effect: str
    started_sequence: int
    disposition: RecoveryDisposition


@dataclass(frozen=True)
class RunInspection:
    run_id: str
    status: Literal["completed", "failed", "interrupted"]
    event_count: int
    last_sequence: int
    incomplete_tools: tuple[IncompleteToolExecution, ...]


class EventStore(Protocol):
    """Runtime 依赖的最小 append-only 事件存储合同。"""

    def append(
        self,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> RunEvent:
        """原子分配下一个 sequence 并持久化事件。"""

    def load(self, run_id: str) -> list[RunEvent]:
        """按 sequence 升序读取一次运行的全部事件。"""

    def claim_resume(
        self,
        run_id: str,
        expected_sequence: int,
        payload: dict[str, Any],
    ) -> RunEvent:
        """仅当 Journal 未变化时原子追加 run_resumed，防止并发重复恢复。"""


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def content_summary(content: str) -> dict[str, Any]:
    """只记录内容长度与摘要，避免日志复制 Prompt、代码和工具输出。"""

    encoded = content.encode("utf-8")
    return {
        "characters": len(content),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def sanitize_payload(value: Any) -> Any:
    """在持久化边界递归清理常见凭据字段和凭据形态。"""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
        return _OPENAI_KEY_PATTERN.sub("[REDACTED]", redacted)
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.lower().replace("-", "_")
            cleaned[key] = (
                "[REDACTED]"
                if normalized in _SENSITIVE_KEYS
                else sanitize_payload(item)
            )
        return cleaned
    raise TypeError(f"event payload value is not JSON-compatible: {type(value).__name__}")


def _validate_event_payload(event_type: str, payload: dict[str, Any]) -> EventType:
    required_keys = _REQUIRED_PAYLOAD_KEYS.get(event_type)
    if required_keys is None:
        raise ValueError(f"unsupported event_type: {event_type}")
    missing_keys = sorted(required_keys.difference(payload))
    if missing_keys:
        raise ValueError(
            f"event {event_type} is missing required payload keys: {missing_keys}"
        )
    if event_type == "tool_execution_reconciled" and payload.get("status") not in {
        "executed",
        "not_executed",
    }:
        raise ValueError("tool_execution_reconciled has an unsupported status")
    return cast(EventType, event_type)


class SQLiteEventStore:
    """基于 SQLite 的单进程 append-only EventStore。

    sequence 分配和 INSERT 位于同一 ``BEGIN IMMEDIATE`` 事务中；即使未来多个
    Agent 线程共享这个实例，同一个 run_id 也不会获得重复 sequence。
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=5,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (run_id, sequence)
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_run_events_run_sequence
                ON run_events (run_id, sequence)
                """
            )

    def append(
        self,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> RunEvent:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        event_type = _validate_event_payload(event_type, payload)
        cleaned = sanitize_payload(payload)
        if not isinstance(cleaned, dict):
            raise TypeError("event payload must be an object")
        payload_json = json.dumps(
            cleaned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        event_id = f"evt-{uuid.uuid4().hex}"
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                    FROM run_events
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                sequence = int(row["next_sequence"])
                cursor.execute(
                    """
                    INSERT INTO run_events (
                        event_id,
                        run_id,
                        sequence,
                        event_type,
                        schema_version,
                        created_at,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        run_id,
                        sequence,
                        event_type,
                        1,
                        created_at,
                        payload_json,
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

        return RunEvent(
            event_id=event_id,
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            schema_version=1,
            created_at=created_at,
            payload=cleaned,
        )

    def claim_resume(
        self,
        run_id: str,
        expected_sequence: int,
        payload: dict[str, Any],
    ) -> RunEvent:
        """以 compare-and-append 方式取得一次恢复执行权。"""

        event_type = _validate_event_payload("run_resumed", payload)
        cleaned = sanitize_payload(payload)
        if not isinstance(cleaned, dict):
            raise TypeError("event payload must be an object")
        payload_json = json.dumps(
            cleaned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        event_id = f"evt-{uuid.uuid4().hex}"
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) AS latest_sequence
                    FROM run_events
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                latest_sequence = int(row["latest_sequence"])
                if latest_sequence != expected_sequence:
                    raise RuntimeError(
                        "resume claim conflict: journal changed after analysis"
                    )
                sequence = latest_sequence + 1
                cursor.execute(
                    """
                    INSERT INTO run_events (
                        event_id, run_id, sequence, event_type,
                        schema_version, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        run_id,
                        sequence,
                        event_type,
                        1,
                        created_at,
                        payload_json,
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

        return RunEvent(
            event_id=event_id,
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            schema_version=1,
            created_at=created_at,
            payload=cleaned,
        )

    def load(self, run_id: str) -> list[RunEvent]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_id, run_id, sequence, event_type, schema_version,
                       created_at, payload_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_run_ids(self, limit: int = 20) -> list[str]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT run_id, MAX(created_at) AS last_created_at
                FROM run_events
                GROUP BY run_id
                ORDER BY last_created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row["run_id"]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteEventStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> RunEvent:
        schema_version = int(row["schema_version"])
        if schema_version != 1:
            raise ValueError(f"unsupported event schema_version: {schema_version}")
        event_type = str(row["event_type"])
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise ValueError("persisted event payload must be a JSON object")
        event_type = _validate_event_payload(event_type, payload)
        return RunEvent(
            event_id=str(row["event_id"]),
            run_id=str(row["run_id"]),
            sequence=int(row["sequence"]),
            event_type=event_type,
            schema_version=schema_version,
            created_at=str(row["created_at"]),
            payload=payload,
        )


def inspect_run(events: list[RunEvent]) -> RunInspection:
    if not events:
        raise ValueError("cannot inspect an empty run")
    run_id = events[0].run_id
    if any(event.run_id != run_id for event in events):
        raise ValueError("all events must belong to one run")

    pending_starts: dict[str, RunEvent] = {}
    for event in events:
        call_id = str(event.payload.get("call_id", ""))
        if event.event_type == "tool_execution_started":
            pending_starts[call_id] = event
        elif event.event_type in {
            "tool_execution_completed",
            "tool_execution_reconciled",
        }:
            pending_starts.pop(call_id, None)

    incomplete: list[IncompleteToolExecution] = []
    for call_id, event in pending_starts.items():
        permission = str(event.payload.get("permission", "unknown"))
        effect = str(event.payload.get("effect", "none"))
        disposition: RecoveryDisposition = (
            "safe_to_retry"
            if permission == "read" and effect != "workspace_mutation"
            else "unknown_outcome"
        )
        incomplete.append(
            IncompleteToolExecution(
                call_id=call_id,
                tool_name=str(event.payload.get("tool", "unknown")),
                permission=permission,
                effect=effect,
                started_sequence=event.sequence,
                disposition=disposition,
            )
        )

    status: Literal["completed", "failed", "interrupted"] = "interrupted"
    for event in events:
        if event.event_type == "run_completed":
            status = "completed"
        elif event.event_type == "run_failed":
            status = "failed"
    return RunInspection(
        run_id=run_id,
        status=status,
        event_count=len(events),
        last_sequence=events[-1].sequence,
        incomplete_tools=tuple(incomplete),
    )


__all__ = [
    "EventStore",
    "EventType",
    "IncompleteToolExecution",
    "RunEvent",
    "RunInspection",
    "SQLiteEventStore",
    "content_summary",
    "inspect_run",
    "new_run_id",
    "sanitize_payload",
]
