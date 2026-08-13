from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from mini_agent.artifacts import ArtifactRef, ArtifactStore
from mini_agent.schema import Message, ToolCall, ToolResult
from mini_agent.verification import VerificationState


SNAPSHOT_SCHEMA_VERSION = 1
ResumePoint = Literal["model_request", "tool_batch", "run_completion"]
_RESUME_POINTS = frozenset({"model_request", "tool_batch", "run_completion"})
_MESSAGE_ROLES = frozenset({"user", "assistant", "tool", "system"})


@dataclass(frozen=True)
class WorkspaceIdentity:
    root: str
    git_head: str | None
    state_sha256: str


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    snapshot_version: int
    last_event_sequence: int
    next_step: int
    resume_point: ResumePoint
    messages: list[Message]
    seen_call_ids: list[str]
    verification: dict[str, int]
    max_steps: int
    require_verification: bool
    llm_identity: str
    tool_schema_sha256: str
    workspace: WorkspaceIdentity
    # 5B1 生成的 v1 Snapshot 没有该字段；它们可继续分析，但不能自动恢复。
    permission_policy_identity: str | None = None


class SnapshotStore(Protocol):
    def save(self, snapshot: RunSnapshot) -> ArtifactRef: ...

    def load(self, ref: ArtifactRef) -> RunSnapshot: ...


def _tool_call_to_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "call_id": call.call_id,
        "name": call.name,
        "arguments": call.arguments,
        "raw_arguments": call.raw_arguments,
        "argument_error": call.argument_error,
    }


def _tool_call_from_dict(value: dict[str, Any]) -> ToolCall:
    return ToolCall(
        call_id=str(value["call_id"]),
        name=str(value["name"]),
        arguments=dict(value["arguments"]),
        raw_arguments=value.get("raw_arguments"),
        argument_error=value.get("argument_error"),
    )


def message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [_tool_call_to_dict(call) for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
    }


def message_from_dict(value: dict[str, Any]) -> Message:
    if not isinstance(value, dict):
        raise TypeError("snapshot message must be an object")
    role = value.get("role")
    if role not in _MESSAGE_ROLES:
        raise ValueError(f"unsupported snapshot message role: {role}")
    content = value.get("content")
    if not isinstance(content, str):
        raise TypeError("snapshot message content must be a string")
    tool_calls = value.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise TypeError("snapshot message tool_calls must be an array")
    return Message(
        role=role,
        content=content,
        tool_calls=tuple(
            _tool_call_from_dict(call) for call in tool_calls
        ),
        tool_call_id=value.get("tool_call_id"),
        tool_name=value.get("tool_name"),
    )


def snapshot_to_bytes(snapshot: RunSnapshot) -> bytes:
    payload = {
        "snapshot_version": snapshot.snapshot_version,
        "run_id": snapshot.run_id,
        "last_event_sequence": snapshot.last_event_sequence,
        "next_step": snapshot.next_step,
        "resume_point": snapshot.resume_point,
        "messages": [message_to_dict(message) for message in snapshot.messages],
        "seen_call_ids": snapshot.seen_call_ids,
        "verification": snapshot.verification,
        "max_steps": snapshot.max_steps,
        "require_verification": snapshot.require_verification,
        "llm_identity": snapshot.llm_identity,
        "tool_schema_sha256": snapshot.tool_schema_sha256,
        "workspace": asdict(snapshot.workspace),
        "permission_policy_identity": snapshot.permission_policy_identity,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_from_bytes(content: bytes) -> RunSnapshot:
    value = json.loads(content.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("snapshot payload must be an object")
    if value.get("snapshot_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot_version: {value.get('snapshot_version')}")
    resume_point = value.get("resume_point")
    if resume_point not in _RESUME_POINTS:
        raise ValueError(f"unsupported resume_point: {resume_point}")
    max_steps = int(value["max_steps"])
    next_step = int(value["next_step"])
    if max_steps <= 0:
        raise ValueError("snapshot max_steps must be positive")
    if next_step <= 0 or next_step > max_steps + 1:
        raise ValueError("snapshot next_step is outside the run step range")
    if resume_point in {"tool_batch", "run_completion"} and next_step > max_steps:
        raise ValueError(f"snapshot {resume_point} step exceeds max_steps")
    last_event_sequence = int(value["last_event_sequence"])
    if last_event_sequence < 0:
        raise ValueError("snapshot last_event_sequence cannot be negative")
    seen_call_ids = [str(item) for item in value["seen_call_ids"]]
    if any(not call_id for call_id in seen_call_ids):
        raise ValueError("snapshot call IDs must be non-empty")
    if len(seen_call_ids) != len(set(seen_call_ids)):
        raise ValueError("snapshot call IDs must be unique")
    workspace = value["workspace"]
    if not isinstance(workspace, dict):
        raise TypeError("snapshot workspace must be an object")
    return RunSnapshot(
        run_id=str(value["run_id"]),
        snapshot_version=int(value["snapshot_version"]),
        last_event_sequence=last_event_sequence,
        next_step=next_step,
        resume_point=resume_point,
        messages=[message_from_dict(item) for item in value["messages"]],
        seen_call_ids=seen_call_ids,
        verification={
            "mutation_version": int(value["verification"]["mutation_version"]),
            "diff_version": int(value["verification"]["diff_version"]),
            "tests_version": int(value["verification"]["tests_version"]),
        },
        max_steps=max_steps,
        require_verification=bool(value["require_verification"]),
        llm_identity=str(value["llm_identity"]),
        tool_schema_sha256=str(value["tool_schema_sha256"]),
        workspace=WorkspaceIdentity(
            root=str(workspace["root"]),
            git_head=workspace.get("git_head"),
            state_sha256=str(workspace["state_sha256"]),
        ),
        permission_policy_identity=(
            str(value["permission_policy_identity"])
            if value.get("permission_policy_identity") is not None
            else None
        ),
    )


class ArtifactSnapshotStore:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def save(self, snapshot: RunSnapshot) -> ArtifactRef:
        return self.artifacts.put(snapshot_to_bytes(snapshot), "application/json")

    def load(self, ref: ArtifactRef) -> RunSnapshot:
        if ref.media_type != "application/json":
            raise ValueError("snapshot artifact must use application/json")
        return snapshot_from_bytes(self.artifacts.get(ref))


def tool_schema_fingerprint(tools: list[Any]) -> str:
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "permission": tool.permission,
            "effect": tool.effect,
        }
        for tool in tools
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def workspace_identity(root: Path) -> WorkspaceIdentity:
    resolved = root.resolve()
    git_head: str | None = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved,
            shell=False,
            text=True,
            capture_output=True,
            check=False,
            timeout=1,
        )
        if head.returncode == 0:
            git_head = head.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        git_head = None
    return WorkspaceIdentity(
        root=str(resolved),
        git_head=git_head,
        state_sha256=_workspace_tree_sha256(resolved),
    )


def _workspace_tree_sha256(root: Path) -> str:
    ignored = {
        ".git",
        ".mini_agent",
        ".pytest_cache",
        ".venv",
        "venv",
        "__pycache__",
        "artifacts",
    }
    digest = hashlib.sha256()
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in ignored)
        base = Path(directory)
        for name in sorted(files):
            if name == "runs.db" or name.startswith("runs.db-"):
                continue
            path = base / name
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            if path.is_symlink():
                digest.update(os.readlink(path).encode("utf-8"))
            else:
                try:
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                except OSError as error:
                    digest.update(f"unreadable:{type(error).__name__}".encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def verification_from_snapshot(value: dict[str, int]) -> VerificationState:
    return VerificationState(
        mutation_version=value["mutation_version"],
        diff_version=value["diff_version"],
        tests_version=value["tests_version"],
    )


__all__ = [
    "ArtifactSnapshotStore",
    "RunSnapshot",
    "ResumePoint",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotStore",
    "WorkspaceIdentity",
    "message_from_dict",
    "message_to_dict",
    "snapshot_from_bytes",
    "snapshot_to_bytes",
    "tool_schema_fingerprint",
    "verification_from_snapshot",
    "workspace_identity",
]
