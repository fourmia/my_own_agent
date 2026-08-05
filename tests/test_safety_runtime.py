from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import pytest

from mini_agent.agent import Agent
from mini_agent.llm import LLM
from mini_agent.permissions import (
    FullAccessPermissionPolicy,
    OnRequestPermissionPolicy,
    ReadOnlyPermissionPolicy,
)
from mini_agent.provider_errors import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from mini_agent.retry import ProviderRetryPolicy
from mini_agent.schema import ModelRequest, ModelTurn, ToolCall
from mini_agent.tools import create_coding_registry, create_default_registry


class ScriptedProvider(LLM):
    """Return scripted turns or Provider errors in a deterministic order."""

    def __init__(self, outcomes: list[ModelTurn | Exception]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("the Agent requested more turns than scripted")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FixedApprovalHandler:
    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.calls: list[str] = []

    def approve(self, tool: object, call: ToolCall) -> bool:
        self.calls.append(call.name)
        return self.approved


def patch_call(path: Path, *, call_id: str = "patch-1") -> ToolCall:
    data = path.read_bytes()
    return ToolCall(
        call_id,
        "apply_patch",
        {
            "path": path.name,
            "old_text": "before",
            "new_text": "after",
            "expected_sha256": hashlib.sha256(data).hexdigest(),
        },
    )


def test_retry_policy_uses_retry_after_then_succeeds() -> None:
    delays: list[float] = []
    provider = ScriptedProvider(
        [
            ProviderRateLimitedError(
                "slow down",
                details={"retry_after": "2.5"},
            ),
            ModelTurn(content="done", finish_reason="completed"),
        ]
    )
    policy = ProviderRetryPolicy(
        max_attempts=2,
        base_delay_seconds=0.1,
        max_delay_seconds=8,
        sleeper=delays.append,
    )

    result = policy.generate(provider, ModelRequest(messages=[], tools=[]))

    assert result.content == "done"
    assert delays == [2.5]
    assert len(provider.requests) == 2


def test_retry_policy_uses_bounded_exponential_backoff() -> None:
    delays: list[float] = []
    provider = ScriptedProvider(
        [
            ProviderTimeoutError("first"),
            ProviderTimeoutError("second"),
            ModelTurn(content="done", finish_reason="completed"),
        ]
    )
    policy = ProviderRetryPolicy(
        max_attempts=3,
        base_delay_seconds=2,
        max_delay_seconds=3,
        sleeper=delays.append,
    )

    policy.generate(provider, ModelRequest(messages=[], tools=[]))

    assert delays == [2, 3]


def test_retry_policy_does_not_retry_non_retryable_errors() -> None:
    provider = ScriptedProvider([ProviderAuthError("bad credentials")])
    policy = ProviderRetryPolicy(max_attempts=3, sleeper=lambda _: None)

    with pytest.raises(ProviderAuthError):
        policy.generate(provider, ModelRequest(messages=[], tools=[]))

    assert len(provider.requests) == 1


def test_provider_retry_does_not_repeat_a_completed_tool_call(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="inspect",
                tool_calls=(ToolCall("list-1", "list_files", {}),),
                finish_reason="tool_calls",
            ),
            ProviderTimeoutError("temporary timeout"),
            ModelTurn(content="done", finish_reason="completed"),
        ]
    )
    registry = create_default_registry(tmp_path)
    list_tool = registry.get("list_files")
    assert list_tool is not None
    executions = 0

    def counted_list_files(args: dict[str, object]) -> str:
        nonlocal executions
        executions += 1
        return list_tool.fn(args)

    registry.register(
        type(list_tool)(
            name=list_tool.name,
            description=list_tool.description,
            parameters=list_tool.parameters,
            fn=counted_list_files,
            permission=list_tool.permission,
            effect=list_tool.effect,
        )
    )
    agent = Agent(
        provider,
        registry,
        retry_policy=ProviderRetryPolicy(
            max_attempts=2,
            sleeper=lambda _: None,
        ),
    )

    result = agent.run("list files", trace=False)

    assert result.answer == "done"
    assert executions == 1
    assert len(provider.requests) == 3


def test_read_only_policy_denies_patch_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="change it",
                tool_calls=(patch_call(path),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="denied", finish_reason="completed"),
        ]
    )

    result = Agent(
        provider,
        create_coding_registry(tmp_path),
        permission_policy=ReadOnlyPermissionPolicy(),
    ).run("change note.txt", trace=False)

    assert path.read_text(encoding="utf-8") == "before"
    observation = next(message for message in result.messages if message.role == "tool")
    assert '"code": "permission_denied"' in observation.content


def test_on_request_policy_requires_and_honors_approval(tmp_path: Path) -> None:
    denied_path = tmp_path / "denied.txt"
    denied_path.write_text("before", encoding="utf-8")
    denied_approval = FixedApprovalHandler(False)
    denied_provider = ScriptedProvider(
        [
            ModelTurn(
                content="change it",
                tool_calls=(patch_call(denied_path),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="not changed", finish_reason="completed"),
        ]
    )
    Agent(
        denied_provider,
        create_coding_registry(tmp_path),
        permission_policy=OnRequestPermissionPolicy(),
        approval_handler=denied_approval,
    ).run("change denied.txt", trace=False)

    approved_path = tmp_path / "approved.txt"
    approved_path.write_text("before", encoding="utf-8")
    approved_approval = FixedApprovalHandler(True)
    approved_provider = ScriptedProvider(
        [
            ModelTurn(
                content="change it",
                tool_calls=(patch_call(approved_path, call_id="patch-2"),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="stop", finish_reason="refusal"),
        ]
    )
    Agent(
        approved_provider,
        create_coding_registry(tmp_path),
        permission_policy=OnRequestPermissionPolicy(),
        approval_handler=approved_approval,
    ).run("change approved.txt", trace=False)

    assert denied_path.read_text(encoding="utf-8") == "before"
    assert denied_approval.calls == ["apply_patch"]
    assert approved_path.read_text(encoding="utf-8") == "after"
    assert approved_approval.calls == ["apply_patch"]


def test_full_access_policy_executes_patch_without_approval(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before", encoding="utf-8")
    approval = FixedApprovalHandler(False)
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="change it",
                tool_calls=(patch_call(path),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="stop", finish_reason="refusal"),
        ]
    )

    Agent(
        provider,
        create_coding_registry(tmp_path),
        permission_policy=FullAccessPermissionPolicy(),
        approval_handler=approval,
    ).run("change note.txt", trace=False)

    assert path.read_text(encoding="utf-8") == "after"
    assert approval.calls == []


def test_completion_is_blocked_until_latest_change_is_verified(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before", encoding="utf-8")
    registry = create_coding_registry(tmp_path)
    for name in ("git_diff", "run_tests"):
        tool = registry.get(name)
        assert tool is not None
        registry.register(
            type(tool)(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                fn=lambda _: "verification passed",
                permission=tool.permission,
                effect=tool.effect,
            )
        )
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="change it",
                tool_calls=(patch_call(path),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="done too early", finish_reason="completed"),
            ModelTurn(
                content="verify",
                tool_calls=(
                    ToolCall("diff-1", "git_diff", {}),
                    ToolCall("tests-1", "run_tests", {"targets": []}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="verified", finish_reason="completed"),
        ]
    )

    result = Agent(
        provider,
        registry,
        max_steps=4,
        permission_policy=FullAccessPermissionPolicy(),
    ).run("change note.txt", trace=False)

    assert result.answer == "verified"
    assert result.steps == 4
    blocking_messages = [
        message.content
        for message in result.messages
        if message.role == "system" and "completion_blocked" in message.content
    ]
    assert len(blocking_messages) == 1
    assert '"git_diff"' in blocking_messages[0]
    assert '"run_tests"' in blocking_messages[0]


@pytest.mark.parametrize("finish_reason", ["refusal", "incomplete"])
def test_non_completed_model_status_is_not_hidden_by_verification_guard(
    tmp_path: Path,
    finish_reason: str,
) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="change it",
                tool_calls=(patch_call(path),),
                finish_reason="tool_calls",
            ),
            ModelTurn(content=finish_reason, finish_reason=finish_reason),
        ]
    )

    result = Agent(
        provider,
        create_coding_registry(tmp_path),
        permission_policy=FullAccessPermissionPolicy(),
    ).run("change note.txt", trace=False)

    assert result.answer == finish_reason
    assert result.steps == 2
