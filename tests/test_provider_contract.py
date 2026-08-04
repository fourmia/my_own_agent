from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

from mini_agent.agent import Agent
from mini_agent.llm import LLM
from mini_agent.schema import Message, ModelRequest, ModelTurn, ToolCall
from mini_agent.tools import create_default_registry


class ScriptedProvider(LLM):
    """只实现领域接口的 Fake Provider，用于测试 Runtime 合同而非模型能力。"""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = deque(turns)
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.turns:
            raise AssertionError("the Agent requested more model turns than scripted")
        return self.turns.popleft()


def test_agent_executes_multiple_calls_in_provider_order(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="I will inspect both files.",
                tool_calls=(
                    ToolCall("call-a", "read_file", {"path": "a.txt"}),
                    ToolCall("call-b", "read_file", {"path": "b.txt"}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="Both files were read.", finish_reason="completed"),
        ]
    )

    result = Agent(provider, create_default_registry(tmp_path)).run(
        "inspect both files",
        trace=False,
    )

    assert result.answer == "Both files were read."
    assert result.steps == 2
    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert [
        (message.tool_call_id, message.tool_name, message.content)
        for message in tool_messages
    ] == [
        ("call-a", "read_file", "A"),
        ("call-b", "read_file", "B"),
    ]
    assert [tool.name for tool in provider.requests[0].tools] == [
        "list_files",
        "read_file",
        "write_file",
        "search_files",
        "run_command",
        "get_file_stats",
    ]
    assert all(not hasattr(tool, "fn") for tool in provider.requests[0].tools)
    assert [message.tool_call_id for message in provider.requests[1].messages[-2:]] == [
        "call-a",
        "call-b",
    ]


def test_one_failed_call_does_not_drop_later_calls(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("still executed", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="Try both operations.",
                tool_calls=(
                    ToolCall("call-unknown", "not_registered", {}),
                    ToolCall("call-ok", "read_file", {"path": "ok.txt"}),
                ),
                finish_reason="tool_calls",
            ),
            ModelTurn(content="I observed both results.", finish_reason="completed"),
        ]
    )

    result = Agent(provider, create_default_registry(tmp_path)).run(
        "run two operations",
        trace=False,
    )

    assert result.answer == "I observed both results."
    observations = [
        message.content for message in result.messages if message.role == "tool"
    ]
    assert '"code": "unknown_tool"' in observations[0]
    assert observations[1] == "still executed"


def test_model_turn_rejects_inconsistent_finish_reason() -> None:
    call = ToolCall("call-1", "read_file", {"path": "README.md"})

    with pytest.raises(ValueError, match="finish with tool_calls"):
        ModelTurn(
            content="inconsistent",
            tool_calls=(call,),
            finish_reason="completed",
        )

    with pytest.raises(ValueError, match="requires at least one"):
        ModelTurn(content="also inconsistent", finish_reason="tool_calls")

    with pytest.raises(ValueError, match="unique"):
        ModelTurn(
            content="duplicate calls",
            tool_calls=(call, call),
            finish_reason="tool_calls",
        )


def test_tool_result_keeps_the_call_id(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)
    result = registry.run(ToolCall("call-42", "read_file", {"path": "missing.txt"}))

    assert result.call_id == "call-42"
    assert '"call_id": "call-42"' in result.to_observation()


def test_agent_rejects_call_id_reuse_across_turns(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ModelTurn(
                content="first",
                tool_calls=(ToolCall("same-id", "list_files", {}),),
                finish_reason="tool_calls",
            ),
            ModelTurn(
                content="second",
                tool_calls=(ToolCall("same-id", "list_files", {}),),
                finish_reason="tool_calls",
            ),
        ]
    )

    with pytest.raises(ValueError, match="duplicate ToolCall.call_id"):
        Agent(provider, create_default_registry(tmp_path)).run(
            "reuse an id",
            trace=False,
        )
