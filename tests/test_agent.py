import json
from pathlib import Path

from mini_agent.agent import Agent
from mini_agent.llm import LLM, MockLLM
from mini_agent.schema import LLMResponse, Message, ToolCall
from mini_agent.tools import Tool, create_default_registry


class RepeatingToolLLM(LLM):
    """始终请求同一工具，用来验证 Agent 的 max_steps 安全阀。"""

    def next(self, messages: list[Message], tools: list[Tool]) -> LLMResponse:
        return LLMResponse(
            content="I will keep listing files.",
            tool_call=ToolCall("list_files", {}),
        )


def test_agent_can_read_file(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("agent loop", encoding="utf-8")
    agent = Agent(MockLLM(), create_default_registry(tmp_path))

    result = agent.run("read note.md", trace=False)

    assert "agent loop" in result.answer
    assert result.steps == 2


def test_agent_can_get_file_stats(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("agent loop\n", encoding="utf-8")
    agent = Agent(MockLLM(), create_default_registry(tmp_path))

    result = agent.run("stats note.md", trace=False)

    payload = json.loads(result.messages[-2].content)
    assert payload["path"] == "note.md"
    assert payload["lines"] == 1
    assert result.steps == 2


def test_agent_recovers_from_misspelled_filename(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("recovered content", encoding="utf-8")
    agent = Agent(MockLLM(), create_default_registry(tmp_path))

    result = agent.run("read REDME.md", trace=False)

    calls = [
        message.tool_call.name
        for message in result.messages
        if message.tool_call is not None
    ]
    assert calls == ["read_file", "list_files", "read_file"]
    assert "recovered content" in result.answer
    assert result.steps == 4


def test_agent_stops_when_no_close_filename_exists(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("unrelated", encoding="utf-8")
    agent = Agent(MockLLM(), create_default_registry(tmp_path))

    result = agent.run("read missing.xyz", trace=False)

    assert "could not find a close filename match" in result.answer
    assert result.steps == 3


def test_agent_stops_at_max_steps(tmp_path: Path) -> None:
    agent = Agent(RepeatingToolLLM(), create_default_registry(tmp_path), max_steps=2)

    result = agent.run("loop forever", trace=False)

    assert result.answer == "Stopped after reaching max_steps=2."
    assert result.steps == 2

