import json
from pathlib import Path

from mini_agent.agent import Agent
from mini_agent.llm import MockLLM
from mini_agent.tools import create_default_registry


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

