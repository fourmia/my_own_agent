from pathlib import Path

from mini_agent.schema import ToolCall
from mini_agent.tools import create_default_registry


def test_read_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello agent", encoding="utf-8")
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("read_file", {"path": "hello.txt"}))

    assert result.ok
    assert result.content == "hello agent"


def test_path_escape_is_blocked(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("read_file", {"path": "../secret.txt"}))

    assert not result.ok
    assert "escapes workspace" in result.content

