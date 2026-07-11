import json
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
    assert result.error_code == "tool_execution_error"


def test_missing_required_argument_is_rejected(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("read_file", {}))

    assert not result.ok
    assert result.error_code == "invalid_arguments"
    observation = json.loads(result.to_observation())
    assert observation["error"]["code"] == "invalid_arguments"


def test_get_file_stats(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_bytes(b"hello\nagent\n")
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("get_file_stats", {"path": "note.txt"}))

    assert result.ok
    assert json.loads(result.content) == {
        "path": "note.txt",
        "bytes": 12,
        "lines": 2,
        "characters": 12,
    }

