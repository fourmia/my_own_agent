import json
import sys
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
    assert result.error_code == "path_escape"


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


def test_missing_file_has_recoverable_error_code(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("read_file", {"path": "missing.md"}))

    assert not result.ok
    assert result.error_code == "file_not_found"
    assert result.details == {"path": "missing.md"}


def test_invalid_utf8_is_classified(tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"\xff\xfe")
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("read_file", {"path": "binary.dat"}))

    assert not result.ok
    assert result.error_code == "invalid_encoding"


def test_nonzero_command_is_a_tool_failure(tmp_path: Path) -> None:
    registry = create_default_registry(tmp_path)
    command = f'"{sys.executable}" -c "import sys; sys.exit(3)"'

    result = registry.run(ToolCall("run_command", {"command": command}))

    assert not result.ok
    assert result.error_code == "command_failed"
    assert result.details == {"exit_code": 3}
    assert "exit_code=3" in result.content


def test_list_files_filters_runtime_noise(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("visible", encoding="utf-8")
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "README.md").write_text("noise", encoding="utf-8")
    registry = create_default_registry(tmp_path)

    result = registry.run(ToolCall("list_files", {}))

    assert result.ok
    assert result.content == "README.md"

