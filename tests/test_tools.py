import hashlib
import json
import sys
from pathlib import Path

import pytest

from mini_agent.schema import ToolCall
from mini_agent.tools import (
    create_coding_registry,
    create_default_registry,
    create_read_only_registry,
)


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
        "sha256": hashlib.sha256(b"hello\nagent\n").hexdigest(),
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


def test_read_only_registry_excludes_mutating_tools(tmp_path: Path) -> None:
    registry = create_read_only_registry(tmp_path)

    assert [tool.name for tool in registry.list_tools()] == [
        "list_files",
        "read_file",
        "search_files",
        "get_file_stats",
    ]
    assert registry.get("write_file") is None
    assert registry.get("run_command") is None


def test_apply_patch_replaces_one_exact_block(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    original = b"header\nbefore\nfooter\n"
    path.write_bytes(original)
    registry = create_default_registry(tmp_path)

    result = registry.run(
        ToolCall(
            "patch-1",
            "apply_patch",
            {
                "path": "note.txt",
                "old_text": "before",
                "new_text": "after",
                "expected_sha256": hashlib.sha256(original).hexdigest(),
            },
        )
    )

    assert result.ok
    assert path.read_text(encoding="utf-8") == "header\nafter\nfooter\n"
    payload = json.loads(result.content)
    assert payload["before_sha256"] == hashlib.sha256(original).hexdigest()
    assert payload["after_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_apply_patch_rejects_a_stale_hash_without_modifying_file(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("before", encoding="utf-8")
    registry = create_default_registry(tmp_path)

    result = registry.run(
        ToolCall(
            "patch-1",
            "apply_patch",
            {
                "path": "note.txt",
                "old_text": "before",
                "new_text": "after",
                "expected_sha256": "0" * 64,
            },
        )
    )

    assert not result.ok
    assert result.error_code == "stale_file"
    assert path.read_text(encoding="utf-8") == "before"


@pytest.mark.parametrize(
    ("contents", "old_text", "error_code"),
    [
        ("current", "missing", "patch_conflict"),
        ("same same", "same", "ambiguous_patch"),
    ],
)
def test_apply_patch_requires_one_unique_match(
    tmp_path: Path,
    contents: str,
    old_text: str,
    error_code: str,
) -> None:
    path = tmp_path / "note.txt"
    path.write_text(contents, encoding="utf-8")
    registry = create_default_registry(tmp_path)

    result = registry.run(
        ToolCall(
            "patch-1",
            "apply_patch",
            {
                "path": "note.txt",
                "old_text": old_text,
                "new_text": "replacement",
                "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
    )

    assert not result.ok
    assert result.error_code == error_code
    assert path.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("target", ["-k", "../outside.py"])
def test_run_tests_rejects_unsafe_targets(tmp_path: Path, target: str) -> None:
    registry = create_default_registry(tmp_path)

    result = registry.run(
        ToolCall("tests-1", "run_tests", {"targets": [target]})
    )

    assert not result.ok
    assert result.error_code == "invalid_test_target"


def test_run_tests_executes_a_workspace_test_with_argv(tmp_path: Path) -> None:
    (tmp_path / "test_smoke.py").write_text(
        "def test_smoke():\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )
    registry = create_default_registry(tmp_path)

    result = registry.run(
        ToolCall("tests-1", "run_tests", {"targets": ["test_smoke.py"]})
    )

    assert result.ok
    assert "1 passed" in result.content


def test_coding_registry_excludes_arbitrary_shell_and_whole_file_write(
    tmp_path: Path,
) -> None:
    registry = create_coding_registry(tmp_path)

    assert [tool.name for tool in registry.list_tools()] == [
        "list_files",
        "read_file",
        "search_files",
        "get_file_stats",
        "apply_patch",
        "git_diff",
        "run_tests",
    ]
    assert registry.get("write_file") is None
    assert registry.get("run_command") is None

