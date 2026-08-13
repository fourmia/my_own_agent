from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from mini_agent.schema import ToolCall, ToolDefinition, ToolResult


# 每个工具都是一个普通 Python 函数：
# 输入：模型给出的结构化参数
# 输出：可放回上下文的字符串 observation
ToolFn = Callable[[dict[str, Any]], str]
ToolPermission = Literal["read", "write", "execute"]
ToolEffect = Literal["none", "workspace_mutation", "diff_check", "test_check"]


class WorkspacePathError(ValueError):
    """模型提供的路径越过了当前工作区边界。"""


class ToolExecutionFailure(RuntimeError):
    """工具主动报告的、可以安全写回模型上下文的预期失败。"""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class Tool:
    """对外暴露给模型的工具说明。

    name 是模型调用工具时使用的稳定标识。
    description 和 parameters 将来会进入真实 LLM 的工具 schema。
    fn 是真正执行动作的 Python 函数。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    # permission 决定调用是否允许或需要审批；effect 驱动完成前验证守卫。
    permission: ToolPermission = "read"
    effect: ToolEffect = "none"

    def definition(self) -> ToolDefinition:
        """返回不包含本地 handler 的模型可见工具定义。"""

        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolRegistry:
    """工具注册表，也是工具执行的安全边界。

    Agent 不应该随便 eval 模型输出的文本。
    它只能从注册表里找已知工具，并把参数交给对应函数执行。
    """

    def __init__(self, root: Path) -> None:
        # root 是当前工作区根目录。所有文件读写都必须限制在这个目录内。
        self.root = root.resolve()
        self._tools: dict[str, Tool] = {}
        # Validator 与工具同生命周期缓存，避免每次调用都重新解析 JSON Schema。
        self._validators: dict[str, Draft202012Validator] = {}

    def register(self, tool: Tool) -> None:
        # 后注册的同名工具会覆盖旧工具。教学项目里保持简单；
        # 生产系统通常会拒绝重复注册，避免工具被意外替换。
        # 注册时先验证 Schema 本身，让开发错误在启动阶段暴露，而不是等模型调用。
        Draft202012Validator.check_schema(tool.parameters)
        self._tools[tool.name] = tool
        self._validators[tool.name] = Draft202012Validator(tool.parameters)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        # 返回工具元数据。真实 LLM 接入时，会把这些信息转成模型能理解的 schema。
        return list(self._tools.values())

    def list_definitions(self) -> list[ToolDefinition]:
        """只返回 Provider 需要的工具元数据，不暴露 Python handler。"""

        return [tool.definition() for tool in self._tools.values()]

    def run(self, call: ToolCall) -> ToolResult:
        # Provider 参数 JSON 解析失败时不执行 handler，但必须保留原 call_id，
        # 让模型下一轮可以纠正同一个函数调用的参数。
        if call.argument_error is not None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"Invalid JSON arguments for {call.name}: {call.argument_error}",
                error_code="invalid_arguments",
                details={"raw_arguments": call.raw_arguments or ""},
            )

        # 只允许调用已注册工具。模型编造出的工具名会被当成一次失败 observation。
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"Unknown tool: {call.name}",
                error_code="unknown_tool",
            )

        # 参数来自模型，必须在进入具体工具函数前经过统一的信任边界。
        errors = sorted(
            self._validators[call.name].iter_errors(call.arguments),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            # 先返回路径最靠前的错误，使相同输入始终产生确定性的 observation。
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "$"
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"Invalid arguments at {location}: {error.message}",
                error_code="invalid_arguments",
                details={
                    "location": location,
                    "validator": error.validator,
                },
            )

        try:
            # 工具内部抛出的异常不直接杀死 agent，而是包装成 ToolResult。
            # 这让模型可以看到错误，例如“文件不存在”，然后决定下一步怎么修正。
            output = tool.fn(call.arguments)
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=True,
                content=output,
            )
        except ToolExecutionFailure as exc:
            # 工具可以主动提供稳定错误码，例如 Shell 非零退出，而不是伪装成成功文本。
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=str(exc),
                error_code=exc.code,
                details=exc.details,
            )
        except WorkspacePathError as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=str(exc),
                error_code="path_escape",
                details={"path": str(call.arguments.get("path", ""))},
            )
        except FileNotFoundError:
            # 不把解析后的绝对路径写回模型，避免 observation 泄漏工作区外的主机信息。
            raw_path = str(call.arguments.get("path", ""))
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"File not found: {raw_path}",
                error_code="file_not_found",
                details={"path": raw_path},
            )
        except UnicodeDecodeError as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content="File is not valid UTF-8 text.",
                error_code="invalid_encoding",
                details={
                    "path": str(call.arguments.get("path", "")),
                    "start": exc.start,
                },
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"Command timed out after {exc.timeout} seconds.",
                error_code="command_timeout",
                details={"timeout_seconds": exc.timeout},
            )
        except Exception as exc:  # noqa: BLE001 - teaching project: tool errors become observations
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"{type(exc).__name__}: {exc}",
                error_code="tool_execution_error",
                details={"exception_type": type(exc).__name__},
            )

    def resolve_workspace_path(self, raw_path: str) -> Path:
        # 这是最小版路径沙箱：把模型提供的相对路径解析成绝对路径，
        # 然后确认它没有逃出 workspace root。
        # 例如 "../secret.txt" 会被拒绝。
        path = (self.root / raw_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise WorkspacePathError(f"path escapes workspace: {raw_path}")
        return path


def create_default_registry(root: Path) -> ToolRegistry:
    """创建教学项目默认工具集。

    这里把工具函数定义在工厂函数内部，是为了让它们自然共享 registry/root。
    项目变大后，可以把每个工具拆到单独模块。
    """

    registry = ToolRegistry(root)
    ignored_directories = {
        ".git",
        ".mini_agent",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
    }

    def is_ignored(path: Path) -> bool:
        # 工具输出会进入模型上下文；缓存、虚拟环境和安装元数据既耗 Token，也会干扰候选排序。
        return any(
            part in ignored_directories or part.endswith(".egg-info")
            for part in path.parts
        )

    def list_files(_: dict[str, Any]) -> str:
        # rglob 会递归列出目录，is_ignored 负责过滤不应暴露给模型的运行时噪声。
        files = []
        for path in sorted(registry.root.rglob("*")):
            if is_ignored(path):
                continue
            if path.is_file():
                files.append(str(path.relative_to(registry.root)))
        return "\n".join(files) if files else "(no files)"

    def read_file(args: dict[str, Any]) -> str:
        # 参数来自模型，所以即使当前 MockLLM 很简单，也要走统一的路径检查。
        path = registry.resolve_workspace_path(str(args["path"]))
        return path.read_text(encoding="utf-8")

    def write_file(args: dict[str, Any]) -> str:
        # 整文件覆盖仅保留在教学用默认工具集，并标记为 write 权限。
        # 真实 coding registry 不注册它，避免模型绕过哈希前置条件覆盖并发修改。
        path = registry.resolve_workspace_path(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args["content"]), encoding="utf-8")
        return f"wrote {path.relative_to(registry.root)}"

    def search_files(args: dict[str, Any]) -> str:
        # 这是一个很朴素的全文搜索工具。真实 coding agent 通常会优先用 rg，
        # 并做结果截断、相关性排序和二次读取。
        query = str(args["query"])
        matches: list[str] = []
        for path in sorted(registry.root.rglob("*")):
            if is_ignored(path) or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    rel = path.relative_to(registry.root)
                    matches.append(f"{rel}:{number}: {line.strip()}")
        # 限制返回数量，避免 observation 太长撑爆上下文窗口。
        return "\n".join(matches[:50]) if matches else "(no matches)"

    def run_command(args: dict[str, Any]) -> str:
        # 任意 Shell 仅保留在教学用默认工具集，并标记为 execute 权限。
        # 真实 coding registry 只开放参数受限的 run_tests，不暴露此 handler。
        command = str(args["command"])
        completed = subprocess.run(
            command,
            cwd=registry.root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        result = format_process_result(completed)
        if completed.returncode != 0:
            # subprocess.run(check=False) 让我们能保留 stdout/stderr，再用领域错误表达失败。
            raise ToolExecutionFailure(
                "command_failed",
                result,
                details={"exit_code": completed.returncode},
            )
        return result

    def format_process_result(completed: subprocess.CompletedProcess[str]) -> str:
        # 把 stdout/stderr/exit code 都返回给模型。exit code 对调试测试失败尤其重要。
        output = completed.stdout.strip()
        error = completed.stderr.strip()
        parts = [f"exit_code={completed.returncode}"]
        if output:
            parts.append(f"stdout:\n{output}")
        if error:
            parts.append(f"stderr:\n{error}")
        result = "\n".join(parts)
        if len(result) > 20_000:
            return result[:20_000] + "\n... output truncated ..."
        return result

    def get_file_stats(args: dict[str, Any]) -> str:
        # 复用路径沙箱，统计工具也不能通过 ../ 或绝对路径逃出工作区。
        path = registry.resolve_workspace_path(str(args["path"]))
        data = path.read_bytes()
        text = data.decode("utf-8")
        return json.dumps(
            {
                "path": str(path.relative_to(registry.root)),
                # bytes 来自磁盘，characters/lines 来自 UTF-8 解码后的文本；CRLF 下两者可能不同。
                "bytes": len(data),
                "lines": len(text.splitlines()),
                "characters": len(text),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            ensure_ascii=False,
        )

    def apply_patch(args: dict[str, Any]) -> str:
        """以哈希和唯一文本匹配为前置条件执行一次局部替换。"""

        path = registry.resolve_workspace_path(str(args["path"]))
        data = path.read_bytes()
        before_hash = hashlib.sha256(data).hexdigest()
        expected_hash = str(args["expected_sha256"]).lower()
        if before_hash != expected_hash:
            raise ToolExecutionFailure(
                "stale_file",
                "File changed after it was inspected; refresh content and hash.",
                details={
                    "path": str(path.relative_to(registry.root)),
                    "expected_sha256": expected_hash,
                    "actual_sha256": before_hash,
                },
            )

        text = data.decode("utf-8")
        old_text = str(args["old_text"])
        new_text = str(args["new_text"])
        occurrences = text.count(old_text)
        if occurrences == 0:
            raise ToolExecutionFailure(
                "patch_conflict",
                "The expected old_text was not found in the current file.",
                details={"path": str(path.relative_to(registry.root))},
            )
        if occurrences > 1:
            raise ToolExecutionFailure(
                "ambiguous_patch",
                "The expected old_text occurs more than once; provide more context.",
                details={
                    "path": str(path.relative_to(registry.root)),
                    "occurrences": occurrences,
                },
            )
        if old_text == new_text:
            raise ToolExecutionFailure(
                "patch_no_change",
                "old_text and new_text are identical.",
                details={"path": str(path.relative_to(registry.root))},
            )

        updated = text.replace(old_text, new_text, 1).encode("utf-8")
        path.write_bytes(updated)
        return json.dumps(
            {
                "path": str(path.relative_to(registry.root)),
                "before_sha256": before_hash,
                "after_sha256": hashlib.sha256(updated).hexdigest(),
                "replacements": 1,
            },
            ensure_ascii=False,
        )

    def git_diff(_: dict[str, Any]) -> str:
        completed = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--unified=3", "--"],
            cwd=registry.root,
            shell=False,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            raise ToolExecutionFailure(
                "command_failed",
                format_process_result(completed),
                details={"exit_code": completed.returncode},
            )
        output = completed.stdout.strip()
        if not output:
            return "(no changes)"
        return output[:20_000] + ("\n... diff truncated ..." if len(output) > 20_000 else "")

    def run_tests(args: dict[str, Any]) -> str:
        targets = [str(target) for target in args["targets"]]
        for target in targets:
            # 使用 argv + shell=False 已阻止 Shell 注入；同时禁止 pytest 选项注入，
            # 并限制 node ID 的路径部分必须位于工作区内。
            if target.startswith("-") or "\x00" in target:
                raise ToolExecutionFailure(
                    "invalid_test_target",
                    f"Unsafe pytest target: {target!r}",
                    details={"target": target},
                )
            target_path = target.split("::", 1)[0]
            try:
                registry.resolve_workspace_path(target_path)
            except WorkspacePathError as exc:
                raise ToolExecutionFailure(
                    "invalid_test_target",
                    f"Unsafe pytest target: {target!r}",
                    details={"target": target, "reason": str(exc)},
                ) from exc
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *targets],
            cwd=registry.root,
            shell=False,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        result = format_process_result(completed)
        if completed.returncode != 0:
            raise ToolExecutionFailure(
                "command_failed",
                result,
                details={"exit_code": completed.returncode, "targets": targets},
            )
        return result

    registry.register(
        Tool(
            name="list_files",
            description="List files in the current workspace.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=list_files,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Workspace-relative file path.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            fn=read_file,
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description="Write a UTF-8 text file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Workspace-relative file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file content.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            fn=write_file,
            permission="write",
            effect="workspace_mutation",
        )
    )
    registry.register(
        Tool(
            name="search_files",
            description="Search text files in the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Case-insensitive search string.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            fn=search_files,
        )
    )
    registry.register(
        Tool(
            name="run_command",
            description="Run a shell command in the workspace with a short timeout.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Shell command to run from the workspace root.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            fn=run_command,
            permission="execute",
        )
    )
    registry.register(
        Tool(
            name="get_file_stats",
            description="Return byte, line, and character counts for a UTF-8 workspace file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Workspace-relative UTF-8 file path.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            fn=get_file_stats,
        )
    )
    registry.register(
        Tool(
            name="apply_patch",
            description=(
                "Replace one exact, unique text block in an existing UTF-8 file. "
                "Call get_file_stats first and pass its sha256 as expected_sha256."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Workspace-relative existing file path.",
                    },
                    "old_text": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Exact unique text currently present in the file.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-fA-F]{64}$",
                        "description": "SHA-256 returned by get_file_stats.",
                    },
                },
                "required": ["path", "old_text", "new_text", "expected_sha256"],
                "additionalProperties": False,
            },
            fn=apply_patch,
            permission="write",
            effect="workspace_mutation",
        )
    )
    registry.register(
        Tool(
            name="git_diff",
            description="Inspect the current workspace git diff without modifying files.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            fn=git_diff,
            effect="diff_check",
        )
    )
    registry.register(
        Tool(
            name="run_tests",
            description=(
                "Run pytest with safe argv handling. Use an empty targets array for "
                "the full suite or provide test paths/node IDs."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 20,
                        "description": "Pytest paths or node IDs; options are forbidden.",
                    }
                },
                "required": ["targets"],
                "additionalProperties": False,
            },
            fn=run_tests,
            permission="execute",
            effect="test_check",
        )
    )
    return registry


def create_read_only_registry(root: Path) -> ToolRegistry:
    """创建物理上不包含任何副作用 handler 的最小只读工具集。

    它与 ReadOnlyPermissionPolicy 的用途不同：Policy 可以拒绝已注册工具，而本函数
    让高风险工具根本不会进入模型 schema，适合能力必须最小化的宿主环境。
    """

    all_tools = create_default_registry(root)
    registry = ToolRegistry(root)
    for name in ("list_files", "read_file", "search_files", "get_file_stats"):
        tool = all_tools.get(name)
        if tool is None:  # 防御性检查：默认工具重命名时应在启动阶段失败。
            raise RuntimeError(f"missing built-in read-only tool: {name}")
        registry.register(tool)
    return registry


def create_coding_registry(root: Path) -> ToolRegistry:
    """真实 coding Provider 使用的受控工具集，不暴露整文件写入或任意 Shell。"""

    all_tools = create_default_registry(root)
    registry = ToolRegistry(root)
    names = (
        "list_files",
        "read_file",
        "search_files",
        "get_file_stats",
        "apply_patch",
        "git_diff",
        "run_tests",
    )
    for name in names:
        tool = all_tools.get(name)
        if tool is None:
            raise RuntimeError(f"missing built-in coding tool: {name}")
        registry.register(tool)
    return registry
