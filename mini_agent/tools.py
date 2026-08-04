from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from mini_agent.schema import ToolCall, ToolDefinition, ToolResult


# 每个工具都是一个普通 Python 函数：
# 输入：模型给出的结构化参数
# 输出：可放回上下文的字符串 observation
ToolFn = Callable[[dict[str, Any]], str]


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
    ignored_directories = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}

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
        # 写文件是高风险工具。现在为了学习先开放，后面会引入 permission mode。
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
        # shell 工具能力很强，也最危险。这里加了 timeout，并且固定在 workspace root 执行。
        # 后面学习权限系统时，我们会把 run_command 放到需要审批的工具类别里。
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
        # 把 stdout/stderr/exit code 都返回给模型。exit code 对调试测试失败尤其重要。
        output = completed.stdout.strip()
        error = completed.stderr.strip()
        parts = [f"exit_code={completed.returncode}"]
        if output:
            parts.append(f"stdout:\n{output}")
        if error:
            parts.append(f"stderr:\n{error}")
        result = "\n".join(parts)
        if completed.returncode != 0:
            # subprocess.run(check=False) 让我们能保留 stdout/stderr，再用领域错误表达失败。
            raise ToolExecutionFailure(
                "command_failed",
                result,
                details={"exit_code": completed.returncode},
            )
        return result

    def get_file_stats(args: dict[str, Any]) -> str:
        # 复用路径沙箱，统计工具也不能通过 ../ 或绝对路径逃出工作区。
        path = registry.resolve_workspace_path(str(args["path"]))
        text = path.read_text(encoding="utf-8")
        return json.dumps(
            {
                "path": str(path.relative_to(registry.root)),
                # bytes 来自磁盘，characters/lines 来自 UTF-8 解码后的文本；CRLF 下两者可能不同。
                "bytes": path.stat().st_size,
                "lines": len(text.splitlines()),
                "characters": len(text),
            },
            ensure_ascii=False,
        )

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
    return registry


def create_read_only_registry(root: Path) -> ToolRegistry:
    """为尚未接入权限审批的真实 Provider 创建只读工具集。

    handler 和 Schema 仍来自同一个默认注册表，避免安全配置复制出两套工具定义。
    Iteration 5 引入 Permission Policy 后，再由策略层决定何时开放写文件和 Shell。
    """

    all_tools = create_default_registry(root)
    registry = ToolRegistry(root)
    for name in ("list_files", "read_file", "search_files", "get_file_stats"):
        tool = all_tools.get(name)
        if tool is None:  # 防御性检查：默认工具重命名时应在启动阶段失败。
            raise RuntimeError(f"missing built-in read-only tool: {name}")
        registry.register(tool)
    return registry
