from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_agent.schema import ToolCall, ToolResult


# 每个工具都是一个普通 Python 函数：
# 输入：模型给出的结构化参数
# 输出：可放回上下文的字符串 observation
ToolFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    """对外暴露给模型的工具说明。

    name 是模型调用工具时使用的稳定标识。
    description 和 parameters 将来会进入真实 LLM 的工具 schema。
    fn 是真正执行动作的 Python 函数。
    """

    name: str
    description: str
    parameters: dict[str, str]
    fn: ToolFn


class ToolRegistry:
    """工具注册表，也是工具执行的安全边界。

    Agent 不应该随便 eval 模型输出的文本。
    它只能从注册表里找已知工具，并把参数交给对应函数执行。
    """

    def __init__(self, root: Path) -> None:
        # root 是当前工作区根目录。所有文件读写都必须限制在这个目录内。
        self.root = root.resolve()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        # 后注册的同名工具会覆盖旧工具。教学项目里保持简单；
        # 生产系统通常会拒绝重复注册，避免工具被意外替换。
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        # 返回工具元数据。真实 LLM 接入时，会把这些信息转成模型能理解的 schema。
        return list(self._tools.values())

    def run(self, call: ToolCall) -> ToolResult:
        # 只允许调用已注册工具。模型编造出的工具名会被当成一次失败 observation。
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(call.name, False, f"Unknown tool: {call.name}")

        try:
            # 工具内部抛出的异常不直接杀死 agent，而是包装成 ToolResult。
            # 这让模型可以看到错误，例如“文件不存在”，然后决定下一步怎么修正。
            output = tool.fn(call.arguments)
            return ToolResult(call.name, True, output)
        except Exception as exc:  # noqa: BLE001 - teaching project: tool errors become observations
            return ToolResult(call.name, False, f"{type(exc).__name__}: {exc}")

    def resolve_workspace_path(self, raw_path: str) -> Path:
        # 这是最小版路径沙箱：把模型提供的相对路径解析成绝对路径，
        # 然后确认它没有逃出 workspace root。
        # 例如 "../secret.txt" 会被拒绝。
        path = (self.root / raw_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"path escapes workspace: {raw_path}")
        return path


def create_default_registry(root: Path) -> ToolRegistry:
    """创建教学项目默认工具集。

    这里把工具函数定义在工厂函数内部，是为了让它们自然共享 registry/root。
    项目变大后，可以把每个工具拆到单独模块。
    """

    registry = ToolRegistry(root)

    def list_files(_: dict[str, Any]) -> str:
        # rglob 会递归列出目录。这里跳过 .git 和 __pycache__，减少无关噪声。
        files = []
        for path in sorted(registry.root.rglob("*")):
            if ".git" in path.parts or "__pycache__" in path.parts:
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
            if ".git" in path.parts or "__pycache__" in path.parts or not path.is_file():
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
        return "\n".join(parts)

    registry.register(
        Tool(
            name="list_files",
            description="List files in the current workspace.",
            parameters={},
            fn=list_files,
        )
    )
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file from the workspace.",
            parameters={"path": "relative file path"},
            fn=read_file,
        )
    )
    registry.register(
        Tool(
            name="write_file",
            description="Write a UTF-8 text file inside the workspace.",
            parameters={"path": "relative file path", "content": "file content"},
            fn=write_file,
        )
    )
    registry.register(
        Tool(
            name="search_files",
            description="Search text files in the workspace.",
            parameters={"query": "case-insensitive search string"},
            fn=search_files,
        )
    )
    registry.register(
        Tool(
            name="run_command",
            description="Run a shell command in the workspace with a short timeout.",
            parameters={"command": "shell command"},
            fn=run_command,
        )
    )
    return registry
