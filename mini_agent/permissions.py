from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from mini_agent.schema import ToolCall
from mini_agent.tools import Tool


PermissionOutcome = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PermissionDecision:
    outcome: PermissionOutcome
    reason: str


class PermissionPolicy(Protocol):
    def evaluate(self, tool: Tool, call: ToolCall) -> PermissionDecision:
        """在工具 handler 执行前返回权限决策。"""


class ApprovalHandler(Protocol):
    def approve(self, tool: Tool, call: ToolCall) -> bool:
        """处理 ask 决策；False 表示用户或宿主拒绝。"""


class ReadOnlyPermissionPolicy:
    def evaluate(self, tool: Tool, call: ToolCall) -> PermissionDecision:
        if tool.permission == "read":
            return PermissionDecision("allow", "read-only tool")
        return PermissionDecision(
            "deny",
            f"{tool.permission} tools are disabled by read-only policy",
        )


class OnRequestPermissionPolicy:
    def evaluate(self, tool: Tool, call: ToolCall) -> PermissionDecision:
        if tool.permission == "read":
            return PermissionDecision("allow", "read-only tool")
        return PermissionDecision(
            "ask",
            f"{tool.permission} tool requires explicit approval",
        )


class FullAccessPermissionPolicy:
    def evaluate(self, tool: Tool, call: ToolCall) -> PermissionDecision:
        return PermissionDecision("allow", "full access policy")


class DenyApprovalHandler:
    def approve(self, tool: Tool, call: ToolCall) -> bool:
        return False


class ConsoleApprovalHandler:
    """CLI 审批器，只显示目标摘要，不回显完整 Patch 或敏感内容。"""

    def approve(self, tool: Tool, call: ToolCall) -> bool:
        target = call.arguments.get("path")
        if target is None:
            target = call.arguments.get("targets", [])
        try:
            answer = input(
                f"Approve {tool.name} ({tool.permission}) target={target!r}? [y/N] "
            )
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}


__all__ = [
    "ApprovalHandler",
    "ConsoleApprovalHandler",
    "DenyApprovalHandler",
    "FullAccessPermissionPolicy",
    "OnRequestPermissionPolicy",
    "PermissionDecision",
    "PermissionPolicy",
    "ReadOnlyPermissionPolicy",
]
