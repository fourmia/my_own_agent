from __future__ import annotations

import json
from dataclasses import dataclass

from mini_agent.schema import ToolResult
from mini_agent.tools import Tool


@dataclass
class VerificationState:
    """跟踪最近一次工作区修改是否完成 Diff 与测试验证。"""

    mutation_version: int = 0
    diff_version: int = 0
    tests_version: int = 0

    def snapshot(self) -> dict[str, int]:
        """返回可持久化快照，避免 Event Journal 引用后续会变化的对象。"""

        return {
            "mutation_version": self.mutation_version,
            "diff_version": self.diff_version,
            "tests_version": self.tests_version,
        }

    def observe(self, tool: Tool | None, result: ToolResult) -> None:
        if tool is None or not result.ok:
            return
        if tool.effect == "workspace_mutation":
            self.mutation_version += 1
        elif tool.effect == "diff_check":
            self.diff_version = self.mutation_version
        elif tool.effect == "test_check":
            self.tests_version = self.mutation_version

    def missing_requirements(self) -> list[str]:
        if self.mutation_version == 0:
            return []
        missing: list[str] = []
        if self.diff_version != self.mutation_version:
            missing.append("git_diff")
        if self.tests_version != self.mutation_version:
            missing.append("run_tests")
        return missing

    def blocking_message(self) -> str:
        """生成模型可读、机器也能稳定解析的完成阻断消息。"""

        return json.dumps(
            {
                "completion_blocked": True,
                "reason": "workspace_changes_require_verification",
                "required_tools": self.missing_requirements(),
                "instruction": (
                    "Inspect the current diff and run tests after the latest change "
                    "before claiming completion."
                ),
            },
            ensure_ascii=False,
        )


__all__ = ["VerificationState"]
