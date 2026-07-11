from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal


# Role 描述一条消息是谁产生的。
# 真实 LLM API 通常也使用类似的角色体系：
# - user: 用户输入的任务或问题
# - assistant: 模型输出的自然语言、计划或工具调用意图
# - tool: 工具执行后的观察结果
# - system: 系统级约束，例如权限、安全策略、输出格式
Role = Literal["user", "assistant", "tool", "system"]


@dataclass(frozen=True)
class Message:
    """一次 agent 对话历史里的最小消息单元。

    Agent 不只把“用户最后一句话”发给模型，而是把历史消息一起交给模型。
    这就是模型能根据工具结果继续推理的原因。
    """

    role: Role
    content: str


@dataclass(frozen=True)
class ToolCall:
    """模型想调用某个工具时产生的结构化指令。

    关键点：工具调用不是一段自然语言，例如“帮我读 README”，而是：

        name="read_file"
        arguments={"path": "README.md"}

    真实工程里，结构化工具调用可以减少歧义，也方便做权限检查、日志记录和回放。
    """

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """工具执行后的结果。

    ok 用来区分工具本身是否成功。失败也不是直接让 agent 崩溃，
    而是变成 observation 回到模型上下文里，让模型有机会修正。
    """

    name: str
    ok: bool
    content: str
    # error_code 给模型和评估系统一个稳定的机器标识；content 保留给人阅读。
    error_code: str | None = None
    # details 只承载可序列化的诊断信息，不放异常对象等进程内状态。
    details: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> str:
        """转换成写回模型上下文的稳定 observation 协议。"""

        # 成功结果保持原样，避免给每一次正常 observation 增加额外 JSON 包装。
        if self.ok:
            return self.content
        # 失败统一成固定信封，后续真实 LLM 可以按 code 选择重试、纠参或停止。
        return json.dumps(
            {
                "ok": False,
                "tool": self.name,
                "error": {
                    "code": self.error_code or "tool_error",
                    "message": self.content,
                    "details": self.details,
                },
            },
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class LLMResponse:
    """模型的一次响应。

    content 是模型对当前步骤的解释或最终回答。
    tool_call 表示模型还需要外部工具帮它观察世界。
    done 表示模型认为任务已经完成。
    """

    content: str
    tool_call: ToolCall | None = None
    done: bool = False
