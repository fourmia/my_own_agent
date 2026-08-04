from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


# Role 描述一条消息是谁产生的。
# 真实 LLM API 通常也使用类似的角色体系：
# - user: 用户输入的任务或问题
# - assistant: 模型输出的自然语言、计划或工具调用意图
# - tool: 工具执行后的观察结果
# - system: 系统级约束，例如权限、安全策略、输出格式
Role = Literal["user", "assistant", "tool", "system"]

# finish_reason 是 Provider 响应的领域化结束状态。
# 不再用一个 done 布尔值同时表达完成、拒绝、截断和待执行工具。
FinishReason = Literal["completed", "tool_calls", "incomplete", "refusal"]


@dataclass(frozen=True)
class ToolDefinition:
    """模型可见的工具定义，不包含本地 Python handler。

    Provider Adapter 只接触这个稳定结构，不需要知道 ToolRegistry 如何执行工具。
    这样 OpenAI、Anthropic 或本地 Provider 的 SDK 类型不会泄漏进 Runtime。
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, init=False)
class ToolCall:
    """模型请求执行某个工具的结构化指令。

    call_id 是本轮协议的关键关联键：Provider 返回的工具结果必须使用同一个
    call_id。为了让旧教学代码平滑迁移，仍接受 ``ToolCall(name, arguments)``，
    但新代码和真实 Adapter 必须显式传入 Provider 提供的 call_id。
    """

    call_id: str
    name: str
    arguments: dict[str, Any]
    # Provider 通常把函数参数作为 JSON 字符串返回。保留原文便于审计，
    # argument_error 则让解析失败安全地转成关联的 invalid_arguments 结果。
    raw_arguments: str | None
    argument_error: str | None

    def __init__(
        self,
        call_id: str | None = None,
        name: str | dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        *,
        raw_arguments: str | None = None,
        argument_error: str | None = None,
    ) -> None:
        # 兼容旧的 ToolCall(name, arguments) 和 ToolCall(name=..., arguments=...)。
        # 旧调用没有 Provider call_id，只能生成稳定但不适合多调用的过渡 ID。
        if isinstance(name, dict) and arguments is None:
            legacy_name = call_id
            call_id = f"legacy-{legacy_name}"
            arguments = name
            name = legacy_name
        elif call_id is None and isinstance(name, str) and arguments is not None:
            call_id = f"legacy-{name}"

        if not isinstance(call_id, str) or not call_id:
            raise ValueError("ToolCall.call_id must be a non-empty string")
        if not isinstance(name, str) or not name:
            raise ValueError("ToolCall.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise TypeError("ToolCall.arguments must be a dictionary")

        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "raw_arguments", raw_arguments)
        object.__setattr__(self, "argument_error", argument_error)


@dataclass(frozen=True)
class Message:
    """一次 Agent 对话历史里的最小消息单元。

    assistant 消息可以包含零到多个 ToolCall；tool 消息使用 tool_call_id
    指明它对应的调用。历史保留结构化调用，避免后续模型只能从自然语言猜测
    哪个 observation 对应哪个参数。
    """

    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    # 兼容 Iteration 1/2 的单数写法；新代码请使用 tool_calls。
    tool_call: ToolCall | None = field(default=None, repr=False, compare=False)
    tool_call_id: str | None = None
    tool_name: str | None = None

    def __post_init__(self) -> None:
        calls = tuple(self.tool_calls)
        if self.tool_call is not None:
            if calls and calls[0] != self.tool_call:
                raise ValueError("tool_call conflicts with tool_calls")
            if not calls:
                calls = (self.tool_call,)
        object.__setattr__(self, "tool_calls", calls)
        # 让旧的 message.tool_call 读取方式继续得到单个调用。
        if self.tool_call is None and len(calls) == 1:
            object.__setattr__(self, "tool_call", calls[0])


@dataclass(frozen=True)
class ToolResult:
    """工具执行后的结果。

    工具失败也会变成 observation 回到模型上下文，让模型有机会纠参、换工具
    或停止。call_id 不参与自然语言推理，但会在 Provider Adapter 回传时关联结果。
    """

    call_id: str
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
        # 失败统一成固定信封。call_id 也写入信封，便于日志脱离 Message 单独回放。
        return json.dumps(
            {
                "ok": False,
                "call_id": self.call_id,
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
class Usage:
    """Provider 返回的 Token 使用量。"""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ModelRequest:
    """一次 Provider 无关的模型请求。"""

    messages: list[Message]
    tools: list[ToolDefinition]
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ModelTurn:
    """模型的一次完整响应。

    文本、零到多个工具调用、结束原因和计量信息在这里汇合。Provider Adapter
    必须把 SDK 响应归一化成这个类型，Agent Runtime 不直接处理 SDK 对象。
    """

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    # None 只用于兼容旧 LLMResponse；新 Provider 必须显式返回结束原因。
    finish_reason: FinishReason | None = None
    provider_response_id: str | None = None
    usage: Usage | None = None
    # 过渡字段：旧 Mock/测试可以继续传 tool_call 和 done，新实现不应依赖它们。
    tool_call: ToolCall | None = field(default=None, repr=False, compare=False)
    done: bool | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        calls = tuple(self.tool_calls)
        if self.tool_call is not None:
            if calls and calls[0] != self.tool_call:
                raise ValueError("tool_call conflicts with tool_calls")
            if not calls:
                calls = (self.tool_call,)

        # 旧代码传入 tool_call 但未传 finish_reason 时，推导出新的状态。
        reason = self.finish_reason
        if reason is None:
            # 旧代码未提供 finish_reason 时才允许推导新状态。
            reason = "tool_calls" if calls else "completed"
        if reason not in {"completed", "tool_calls", "incomplete", "refusal"}:
            raise ValueError(f"unsupported finish_reason: {reason}")
        call_ids = [call.call_id for call in calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("ToolCall.call_id values must be unique within a ModelTurn")
        if calls and reason != "tool_calls":
            raise ValueError("a ModelTurn with tool calls must finish with tool_calls")
        if not calls and reason == "tool_calls":
            raise ValueError("finish_reason=tool_calls requires at least one ToolCall")
        if self.done is True and calls:
            raise ValueError("done=True cannot be combined with pending tool calls")

        object.__setattr__(self, "tool_calls", calls)
        object.__setattr__(self, "finish_reason", reason)
        if self.tool_call is None and len(calls) == 1:
            object.__setattr__(self, "tool_call", calls[0])


# 旧名称保留为别名，便于教学阶段的 Provider Double 逐步迁移到 ModelTurn。
LLMResponse = ModelTurn
