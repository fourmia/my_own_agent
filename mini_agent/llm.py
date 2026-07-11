from __future__ import annotations

import shlex
from abc import ABC, abstractmethod

from mini_agent.schema import LLMResponse, Message, ToolCall
from mini_agent.tools import Tool


class LLM(ABC):
    """模型适配器接口。

    Agent 只依赖这个抽象接口，不依赖具体厂商 API。
    之后接 OpenAI、Anthropic、Qwen、本地模型时，都可以实现这个接口。
    """

    @abstractmethod
    def next(self, messages: list[Message], tools: list[Tool]) -> LLMResponse:
        """基于历史消息和可用工具，返回下一步响应或工具调用。"""


class MockLLM(LLM):
    """接入真实 LLM 前使用的确定性“假模型”。

    它不是为了聪明，而是为了教学：
    - 输入固定，输出固定，方便测试
    - 不需要 API key
    - 让我们先专注 agent loop，而不是 prompt 调参
    """

    def next(self, messages: list[Message], tools: list[Tool]) -> LLMResponse:
        # 简化假设：MockLLM 只看最后一条消息。
        # 真实模型会看完整 messages，并根据上下文做更复杂的推理。
        last = messages[-1]

        # 如果最后一条是 tool，说明工具刚执行完。
        # MockLLM 在这里直接把观察结果总结为最终回答。
        if last.role == "tool":
            return LLMResponse(
                content=f"I observed the tool result:\n{last.content}",
                done=True,
            )

        prompt = last.content.strip()
        lowered = prompt.lower()

        # 下面这些 if 分支模拟“模型选择工具”的过程。
        # Stage 2 接真实 LLM 后，这些规则会被 function calling / structured output 替代。
        if lowered.startswith("read "):
            return LLMResponse(
                content="I should inspect the requested file.",
                tool_call=ToolCall("read_file", {"path": prompt[5:].strip()}),
            )

        if lowered.startswith("search "):
            return LLMResponse(
                content="I should search the workspace.",
                tool_call=ToolCall("search_files", {"query": prompt[7:].strip()}),
            )

        if lowered.startswith("stats "):
            # 这一规则用于演示：新增能力需要模型选择与工具 Schema 两端同时接通。
            return LLMResponse(
                content="I should inspect the requested file metadata.",
                tool_call=ToolCall("get_file_stats", {"path": prompt[6:].strip()}),
            )

        if lowered.startswith("run "):
            command = prompt[4:].strip()
            # shlex.split 在这里不是为了执行命令，而是提前发现明显不完整的 shell 输入。
            # 例如未闭合引号会抛 ValueError，并被上层工具/调用流程暴露出来。
            shlex.split(command)
            return LLMResponse(
                content="I should run the requested command.",
                tool_call=ToolCall("run_command", {"command": command}),
            )

        if "list" in lowered and "file" in lowered:
            return LLMResponse(
                content="I should list files first.",
                tool_call=ToolCall("list_files", {}),
            )

        # 不认识的任务直接结束。真实 agent 也需要“不会就停”的能力，
        # 否则容易胡乱调用工具或产生伪造结果。
        return LLMResponse(
            content=(
                "MockLLM does not know how to solve this yet. Try: "
                "'list files', 'read README.md', 'search agent', or 'run python --version'."
            ),
            done=True,
        )
