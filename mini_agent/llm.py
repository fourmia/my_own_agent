from __future__ import annotations

import json
import shlex
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from typing import Any

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
        last = messages[-1]

        # Iteration 2 开始真正使用完整历史：最后一条 observation 决定当前状态，
        # 更早的 assistant.tool_call 则提供原始参数和已经尝试过的动作。
        if last.role == "tool":
            return self._after_tool(messages, last)

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

    def _after_tool(self, messages: list[Message], observation: Message) -> LLMResponse:
        """根据工具来源、错误码和历史调用选择下一步确定性动作。"""

        error = self._read_error(observation.content)

        if observation.tool_name == "read_file":
            if error is None:
                return LLMResponse(
                    content=f"I observed the tool result:\n{observation.content}",
                    done=True,
                )

            if error.get("code") == "file_not_found":
                # 只允许一次自动恢复。候选文件在重试前消失时，继续循环只会浪费步数。
                if self._count_tool_calls(messages, "read_file") >= 2:
                    return LLMResponse(
                        content="The closest file candidate also disappeared; I stopped retrying.",
                        done=True,
                    )
                return LLMResponse(
                    content="The file was not found. I should inspect workspace filenames.",
                    tool_call=ToolCall("list_files", {}),
                )

            return LLMResponse(
                content=(
                    "I could not read the file and this error is not safe to auto-recover: "
                    f"{error.get('code', 'tool_error')}."
                ),
                done=True,
            )

        if observation.tool_name == "list_files":
            if error is not None:
                return LLMResponse(
                    content="I could not list workspace files, so recovery stopped.",
                    done=True,
                )

            requested = self._latest_read_path(messages)
            candidate = self._closest_file(requested, observation.content)
            if candidate is None:
                return LLMResponse(
                    content=f"I could not find a close filename match for {requested!r}.",
                    done=True,
                )
            return LLMResponse(
                content=f"I found a likely filename match and should retry with {candidate}.",
                tool_call=ToolCall("read_file", {"path": candidate}),
            )

        # 其他工具仍保持 Stage 1 行为；后续迭代会为命令失败等错误增加专门恢复策略。
        return LLMResponse(
            content=f"I observed the tool result:\n{observation.content}",
            done=True,
        )

    @staticmethod
    def _read_error(content: str) -> dict[str, Any] | None:
        """解析 ToolResult 的失败信封；普通文本代表成功 observation。"""

        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict) or payload.get("ok") is not False:
            return None
        error = payload.get("error")
        return error if isinstance(error, dict) else {"code": "tool_error"}

    @staticmethod
    def _count_tool_calls(messages: list[Message], name: str) -> int:
        return sum(
            1
            for message in messages
            if message.tool_call is not None and message.tool_call.name == name
        )

    @staticmethod
    def _latest_read_path(messages: list[Message]) -> str:
        """从历史中的结构化 ToolCall 取路径，而不是解析 assistant 自然语言。"""

        for message in reversed(messages):
            if message.tool_call is not None and message.tool_call.name == "read_file":
                return str(message.tool_call.arguments.get("path", ""))
        return ""

    @staticmethod
    def _closest_file(requested: str, listing: str) -> str | None:
        """用确定性相似度选择候选；低于阈值时宁可停止也不读取无关文件。"""

        candidates = [
            line.strip()
            for line in listing.splitlines()
            if line.strip() and line.strip() != "(no files)"
        ]
        if not candidates:
            return None

        normalized_requested = requested.replace("\\", "/")
        requested_name = PurePosixPath(normalized_requested).name.lower()

        def score(candidate: str) -> float:
            normalized_candidate = candidate.replace("\\", "/")
            candidate_name = PurePosixPath(normalized_candidate).name.lower()
            # 完整路径适合目录拼写错误，basename 适合 README 这类常见文件名拼写错误。
            return max(
                SequenceMatcher(
                    None, normalized_requested.lower(), normalized_candidate.lower()
                ).ratio(),
                SequenceMatcher(None, requested_name, candidate_name).ratio(),
            )

        best = max(candidates, key=lambda candidate: (score(candidate), candidate))
        return best if score(best) >= 0.6 else None
