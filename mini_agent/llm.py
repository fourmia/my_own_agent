from __future__ import annotations

import json
import shlex
from abc import ABC
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from typing import Any

from mini_agent.schema import (
    LLMResponse,
    Message,
    ModelRequest,
    ModelTurn,
    ToolCall,
)
from mini_agent.tools import Tool


class LLM(ABC):
    """Provider 无关的模型适配器接口。

    新 Provider 实现 ``generate``。为了让 Iteration 1/2 的测试 Double 不必一次
    性重写，基类还能发现并调用旧的 ``next(messages, tools)`` 实现；这个兼容入口
    会在后续版本移除。
    """

    def generate(self, request: ModelRequest) -> ModelTurn:
        """根据领域请求返回一个完整的模型轮次。"""

        legacy_next = getattr(type(self), "next", None)
        if legacy_next is not None and legacy_next is not LLM.next:
            # 旧 Provider Double 通常只关心 messages，tools 的具体类型不影响其测试。
            turn = legacy_next(self, request.messages, request.tools)
            return self._normalize_legacy_call_ids(turn, request.messages)
        raise NotImplementedError(
            f"{type(self).__name__} must implement generate() or legacy next()"
        )

    def next(self, messages: list[Message], tools: list[Tool]) -> ModelTurn:
        """Iteration 2 兼容入口；新 Runtime 应调用 generate。"""

        return self.generate(ModelRequest(messages=messages, tools=tools))

    @staticmethod
    def _normalize_legacy_call_ids(
        turn: ModelTurn,
        messages: list[Message],
    ) -> ModelTurn:
        """为旧 ToolCall(name, arguments) 分配可回放且跨轮次唯一的 ID。"""

        next_number = 1 + sum(len(message.tool_calls) for message in messages)
        normalized: list[ToolCall] = []
        changed = False
        for offset, call in enumerate(turn.tool_calls):
            if call.call_id.startswith("legacy-"):
                call = ToolCall(
                    call_id=f"compat-call-{next_number + offset:04d}",
                    name=call.name,
                    arguments=call.arguments,
                    raw_arguments=call.raw_arguments,
                    argument_error=call.argument_error,
                )
                changed = True
            normalized.append(call)

        if not changed:
            return turn
        return ModelTurn(
            content=turn.content,
            tool_calls=tuple(normalized),
            finish_reason=turn.finish_reason,
            provider_response_id=turn.provider_response_id,
            usage=turn.usage,
        )


class MockLLM(LLM):
    """接入真实 LLM 前使用的确定性“假模型”。

    它不是为了聪明，而是为了教学：
    - 输入固定，输出固定，方便测试
    - 不需要 API key
    - 让我们先专注 Agent Loop 和 Provider 合同，而不是 Prompt 调参
    """

    def generate(self, request: ModelRequest) -> ModelTurn:
        messages = request.messages
        last = messages[-1]

        # Iteration 2 开始真正使用完整历史：最后一条 observation 决定当前状态，
        # 更早的 assistant.tool_calls 则提供原始参数和已经尝试过的动作。
        if last.role == "tool":
            return self._after_tool(messages, last)

        prompt = last.content.strip()
        lowered = prompt.lower()

        # 下面这些 if 分支模拟“模型选择工具”的过程。
        # 接真实 LLM 后，这些规则会被 function calling / structured output 替代。
        if lowered.startswith("read "):
            return self._tool_turn(
                messages,
                "I should inspect the requested file.",
                "read_file",
                {"path": prompt[5:].strip()},
            )

        if lowered.startswith("search "):
            return self._tool_turn(
                messages,
                "I should search the workspace.",
                "search_files",
                {"query": prompt[7:].strip()},
            )

        if lowered.startswith("stats "):
            # 这一规则用于演示：新增能力需要模型选择与工具 Schema 两端同时接通。
            return self._tool_turn(
                messages,
                "I should inspect the requested file metadata.",
                "get_file_stats",
                {"path": prompt[6:].strip()},
            )

        if lowered.startswith("run "):
            command = prompt[4:].strip()
            # shlex.split 在这里不是为了执行命令，而是提前发现明显不完整的 shell 输入。
            # 例如未闭合引号会抛 ValueError，并被上层工具/调用流程暴露出来。
            shlex.split(command)
            return self._tool_turn(
                messages,
                "I should run the requested command.",
                "run_command",
                {"command": command},
            )

        if "list" in lowered and "file" in lowered:
            return self._tool_turn(
                messages,
                "I should list files first.",
                "list_files",
                {},
            )

        # 不认识的任务直接结束。真实 Agent 也需要“不会就停”的能力，
        # 否则容易胡乱调用工具或产生伪造结果。
        return ModelTurn(
            content=(
                "MockLLM does not know how to solve this yet. Try: "
                "'list files', 'read README.md', 'search agent', or 'run python --version'."
            ),
            finish_reason="completed",
        )

    def _after_tool(self, messages: list[Message], observation: Message) -> ModelTurn:
        """根据工具来源、错误码和历史调用选择下一步确定性动作。"""

        error = self._read_error(observation.content)

        if observation.tool_name == "read_file":
            if error is None:
                return ModelTurn(
                    content=f"I observed the tool result:\n{observation.content}",
                    finish_reason="completed",
                )

            if error.get("code") == "file_not_found":
                # 只允许一次自动恢复。候选文件在重试前消失时，继续循环只会浪费步数。
                if self._count_tool_calls(messages, "read_file") >= 2:
                    return ModelTurn(
                        content="The closest file candidate also disappeared; I stopped retrying.",
                        finish_reason="completed",
                    )
                return self._tool_turn(
                    messages,
                    "The file was not found. I should inspect workspace filenames.",
                    "list_files",
                    {},
                )

            return ModelTurn(
                content=(
                    "I could not read the file and this error is not safe to auto-recover: "
                    f"{error.get('code', 'tool_error')}."
                ),
                finish_reason="completed",
            )

        if observation.tool_name == "list_files":
            if error is not None:
                return ModelTurn(
                    content="I could not list workspace files, so recovery stopped.",
                    finish_reason="completed",
                )

            requested = self._latest_read_path(messages)
            if not requested:
                # 用户直接请求 list files 时，这只是最终观察，不是 read_file 恢复步骤。
                return ModelTurn(
                    content=f"I observed the tool result:\n{observation.content}",
                    finish_reason="completed",
                )
            candidate = self._closest_file(requested, observation.content)
            if candidate is None:
                return ModelTurn(
                    content=f"I could not find a close filename match for {requested!r}.",
                    finish_reason="completed",
                )
            return self._tool_turn(
                messages,
                f"I found a likely filename match and should retry with {candidate}.",
                "read_file",
                {"path": candidate},
            )

        # 其他工具仍保持 Stage 1 行为；后续迭代会为命令失败等错误增加专门恢复策略。
        return ModelTurn(
            content=f"I observed the tool result:\n{observation.content}",
            finish_reason="completed",
        )

    @staticmethod
    def _tool_turn(
        messages: list[Message],
        content: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ModelTurn:
        """构造带确定性 call_id 的单工具轮次，保证测试可回放。"""

        call_number = 1 + sum(len(message.tool_calls) for message in messages)
        call = ToolCall(
            call_id=f"mock-call-{call_number:04d}",
            name=name,
            arguments=arguments,
        )
        return ModelTurn(
            content=content,
            tool_calls=(call,),
            finish_reason="tool_calls",
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
            for call in message.tool_calls
            if call.name == name
        )

    @staticmethod
    def _latest_read_path(messages: list[Message]) -> str:
        """从历史中的结构化 ToolCall 取路径，而不是解析 assistant 自然语言。"""

        for message in reversed(messages):
            for call in reversed(message.tool_calls):
                if call.name == "read_file":
                    return str(call.arguments.get("path", ""))
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


__all__ = ["LLM", "MockLLM", "LLMResponse"]
