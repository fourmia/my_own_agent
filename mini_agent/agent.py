from __future__ import annotations

from dataclasses import dataclass

from mini_agent.llm import LLM
from mini_agent.permissions import (
    ApprovalHandler,
    DenyApprovalHandler,
    PermissionPolicy,
    ReadOnlyPermissionPolicy,
)
from mini_agent.provider_errors import ProviderError
from mini_agent.retry import ProviderRetryPolicy
from mini_agent.schema import Message, ModelRequest, ToolCall, ToolResult
from mini_agent.tools import Tool, ToolRegistry
from mini_agent.verification import VerificationState


@dataclass(frozen=True)
class AgentResult:
    """Agent 运行结束后的汇总结果。

    messages 被完整保留下来，是为了后续做调试、评估和会话持久化。
    很多 Agent 问题不是出在最终答案，而是出在中间某一步的错误观察或错误决策。
    """

    answer: str
    steps: int
    messages: list[Message]


class Agent:
    """把 LLM、工具和消息历史串起来的最小 Agent Runtime。

    一次模型轮次可以返回零到多个工具调用。Runtime 保留 Provider 返回的顺序，
    逐个执行调用，并且等这一批结果全部进入历史后才请求下一轮模型。这样可以
    在尚未引入并发、审批和写冲突策略前，先固定可回放的协议语义。
    """

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        max_steps: int = 6,
        *,
        permission_policy: PermissionPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        retry_policy: ProviderRetryPolicy | None = None,
        require_verification: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        # 依赖注入：Agent 不关心具体模型是哪家，也不关心工具怎么实现。
        # 这让我们可以先用 MockLLM 学习，之后再替换成真实 Provider Adapter。
        self.llm = llm
        self.tools = tools
        self.permission_policy = permission_policy or ReadOnlyPermissionPolicy()
        self.approval_handler = approval_handler or DenyApprovalHandler()
        self.retry_policy = retry_policy or ProviderRetryPolicy()
        self.require_verification = require_verification
        self.system_prompt = system_prompt

        # max_steps 是安全阀。真实模型可能误判、循环调用工具，甚至被 prompt injection 带偏。
        # 没有步数上限，Agent 可能无限运行、疯狂烧钱或不断修改文件。
        self.max_steps = max_steps

    def run(self, task: str, trace: bool = True) -> AgentResult:
        # 对话历史从用户任务开始。后面每一轮 assistant/tool 消息都会追加进来。
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message("system", self.system_prompt))
        messages.append(Message("user", task))
        seen_call_ids: set[str] = set()
        verification = VerificationState()

        for step in range(1, self.max_steps + 1):
            # 只把 ToolDefinition 交给模型，避免 Provider 看到本地 Python handler。
            request = ModelRequest(
                # 每一轮发送历史快照。若直接传入可变的 messages，后续追加的
                # assistant/tool 消息会篡改已经“发送”的请求，破坏审计和回放。
                messages=list(messages),
                tools=self.tools.list_definitions(),
            )
            response = self.retry_policy.generate(
                self.llm,
                request,
                on_retry=(
                    lambda attempt, delay, error: self._trace_retry(
                        attempt,
                        delay,
                        error,
                        trace=trace,
                    )
                ),
            )

            # assistant 消息必须持久化整批结构化调用。后续模型需要知道每个
            # observation 对应哪个 call_id 和原始参数，而不能依赖打印日志猜测。
            messages.append(
                Message(
                    "assistant",
                    response.content,
                    tool_calls=response.tool_calls,
                )
            )

            if trace:
                print(f"\n[step {step}] assistant")
                print(response.content)

            # 没有工具调用代表模型给出了最终文本、拒绝或不完整结果；这些状态都
            # 需要交给上层展示，不能因为没有工具就继续盲目循环。
            if not response.tool_calls:
                missing = verification.missing_requirements()
                if (
                    self.require_verification
                    and response.finish_reason == "completed"
                    and missing
                ):
                    blocked = verification.blocking_message()
                    messages.append(Message("system", blocked))
                    if trace:
                        print(f"\n[step {step}] completion_blocked")
                        print(blocked)
                    continue
                return AgentResult(response.content, step, messages)

            # 逐个执行同一 ModelTurn 中的调用。一个工具失败不阻止同批次后续调用；
            # 是否 fail-fast 属于未来 Permission/Policy 层，而不是 handler 的职责。
            for call in response.tool_calls:
                if call.call_id in seen_call_ids:
                    raise ValueError(
                        f"duplicate ToolCall.call_id in Agent session: {call.call_id}"
                    )
                seen_call_ids.add(call.call_id)
                if trace:
                    print(
                        f"\n[step {step}] tool_call[{call.call_id}]: "
                        f"{call.name}({call.arguments})"
                    )

                # 权限检查发生在 handler 之前。deny/未批准 ask 会生成普通 ToolResult，
                # 模型可以看到拒绝原因，但任何文件或进程副作用都不会发生。
                tool = self.tools.get(call.name)
                result = self._execute_call(call, tool)
                verification.observe(tool, result)
                observation = result.to_observation()
                messages.append(
                    Message(
                        "tool",
                        observation,
                        tool_call_id=result.call_id,
                        tool_name=result.name,
                    )
                )

                if trace:
                    print(f"\n[step {step}] observation[{result.call_id}]")
                    print(observation)

        # 到达上限时明确停止，并把停止原因也写进消息历史，方便调试。
        final = f"Stopped after reaching max_steps={self.max_steps}."
        messages.append(Message("assistant", final))
        return AgentResult(final, self.max_steps, messages)

    def _execute_call(self, call: ToolCall, tool: Tool | None) -> ToolResult:
        if tool is None:
            return self.tools.run(call)

        decision = self.permission_policy.evaluate(tool, call)
        if decision.outcome == "deny":
            return self._permission_denied(call, decision.reason)
        if decision.outcome == "ask" and not self.approval_handler.approve(tool, call):
            return self._permission_denied(call, "approval was not granted")
        return self.tools.run(call)

    @staticmethod
    def _permission_denied(call: ToolCall, reason: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Permission denied for {call.name}: {reason}",
            error_code="permission_denied",
            details={"reason": reason},
        )

    @staticmethod
    def _trace_retry(
        attempt: int,
        delay: float,
        error: ProviderError,
        *,
        trace: bool,
    ) -> None:
        if not trace:
            return
        print(
            f"\n[provider retry] attempt={attempt} delay={delay:.2f}s "
            f"code={error.code}"
        )
