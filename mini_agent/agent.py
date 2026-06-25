from __future__ import annotations

from dataclasses import dataclass

from mini_agent.llm import LLM
from mini_agent.schema import Message
from mini_agent.tools import ToolRegistry


@dataclass(frozen=True)
class AgentResult:
    """Agent 运行结束后的汇总结果。

    messages 被完整保留下来，是为了后续做调试、评估和会话持久化。
    很多 agent 问题不是出在最终答案，而是出在中间某一步的错误观察或错误决策。
    """

    answer: str
    steps: int
    messages: list[Message]


class Agent:
    """把 LLM、工具和消息历史串起来的最小 agent runtime。

    这个类故意不包含“智能”逻辑。它只负责运行循环：

        1. 把历史消息交给模型
        2. 如果模型要调用工具，就执行工具
        3. 把工具结果作为 observation 放回消息历史
        4. 重复，直到模型完成或达到 max_steps

    真实 coding agent 的核心也类似，只是工具更多、权限更细、上下文管理更复杂。
    """

    def __init__(self, llm: LLM, tools: ToolRegistry, max_steps: int = 6) -> None:
        # 依赖注入：Agent 不关心具体模型是哪家，也不关心工具怎么实现。
        # 这让我们可以先用 MockLLM 学习，之后再替换成真实 LLM。
        self.llm = llm
        self.tools = tools

        # max_steps 是安全阀。真实模型可能误判、循环调用工具，甚至被 prompt injection 带偏。
        # 没有步数上限，agent 可能无限运行、疯狂烧钱或不断修改文件。
        self.max_steps = max_steps

    def run(self, task: str, trace: bool = True) -> AgentResult:
        # 对话历史从用户任务开始。后面每一轮 assistant/tool 消息都会追加进来。
        # 注意：这里没有丢弃历史，所以 MockLLM 可以在看到 tool 消息后做最终总结。
        messages = [Message("user", task)]

        for step in range(1, self.max_steps + 1):
            # 让模型基于“完整消息历史”和“当前可用工具列表”做下一步决策。
            # 在真实 LLM 接入里，tools 通常会被转换成 JSON Schema / function calling schema。
            response = self.llm.next(messages, self.tools.list_tools())
            messages.append(Message("assistant", response.content))

            if trace:
                print(f"\n[step {step}] assistant")
                print(response.content)

            # 没有工具调用就代表模型给出了普通回答，agent 可以结束。
            if response.tool_call is None:
                return AgentResult(response.content, step, messages)

            call = response.tool_call
            if trace:
                print(f"\n[step {step}] tool_call: {call.name}({call.arguments})")

            # 工具是 agent 观察外部世界的方式：读文件、跑命令、搜索、访问 API 等。
            # 工具执行结果不直接返回给用户，而是先回到模型上下文里。
            result = self.tools.run(call)
            observation = result.content if result.ok else f"ERROR: {result.content}"
            messages.append(Message("tool", observation))

            if trace:
                print(f"\n[step {step}] observation")
                print(observation)

            # 有些模型响应可以同时包含“我完成了”和工具调用。
            # 这里保留这个字段，是为了后面学习更复杂的协议时有扩展点。
            if response.done:
                return AgentResult(response.content, step, messages)

        # 到达上限时明确停止，并把停止原因也写进消息历史，方便调试。
        final = f"Stopped after reaching max_steps={self.max_steps}."
        messages.append(Message("assistant", final))
        return AgentResult(final, self.max_steps, messages)
