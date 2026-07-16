# Mini Agent Learning Project

这个项目会带你从零实现一个“可观察、可验证、可扩展”的智能体。

你已经有 Python 经验，所以我们不从语法开始，而是直接学习智能体工程的核心闭环：

```text
用户目标 -> 模型思考 -> 选择工具 -> 执行工具 -> 观察结果 -> 继续决策 -> 完成
```

## 当前阶段：Stage 1 / Iteration 2

我们先实现一个不依赖真实大模型的最小智能体。它使用 `MockLLM` 模拟模型决策，用 JSON Schema 校验工具参数，并能根据结构化错误执行一次确定性的文件名恢复，让你理解多轮 agent loop、消息历史、错误分类和停止策略。

## 运行

```bash
python -m mini_agent "list files"
python -m mini_agent "search pyproject"
python -m mini_agent "read README.md"
python -m mini_agent "stats README.md"
python -m mini_agent "read REDME.md"
python -m mini_agent "run python --version"
```

## 你要重点看

- `mini_agent/agent.py`：智能体主循环
- `mini_agent/llm.py`：模型接口和 mock 决策器
- `mini_agent/tools.py`：工具注册、参数校验、工具执行
- `mini_agent/schema.py`：消息、工具调用、工具结果的数据结构

## 迭代文档

- `docs/stage-1-agent-loop.md`：最小 Agent Loop 的核心概念
- `docs/iteration-1-tool-contracts.md`：严格工具 Schema、参数校验和结构化错误
- `docs/iteration-2-error-recovery.md`：完整消息历史、错误分类和多轮文件恢复

## 学习任务

1. 先运行上面的命令，观察每一轮 trace。
2. 阅读 `mini_agent/agent.py`，确认你能说清楚为什么需要 `max_steps`。
3. 运行 `read REDME.md`，观察 `read_file -> list_files -> read_file` 恢复链路。
4. 阅读 `llm.py` 的恢复状态机，确认你能解释为什么自动重试必须有限制。

下一阶段我们会接入真实 LLM，并让模型通过 structured output 选择工具。

