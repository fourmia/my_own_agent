# Mini Agent Learning Project

这个项目会带你从零实现一个“可观察、可验证、可扩展”的智能体。

你已经有 Python 经验，所以我们不从语法开始，而是直接学习智能体工程的核心闭环：

```text
用户目标 -> 模型思考 -> 选择工具 -> 执行工具 -> 观察结果 -> 继续决策 -> 完成
```

## 当前阶段：Stage 1 / Iteration 1

我们先实现一个不依赖真实大模型的最小智能体。它使用 `MockLLM` 模拟模型决策，并用 JSON Schema 描述和校验工具参数，让你先理解 agent loop、tool schema、observation、迭代上限和结构化工具错误。

## 运行

```bash
python -m mini_agent "list files"
python -m mini_agent "search pyproject"
python -m mini_agent "read README.md"
python -m mini_agent "stats README.md"
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

## 学习任务

1. 先运行上面的 4 个命令，观察每一轮 trace。
2. 阅读 `mini_agent/agent.py`，确认你能说清楚为什么需要 `max_steps`。
3. 阅读 `tools.py` 中的 JSON Schema，观察缺少参数时的结构化错误。
4. 阅读 `get_file_stats` 的实现和测试，理解新增工具的完整链路。

下一阶段我们会接入真实 LLM，并让模型通过 structured output 选择工具。

