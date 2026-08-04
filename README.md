# Mini Agent Learning Project

这个项目会带你从零实现一个“可观察、可验证、可扩展”的智能体。

你已经有 Python 经验，所以我们不从语法开始，而是直接学习智能体工程的核心闭环：

```text
用户目标 -> 模型思考 -> 选择工具 -> 执行工具 -> 观察结果 -> 继续决策 -> 完成
```

## 当前阶段：Stage 1 / Iteration 3B

项目现在同时支持零配置 `MockLLM` 和 OpenAI Responses 非流式 Adapter。Runtime 支持带 `call_id` 的零到多个工具调用，OpenAI SDK 类型被限制在 Provider 模块内，默认测试不访问网络也不需要 API key。

## 运行

```bash
python -m mini_agent "list files"
python -m mini_agent "search pyproject"
python -m mini_agent "read README.md"
python -m mini_agent "stats README.md"
python -m mini_agent "read REDME.md"
python -m mini_agent "run python --version"
```

接入 OpenAI Provider：

```powershell
python -m pip install -e ".[openai]"
$env:OPENAI_API_KEY="your-key"
python -m mini_agent "分析项目结构" --provider openai --model MODEL_ID
```

`--model` 必须显式提供，项目不会硬编码“最新模型”。真实 Provider 当前只暴露只读文件工具；在权限审批和进程沙箱完成前，不开放 `write_file` 与 `run_command`。

## 你要重点看

- `mini_agent/agent.py`：智能体主循环
- `mini_agent/llm.py`：模型接口和 mock 决策器
- `mini_agent/providers/openai_responses.py`：OpenAI Responses 请求/响应映射
- `mini_agent/provider_errors.py`：Provider 错误分类
- `mini_agent/tools.py`：工具注册、参数校验、工具执行
- `mini_agent/schema.py`：消息、工具调用、工具结果、模型请求和模型轮次

## 迭代文档

- `docs/stage-1-agent-loop.md`：最小 Agent Loop 的核心概念
- `docs/iteration-1-tool-contracts.md`：严格工具 Schema、参数校验和结构化错误
- `docs/iteration-2-error-recovery.md`：完整消息历史、错误分类和多轮文件恢复
- `docs/iteration-3a-domain-protocol.md`：`call_id`、多工具调用、请求快照和 Provider 合同
- `docs/iteration-3b-openai-responses.md`：OpenAI Adapter、reasoning item 重放和错误边界
- `docs/iteration-3-real-llm-design.md`：真实 LLM Adapter、调用关联和多工具协议设计
- `docs/adr/0001-own-runtime-use-provider-adapters.md`：自研 Runtime、按边界复用开源方案的架构决策

## 学习任务

1. 先运行上面的命令，观察每一轮 trace。
2. 阅读 `mini_agent/agent.py`，确认你能说清楚为什么需要 `max_steps`。
3. 运行 `read REDME.md`，观察 `read_file -> list_files -> read_file` 恢复链路。
4. 阅读 `llm.py` 的恢复状态机，确认你能解释为什么自动重试必须有限制。
5. 运行测试，重点观察 `tests/test_provider_contract.py` 中的多工具顺序和调用关联。

下一阶段是 Iteration 3C：在不改变最终 `ModelTurn` 语义的前提下增加流式事件。

