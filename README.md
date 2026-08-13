# Mini Agent Learning Project

这个项目会带你从零实现一个“可观察、可验证、可扩展”的智能体。

你已经有 Python 经验，所以我们不从语法开始，而是直接学习智能体工程的核心闭环：

```text
用户目标 -> 模型思考 -> 选择工具 -> 执行工具 -> 观察结果 -> 继续决策 -> 完成
```

## 当前阶段：Stage 1 / Iteration 5B3

项目现在已经形成最小可审计执行闭环：同时支持零配置 `MockLLM` 和 OpenAI Responses 非流式 Adapter，Runtime 支持 Provider 重试、工具权限审批、基于文件哈希的局部修改、修改后的完成守卫、SQLite append-only Execution Journal，以及内容寻址 Artifact Store 和版本化 Run Snapshot。OpenAI SDK 类型仍被限制在 Provider 模块内，默认测试不访问网络也不需要 API key。

## 运行

```bash
python -m mini_agent "list files"
python -m mini_agent "search pyproject"
python -m mini_agent "read README.md"
python -m mini_agent "stats README.md"
python -m mini_agent "read REDME.md"
python -m mini_agent "run python --version"
```

每次 CLI 运行默认把事件写入 `.mini_agent/runs.db`，并输出 `run_id`。检查一次运行：

```powershell
python -m mini_agent --inspect-run RUN_ID
python -m mini_agent --inspect-run RUN_ID --journal-path .mini_agent/runs.db
```

Journal 记录事件顺序、调用关联、权限决定、Provider 尝试和验证版本，但不保存原始任务、模型正文、Patch 内容或工具输出；这些内容只保存长度与 SHA-256 摘要。

分析一次未完成运行能否进入后续恢复流程：

```powershell
python -m mini_agent --analyze-resume RUN_ID
python -m mini_agent --provider openai --model MODEL_ID --analyze-resume RUN_ID
```

分析命令不会请求模型或执行工具。5B2 允许显式恢复分析通过的运行：

```powershell
python -m mini_agent --resume-run RUN_ID --approval-mode full
python -m mini_agent --provider openai --model MODEL_ID --approval-mode on-request --resume-run RUN_ID
```

必须使用原运行相同的 Provider、模型、审批模式和工具集。只允许 `model_request` 或只读 `tool_batch` 恢复；未知 write、execute、mutation 副作用仍然阻断。CLI 尚不会从旧快照自动选择运行配置。

对 started-only 的受控 `apply_patch`，可以先执行只读协调：

```powershell
python -m mini_agent --reconcile-run RUN_ID
```

它只报告 `not_executed`、`executed` 或 `indeterminate`，不请求模型、不运行工具，也不自动重放 Patch。只有工作区精确匹配 Patch 的预期前后状态时，5B3 才能把协调结果写回恢复消息历史。

如果你是第一次学习 Agent，请先阅读：

- [新手实现指南](docs/beginner-implementation-guide.md)：从最小 Agent Loop 到 Iteration 5A 的完整实现主线、实验和自测题。

接入 OpenAI Provider：

```powershell
python -m pip install -e ".[openai]"
$env:OPENAI_API_KEY="your-key"
python -m mini_agent "分析项目结构" --provider openai --model MODEL_ID
python -m mini_agent "修复失败测试" --provider openai --model MODEL_ID --approval-mode on-request
```

`--model` 必须显式提供，项目不会硬编码“最新模型”。OpenAI 默认使用 `on-request`：读取自动允许，`apply_patch` 和 `run_tests` 在执行前请求终端审批。还可以显式选择：

- `--approval-mode read-only`：拒绝所有写入和进程执行。
- `--approval-mode on-request`：读取自动允许，写入和测试逐次审批。
- `--approval-mode full`：允许注册表中的全部工具，适合受控 CI 或测试环境。

`--max-provider-attempts` 控制可重试 Provider 请求的最大尝试次数。重试范围只包含 `LLM.generate()`，已经执行的工具不会被网络重试自动重放。

真实 Provider 使用受控 coding registry，只暴露 `apply_patch`、`git_diff` 和参数受限的 `run_tests`；不暴露整文件 `write_file` 或任意 `run_command`。

## 你要重点看

- `mini_agent/agent.py`：智能体主循环
- `mini_agent/llm.py`：模型接口和 mock 决策器
- `mini_agent/providers/openai_responses.py`：OpenAI Responses 请求/响应映射
- `mini_agent/provider_errors.py`：Provider 错误分类
- `mini_agent/retry.py`：Provider 指数退避和 `Retry-After` 策略
- `mini_agent/permissions.py`：allow / ask / deny 权限决策与审批器
- `mini_agent/verification.py`：修改后的 diff / 测试完成守卫
- `mini_agent/journal.py`：运行事件 Schema、SQLite EventStore 和中断检查
- `mini_agent/artifacts.py`：内容寻址 Artifact Store 和完整性校验
- `mini_agent/snapshots.py`：版本化运行快照、工具与工作区指纹
- `mini_agent/recorder.py`：Journal 与 Snapshot 的统一写入顺序
- `mini_agent/resume.py`：恢复分析器和受限恢复执行器
- `mini_agent/resume_types.py`：协调结果的数据合同
- `mini_agent/tools.py`：工具注册、参数校验、工具执行
- `mini_agent/schema.py`：消息、工具调用、工具结果、模型请求和模型轮次

## 迭代文档

- [新手实现指南](docs/beginner-implementation-guide.md)：新手主线教程，建议首先阅读
- `docs/stage-1-agent-loop.md`：最小 Agent Loop 的核心概念
- `docs/iteration-1-tool-contracts.md`：严格工具 Schema、参数校验和结构化错误
- `docs/iteration-2-error-recovery.md`：完整消息历史、错误分类和多轮文件恢复
- `docs/iteration-3a-domain-protocol.md`：`call_id`、多工具调用、请求快照和 Provider 合同
- `docs/iteration-3b-openai-responses.md`：OpenAI Adapter、reasoning item 重放和错误边界
- `docs/iteration-4-safe-execution.md`：重试边界、权限审批、受控 Patch 和完成验证
- `docs/iteration-5a-execution-journal.md`：append-only Journal、事件顺序与恢复风险分类
- `docs/iteration-5b1-recovery-foundation.md`：Artifact、Snapshot、恢复指纹和只读分析决策
- `docs/iteration-5b2-safe-resume-execution.md`：原子恢复 claim、状态重建和只读恢复边界
- `docs/iteration-5b3-patch-reconciliation.md`：Patch 证据、三态协调和不重放原则
- `docs/iteration-3-real-llm-design.md`：真实 LLM Adapter、调用关联和多工具协议设计
- `docs/adr/0001-own-runtime-use-provider-adapters.md`：自研 Runtime、按边界复用开源方案的架构决策

## 学习任务

1. 先运行上面的命令，观察每一轮 trace。
2. 阅读 `mini_agent/agent.py`，确认你能说清楚为什么需要 `max_steps`。
3. 运行 `read REDME.md`，观察 `read_file -> list_files -> read_file` 恢复链路。
4. 阅读 `llm.py` 的恢复状态机，确认你能解释为什么自动重试必须有限制。
5. 阅读 `tests/test_safety_runtime.py`，观察为什么 Provider retry 不会重复执行工具。
6. 运行测试，重点观察拒绝审批、陈旧哈希和完成阻断三个失败路径。
7. 执行一个 CLI 任务，再用输出的 `run_id` 检查完整事件时间线。

下一阶段是 Iteration 6：把 reconciliation 变成工具级合同。每个有副作用工具都应显式声明证据 Schema、状态查询、幂等键、补偿与人工确认路径；任意 Shell、外部 API 和数据库操作不能按 Patch 规则推断。

