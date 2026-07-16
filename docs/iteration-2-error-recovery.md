# Iteration 2：多轮错误恢复

## 目标

Iteration 1 让工具失败变成了结构化 JSON，但 `MockLLM` 仍然只会复述 observation。Iteration 2 让错误真正参与下一步决策：

```text
read REDME.md
  -> read_file(path="REDME.md")
  -> file_not_found
  -> list_files()
  -> 找到 README.md
  -> read_file(path="README.md")
  -> 文件内容
  -> 最终回答
```

整个过程保持确定性，不依赖 API key，便于观察消息历史、恢复策略和停止条件。

## 1. 为什么 Message 必须保留结构化调用

上一轮的消息只保存：

```python
Message(role="assistant", content="I should inspect the requested file.")
Message(role="tool", content="...")
```

模型无法可靠知道：

- assistant 实际调用了哪个工具；
- 工具参数中的原始路径是什么；
- 当前 observation 属于哪个调用；
- 同一个动作已经尝试了几次。

Iteration 2 为 `Message` 增加：

```python
tool_call: ToolCall | None
tool_name: str | None
```

因此消息历史同时保存自然语言和机器协议。`MockLLM` 可以扫描历史中的 `assistant.tool_call`，而不是从日志或自然语言中猜测状态。

生产系统通常还会增加 `tool_call_id`，用于关联同一轮里的多个并行工具调用。

## 2. 错误分类

笼统的 `tool_execution_error` 无法指导恢复。现在 Registry 会把常见失败映射成稳定错误码：

| 错误码 | 来源 | 当前策略 |
| --- | --- | --- |
| `file_not_found` | 目标文件不存在 | 列出文件并寻找候选 |
| `path_escape` | 路径逃出工作区 | 停止，不自动绕过安全边界 |
| `invalid_encoding` | 文件不是 UTF-8 文本 | 停止，等待未来二进制工具 |
| `command_timeout` | 命令超过执行上限 | 停止，保留超时信息 |
| `command_failed` | Shell 返回非零退出码 | 标记失败并保留退出码和输出 |
| `tool_execution_error` | 未分类异常 | 安全兜底，不自动猜测 |

特别注意：非零退出码现在是工具失败，而不是带有 `exit_code=3` 的“成功结果”。这能防止模型把测试失败误判成执行成功。

## 3. 恢复状态机

`MockLLM._after_tool()` 根据 `tool_name`、错误码和历史调用进入不同分支。

### 读取成功

```text
read_file -> 普通文本 observation -> 生成最终回答
```

### 首次文件不存在

```text
read_file -> file_not_found -> list_files
```

### 获得文件列表

1. 从最近的结构化 `read_file` ToolCall 取出原始路径。
2. 同时比较完整路径和 basename 的相似度。
3. 选择分数最高的候选。
4. 只有相似度不低于 `0.6` 才重试。

basename 比较可以把 `REDME.md` 匹配到 `README.md`；完整路径比较则兼顾目录拼写错误。分数相同的候选按文件名确定性排序，保证测试和回放结果稳定。

`list_files` 和 `search_files` 会过滤 `.git`、`__pycache__`、`.pytest_cache`、虚拟环境和 `*.egg-info`。这些目录既会消耗上下文 Token，也可能让缓存中的同名文件干扰恢复决策。

### 没有可信候选

Agent 明确回答找不到近似文件，然后停止。生产 Agent 的重要能力之一是“证据不足时不行动”。

### 重试仍然失败

自动读取最多重试一次。如果候选文件在列出后又被删除，Agent 会停止，而不是在 `read_file` 和 `list_files` 之间无限循环。

## 4. 两层停止保护

Iteration 2 有两种不同的停止机制：

1. **策略停止**：恢复状态机判断没有候选或重试已经失败。
2. **运行时停止**：无论模型如何决策，`Agent.max_steps` 都会强制终止循环。

策略停止提高效率，`max_steps` 则是不可省略的最终安全阀。测试中的 `RepeatingToolLLM` 会永远请求 `list_files`，用于证明 Runtime 仍会按上限停止。

## 5. 为什么暂时不用真实 LLM

这一轮故意使用规则状态机，因为它具备：

- 固定输入产生固定输出；
- 不需要网络和 API key；
- 可以精确断言调用顺序；
- 能把 Runtime Bug 与模型随机性分开。

接入真实模型后，恢复决策会由 Tool Calling 模型完成，但消息协议、错误码、重试限制和 `max_steps` 仍然保留。

## 6. 测试范围

新增测试覆盖：

- 拼错文件名后的四步恢复；
- 没有相似候选时诚实停止；
- 永久请求工具时命中 `max_steps`；
- 文件不存在的错误分类；
- 路径逃逸的错误分类；
- 非 UTF-8 文件的错误分类；
- Shell 非零退出码被识别为失败。
- 文件列表过滤运行时缓存和安装元数据。

运行：

```bash
python -m pytest -q
```

## 下一步

Iteration 3 可以接入第一个真实 LLM Adapter，并把 `Tool.parameters` 转换为供应商原生 Tool Calling Schema。建议继续保留 `MockLLM`，将它作为 Runtime 的确定性回归测试模型。
