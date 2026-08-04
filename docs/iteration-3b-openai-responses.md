# Iteration 3B：OpenAI Responses Adapter

- 状态：Implemented
- 日期：2026-08-04
- SDK：OpenAI Python 2.x，可选依赖
- 传输：非流式

## 目标

3B 在不修改 Agent Loop 语义的前提下，把 OpenAI Responses API 映射到 3A 已固定的 `ModelRequest` 和 `ModelTurn`。核心 Runtime 不导入 OpenAI SDK，默认 Mock 测试也不需要 API key。

```text
Agent Runtime
  -> ModelRequest
  -> OpenAIResponsesAdapter
  -> client.responses.create(...)
  -> ordered response.output items
  -> ModelTurn
  -> Agent 执行 ToolCall
```

## SDK 协议依据

实现时核对了本地安装的官方 OpenAI Python SDK `2.26.0` 生成类型：

- `Responses.create` 接收 `input`、`tools`、`model`、`max_output_tokens` 等参数；
- Function Tool 使用 `type=function`、`name`、`parameters`、`strict`；
- Function Call 参数是 JSON 字符串，并通过 `call_id` 关联结果；
- Tool Result 使用 `type=function_call_output`、相同 `call_id` 和 `output`；
- `response.output` 是有序 item 列表，不能假设第一项是文本；
- `response.usage` 提供 input、output 和 total token；
- `response.status` 可以是 `completed`、`incomplete`、`failed` 等状态。

当前会话已安装 OpenAI Docs MCP，但工具尚未被应用重新加载，官方开发者文档入口在当前网络仍返回 403。因此字段实现以官方 SDK 生成类型为直接依据；应用重启后进入 3C 前重新核对官方 Streaming 文档。

参考：

- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses/create)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Python Response type](https://github.com/openai/openai-python/blob/v2.26.0/src/openai/types/responses/response.py)
- [OpenAI Python Function Tool type](https://github.com/openai/openai-python/blob/v2.26.0/src/openai/types/responses/function_tool_param.py)

## 请求映射

| 领域数据 | Responses input |
| --- | --- |
| user/system Message | `type=message` |
| assistant 可见文本 | assistant message |
| assistant ToolCall | `type=function_call` |
| tool Message | `type=function_call_output` |
| ToolDefinition | strict function tool |

Adapter 使用 `store=false` 并重发本地历史，mini_agent 的 Message 历史仍是事实来源。SDK 自动重试被设置为零，避免未来 Runtime Retry Policy 与 SDK 形成双重退避。

工具 Schema 在发送前再次验证：

- 顶层必须是 object；
- 必须定义 properties；
- `additionalProperties` 必须为 false；
- strict 模式下每个 property 都必须出现在 required 中。

## Reasoning Item 重放

Responses 推理模型可能在 function call 前返回 Provider 专用 reasoning item。如果只把领域层的 assistant 文本和 ToolCall 重新构造成输入，会丢失推理状态并降低下一轮正确率。

Adapter 因此在内存中按有序 `call_id` 缓存原始 output items：

```text
response.output
  reasoning
  message
  function_call
       ↓ cache
下一轮 input
  reasoning
  message
  function_call
  function_call_output
```

这些 artifact 不进入通用 `Message` 类型，也不会让 OpenAI SDK 类型泄漏到 Runtime。Session 落盘与跨进程恢复将在后续持久化迭代实现。

## 参数解析失败

函数参数 JSON 无法解析时，Adapter 不丢失整轮，也不伪造成功参数：

1. 保留原始 `raw_arguments`；
2. 保留 Provider 的 `call_id`；
3. 在 `ToolCall.argument_error` 记录解析错误；
4. ToolRegistry 返回关联的 `invalid_arguments` ToolResult；
5. 模型下一轮可以纠正参数。

## Provider 错误边界

| 错误码 | 默认可重试 |
| --- | --- |
| `provider_timeout` | 是 |
| `provider_connection_error` | 是 |
| `provider_rate_limited` | 是 |
| `provider_server_error` | 是 |
| `provider_auth_error` | 否 |
| `provider_bad_request` | 否 |
| `provider_protocol_error` | 否 |

错误对象保留安全的 status code、request id 和 Retry-After 等诊断信息，但 CLI 不打印请求头、Authorization 或 API key。

## CLI 与安全策略

```powershell
python -m pip install -e ".[openai]"
$env:OPENAI_API_KEY="your-key"
python -m mini_agent "分析项目结构" --provider openai --model MODEL_ID
```

- API key 只通过 SDK 标准环境变量读取；
- 不提供 `--api-key`，避免进入 Shell history；
- 模型 ID 必须显式指定；
- Mock 仍是默认 Provider；
- OpenAI 当前只获得读取、搜索和统计工具；
- 写文件和 Shell 要等 Permission Policy 与沙箱完成后才能开放。

## 测试

默认测试使用内存 Fake Client，覆盖：

- 消息、函数调用、函数结果和 strict tools 映射；
- 混合 message/reasoning/function_call 输出；
- 多个 function call 顺序；
- reasoning artifact 下一轮重放；
- 非法 JSON 参数关联；
- refusal、incomplete 和 failed 状态；
- 未知 output item fail closed；
- timeout、429 和 Retry-After 分类；
- 真实 Provider 的只读工具限制。

Live smoke test 默认跳过。只有同时提供以下配置才访问网络：

```powershell
$env:RUN_OPENAI_LIVE_TEST="1"
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="MODEL_ID"
pytest -m live tests/test_openai_live.py
```

## 尚未实现

- 流式 text/tool argument delta；
- Runtime 级有限重试和退避；
- Permission Policy、审批和 Shell 沙箱；
- Provider artifact 持久化；
- Anthropic 或 OpenAI-compatible 第二 Adapter。

下一阶段 3C 只改变传输过程：流式事件最终仍必须组装成与 3B 相同的 `ModelTurn`。
