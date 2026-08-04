# Iteration 3A：领域协议与多工具执行

- 状态：Implemented
- 日期：2026-08-04
- 范围：真实 Provider 接入前的 Runtime 核心

## 目标

这一阶段不访问网络，也不引入任何 Provider SDK。目标是先把真实模型会用到的最小协议固定下来，避免在接入 OpenAI Responses 或其他 Provider 时同时修改 Agent Loop、工具执行和消息历史。

## 领域类型

```text
ModelRequest
  messages: list[Message]
  tools: list[ToolDefinition]

ModelTurn
  content: str
  tool_calls: list[ToolCall]
  finish_reason: completed | tool_calls | incomplete | refusal
  provider_response_id: optional
  usage: optional

ToolCall
  call_id: str
  name: str
  arguments: dict

ToolResult
  call_id: str
  name: str
  ok: bool
  content: str
  error_code: optional
  details: dict
```

`call_id` 是调用和结果之间的协议关联键。工具名称只用于日志和防御性检查，不能替代 `call_id`，因为同一轮可以多次调用同一个工具。

## 一轮执行语义

```text
用户消息
  -> ModelRequest（历史快照 + ToolDefinition）
  -> ModelTurn（按 Provider 顺序的多个 ToolCall）
  -> ToolRegistry.run(call-A)
  -> ToolRegistry.run(call-B)
  -> 两个带原 call_id 的 ToolResult 写入历史
  -> 下一次 ModelRequest
```

Runtime 当前明确采用以下规则：

1. 保留模型返回的工具调用顺序。
2. 同一轮的调用逐个执行，暂不并行。
3. 一个调用失败不会丢弃同批次的后续调用。
4. 所有结果写入历史后，才请求下一轮模型。
5. `max_steps` 统计模型轮次，不统计单个工具数量。

请求使用 `messages=list(messages)` 创建快照。不能把可变的运行中列表直接放进 `ModelRequest`，否则下一轮追加的消息会篡改已经发送过的请求，破坏日志审计和确定性回放。

## 结束原因校验

`ModelTurn` 会拒绝矛盾状态：

- `tool_calls` 非空时，`finish_reason` 必须是 `tool_calls`；
- `finish_reason=tool_calls` 时，至少要有一个工具调用；
- `completed`、`refusal`、`incomplete` 不得携带待执行调用。
- 同一轮和同一 Session 内的 `call_id` 不得重复。

旧的 `LLMResponse`、`LLM.next()` 和单数 `tool_call` 字段暂时保留为迁移兼容入口。新 Provider 必须实现 `LLM.generate(ModelRequest)`，并显式返回 `ModelTurn`。

## 合同测试

`tests/test_provider_contract.py` 用不依赖网络的 `ScriptedProvider` 验证 Runtime，而不是测试某个模型是否聪明：

- 多工具按返回顺序执行；
- 工具结果的 `call_id` 与请求一致；
- 未注册工具失败后，后续工具仍然执行；
- Provider 工具定义不包含本地 Python handler；
- 矛盾的 `finish_reason` 被拒绝；
- 错误 observation 保留 `call_id`。

## 尚未实现

本阶段有意不实现：

- OpenAI 或 Anthropic SDK；
- 并行工具执行；
- 流式事件；
- 权限审批和进程沙箱；
- 上下文压缩和 Session 持久化。

下一阶段 Iteration 3B 将只新增 Provider Adapter，把真实 SDK 映射到本文件定义的领域类型，不改变 Runtime 的执行语义。
