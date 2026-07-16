# ADR 0001：自研 Agent Runtime，使用 Provider Adapter

- 状态：Accepted
- 日期：2026-07-16
- 决策范围：Stage 2 及后续生产化演进

## 背景

项目需要同时满足两个目标：

1. 精确理解 Agent Loop、消息、工具调用、错误恢复、上下文和权限等关键细节。
2. 吸收成熟开源项目和官方 SDK 的优秀实现，避免重复制造基础设施。

如果直接把核心建立在 LangChain Agent 或其他高层 Agent Framework 上，开发速度会提高，但内部消息协议、工具调用关联、恢复状态和停止条件会被框架类型接管。当前项目的主要学习对象恰好是这些边界。

如果完全拒绝第三方库，则需要自行维护 HTTP、认证、退避、序列化、遥测和大量 Provider 兼容逻辑，同样不合理。

## 决策

采用以下结构：

```text
mini_agent 自有领域模型
├── Agent Loop
├── Message / ToolCall / ToolResult
├── Session / Turn / Item（后续）
├── Tool Registry
├── Error Protocol
├── Context Policy（后续）
└── Permission Policy（后续）
            │
            ▼
Provider Adapter Interface
├── Mock Adapter
├── OpenAI Responses Adapter
├── Anthropic Adapter（后续）
└── OpenAI-compatible Adapter（后续）
            │
            ▼
官方 SDK / HTTP 客户端
```

核心 Runtime 不依赖 LangChain 消息、工具或 Agent 类型。Provider SDK 类型只允许出现在对应 Adapter 模块内部。

## 允许复用的内容

优先使用成熟库处理通用基础设施：

- OpenAI、Anthropic 等官方 SDK；
- JSON Schema/Pydantic 校验；
- HTTP、重试和超时基础设施；
- OpenTelemetry；
- 数据库驱动和迁移工具。

吸收开源项目时优先学习：

- 协议边界；
- 状态机；
- 错误分类；
- 测试案例；
- 沙箱和权限原则；
- 上下文与压缩策略。

直接复制源代码前必须单独检查许可证、依赖、维护成本和框架耦合。本 ADR 和 Iteration 3 设计阶段没有复制第三方实现代码。

## 为什么当前不使用 LangChain Agent

LangChain 的优势是 Provider、Retriever、Tool 和可观测生态丰富，适合快速组合业务应用。但当前阶段需要直接控制：

- `call_id` 如何关联调用和结果；
- 多个工具调用如何排序和执行；
- 非法参数如何变成 observation；
- 哪些错误可以重试；
- 历史如何持久化和回放；
- `max_steps` 如何定义；
- 权限检查位于哪一层。

这些是 Runtime 的核心职责，不应被高层框架隐式决定。

## LangGraph 的保留位置

未来出现 Planner、Executor、Reviewer、人工审批和长时间工作流后，可以把 LangGraph 作为可替换的 Orchestrator Adapter：

```text
LangGraph / 自研 Orchestrator
            │
            ▼
mini_agent Agent Runtime
            │
            ▼
Policy -> Tool Executor -> Sandbox
```

LangGraph 不应拥有底层 Tool Registry、权限策略或 Provider Domain Schema。

## 结果

正面结果：

- 核心协议透明且可测试；
- Provider 可以替换；
- 开源方案可以按模块吸收；
- 后续引入编排框架不需要推翻底层 Runtime。

代价：

- 需要自己定义更多领域类型；
- 初期 Provider 接入速度慢于高层框架；
- 必须维护 Adapter 合约测试；
- 多 Provider 兼容需要显式 Schema 转换层。

## 复审条件

出现以下情况时重新评估：

- 自研编排状态机明显阻碍业务交付；
- LangGraph 能在不侵入底层协议的前提下显著减少可靠性工作；
- Provider Adapter 的重复代码无法通过共享接口收敛；
- 团队无法承担 Runtime 的长期维护成本。
