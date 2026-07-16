# Iteration 1：严格工具契约

## 目标

这一轮把工具系统从“依赖开发者默契的字典”升级为可以被程序验证的契约：

```text
模型生成 ToolCall
  -> ToolRegistry 查找工具
  -> JSON Schema 校验参数
  -> 执行工具
  -> ToolResult 转成 observation
  -> 写回消息历史
```

这仍然是 Stage 1：模型继续使用确定性的 `MockLLM`，但模型和工具之间的边界已经接近真实 function calling 系统。

## 1. JSON Schema 成为工具参数协议

每个工具的 `parameters` 现在都是 Draft 2020-12 JSON Schema。例如读取文件工具要求：

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "minLength": 1,
      "description": "Workspace-relative file path."
    }
  },
  "required": ["path"],
  "additionalProperties": false
}
```

这里分别约束了对象类型、字段类型、空字符串、必填字段和多余字段。真实 LLM 接入后，同一份 Schema 可以同时用于模型工具定义和运行时防御。

## 2. 校验发生在统一信任边界

`ToolRegistry.register()` 在启动时检查 Schema 自身是否合法，并为每个工具缓存 validator。`ToolRegistry.run()` 在调用工具函数前校验模型参数。

这样做有两个作用：

1. 错误 Schema 会在工具注册时立即失败。
2. 非法模型参数不会进入文件系统或 Shell 工具。

当前实现返回第一个确定性错误。生产系统还可以把全部错误用于调试，但写回模型时通常应控制长度。

## 3. 结构化错误 observation

成功结果继续使用原始字符串。失败结果统一转换为 JSON：

```json
{
  "ok": false,
  "tool": "read_file",
  "error": {
    "code": "invalid_arguments",
    "message": "Invalid arguments at $: 'path' is a required property",
    "details": {
      "location": "$",
      "validator": "required"
    }
  }
}
```

当前定义了三个基础错误码：

| 错误码 | 含义 | 未来模型策略 |
| --- | --- | --- |
| `unknown_tool` | 模型请求了未注册工具 | 重新查看工具列表 |
| `invalid_arguments` | 参数不符合 Schema | 修正参数后重试 |
| `tool_execution_error` | 工具内部执行失败 | 根据错误原因恢复或停止 |

稳定错误码比 `ERROR: ...` 文本更适合多轮纠错、统计和自动评估。

## 4. `get_file_stats` 完整能力链路

新增命令：

```bash
python -m mini_agent "stats README.md"
```

对应链路包括：

1. `MockLLM` 识别 `stats <path>`。
2. 生成 `ToolCall("get_file_stats", {"path": ...})`。
3. Registry 按 Schema 校验 `path`。
4. 工具复用工作区路径沙箱并读取 UTF-8 文件。
5. 返回路径、磁盘字节数、文本行数和字符数。
6. Agent 在第二轮根据 observation 生成最终回答。

`bytes` 是文件在磁盘上的真实大小，`characters` 是解码后的 Python 字符数量。在 Windows 上，文本文件可能使用 CRLF，因此两个换行字节会被读取为一个换行字符；两项统计不一定相等。

## 5. 验证范围

本轮测试覆盖：

- 原有文件读取行为
- 工作区路径逃逸阻止
- 缺少必填参数时的结构化错误
- 文件统计结果
- Agent 对 `stats` 命令的两轮调用

运行：

```bash
python -m pytest -q
```

当前结果为 `6 passed`。

## 后续迭代

Iteration 2 已实现，详见 `docs/iteration-2-error-recovery.md`。Agent 现在会识别 `file_not_found`，列出工作区文件，选择可信的相似文件名并进行一次有限重试。
