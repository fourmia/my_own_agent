# Stage 1: Agent Loop

这一阶段的目标不是做一个聪明的 agent，而是做一个你能完全看懂的 agent。

真实 coding agent 可能很复杂，但它的核心通常逃不开这个循环：

```text
message history -> LLM decision -> optional tool call -> observation -> next decision
```

## 本项目里的对应关系

- `mini_agent/schema.py`
  定义 agent 系统中流动的数据：消息、工具调用、工具结果、模型响应。

- `mini_agent/tools.py`
  定义工具注册表。模型不直接碰文件系统或 shell，而是通过工具提出结构化请求。

- `mini_agent/llm.py`
  定义模型接口。现在的 `MockLLM` 是规则驱动的假模型，之后会替换成真实 LLM。

- `mini_agent/agent.py`
  定义主循环。它不负责“聪明”，只负责把模型、工具、观察结果串起来。

## 为什么先用 MockLLM

如果一开始就接真实模型，你会同时面对 API、prompt、JSON、模型幻觉和工具执行问题。

`MockLLM` 把不确定性拿掉，让你先看清楚系统骨架：

```text
用户输入: read README.md
模型决策: 调用 read_file(path="README.md")
工具执行: 读取文件
观察结果: 文件内容
模型总结: 我观察到了结果
```

## 你现在应该理解的 3 件事

1. 工具调用必须是结构化的，而不是让模型输出一段自然语言命令。
2. 工具结果必须回到消息历史里，否则模型无法基于执行结果继续推理。
3. `max_steps` 是安全阀，防止模型或工具错误导致无限循环。

## 练习

请你自己完成一个小工具：

```text
get_file_stats(path) -> 返回文件字节数、行数、字符数
```

建议步骤：

1. 在 `mini_agent/tools.py` 里新增内部函数 `get_file_stats`。
2. 在 `create_default_registry` 里注册工具。
3. 在 `mini_agent/llm.py` 里让 `MockLLM` 支持 `stats README.md`。
4. 在 `tests/test_tools.py` 里写一个测试。
5. 运行 `python -m pytest -q`。

这个练习很小，但它会让你亲手走过“新增 agent 能力”的完整路径。

