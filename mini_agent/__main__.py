from __future__ import annotations

import argparse
from pathlib import Path

from mini_agent.agent import Agent
from mini_agent.llm import LLM, MockLLM
from mini_agent.provider_errors import ProviderError
from mini_agent.providers import OpenAIResponsesAdapter
from mini_agent.tools import create_default_registry, create_read_only_registry


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def create_llm(args: argparse.Namespace, parser: argparse.ArgumentParser) -> LLM:
    if args.provider == "mock":
        return MockLLM()
    if not args.model:
        parser.error("--model is required when --provider=openai")
    return OpenAIResponsesAdapter(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mini learning agent.")
    parser.add_argument("task", help="Task for the agent, for example: 'list files'")
    parser.add_argument(
        "--provider",
        choices=("mock", "openai"),
        default="mock",
        help="Model provider. Mock remains the zero-configuration default.",
    )
    parser.add_argument(
        "--model",
        help="Provider model ID. Required for OpenAI; no latest model is hard-coded.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        help="Optional upper bound for one model response.",
    )
    parser.add_argument("--no-trace", action="store_true", help="Only print final answer.")
    args = parser.parse_args()

    root = Path.cwd()
    try:
        llm = create_llm(args, parser)
    except ProviderError as error:
        parser.exit(1, f"provider error [{error.code}]: {error}\n")
    except RuntimeError as error:
        # 可选依赖缺失属于本地配置问题，给出短消息而不是 SDK 导入堆栈。
        parser.exit(1, f"provider configuration error: {error}\n")

    try:
        tools = (
            create_default_registry(root)
            if args.provider == "mock"
            else create_read_only_registry(root)
        )
        agent = Agent(llm=llm, tools=tools)
        result = agent.run(args.task, trace=not args.no_trace)
    except ProviderError as error:
        # 不打印 headers、请求体或密钥，只暴露稳定错误码和安全消息。
        parser.exit(1, f"provider error [{error.code}]: {error}\n")

    print("\n[final]")
    print(result.answer)


if __name__ == "__main__":
    main()

