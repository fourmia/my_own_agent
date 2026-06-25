from __future__ import annotations

import argparse
from pathlib import Path

from mini_agent.agent import Agent
from mini_agent.llm import MockLLM
from mini_agent.tools import create_default_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mini learning agent.")
    parser.add_argument("task", help="Task for the agent, for example: 'list files'")
    parser.add_argument("--no-trace", action="store_true", help="Only print final answer.")
    args = parser.parse_args()

    root = Path.cwd()
    agent = Agent(llm=MockLLM(), tools=create_default_registry(root))
    result = agent.run(args.task, trace=not args.no_trace)

    print("\n[final]")
    print(result.answer)


if __name__ == "__main__":
    main()

