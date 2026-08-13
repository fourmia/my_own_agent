from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from mini_agent.agent import Agent
from mini_agent.artifacts import FileArtifactStore
from mini_agent.journal import SQLiteEventStore, inspect_run
from mini_agent.llm import LLM, MockLLM
from mini_agent.permissions import (
    ApprovalHandler,
    ConsoleApprovalHandler,
    DenyApprovalHandler,
    FullAccessPermissionPolicy,
    OnRequestPermissionPolicy,
    PermissionPolicy,
    ReadOnlyPermissionPolicy,
)
from mini_agent.provider_errors import ProviderError
from mini_agent.providers import OpenAIResponsesAdapter
from mini_agent.retry import ProviderRetryPolicy
from mini_agent.resume import (
    ResumeAnalyzer,
    ResumeExecutor,
    ResumeRejectedError,
)
from mini_agent.snapshots import ArtifactSnapshotStore
from mini_agent.tools import create_coding_registry, create_default_registry


CODING_SYSTEM_PROMPT = """You are a careful coding agent working inside one repository.
Use get_file_stats before apply_patch and pass the observed sha256.
After the latest successful change, inspect git_diff and run run_tests before claiming completion.
Never claim that an edit or test succeeded without its tool result."""


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


def create_permission_policy(
    provider: str,
    requested_mode: str | None,
) -> tuple[PermissionPolicy, ApprovalHandler]:
    mode = requested_mode or ("full" if provider == "mock" else "on-request")
    if mode == "read-only":
        return ReadOnlyPermissionPolicy(), DenyApprovalHandler()
    if mode == "on-request":
        return OnRequestPermissionPolicy(), ConsoleApprovalHandler()
    if mode == "full":
        return FullAccessPermissionPolicy(), DenyApprovalHandler()
    raise ValueError(f"unsupported approval mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mini learning agent.")
    parser.add_argument(
        "task",
        nargs="?",
        help="Task for the agent, for example: 'list files'",
    )
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
    parser.add_argument(
        "--max-provider-attempts",
        type=positive_int,
        default=3,
        help="Maximum attempts for retryable Provider failures.",
    )
    parser.add_argument(
        "--approval-mode",
        choices=("read-only", "on-request", "full"),
        help=(
            "Tool permission mode. Defaults to full for mock and on-request for "
            "real Providers."
        ),
    )
    parser.add_argument(
        "--journal-path",
        default=".mini_agent/runs.db",
        help="SQLite execution journal path relative to the workspace.",
    )
    parser.add_argument(
        "--inspect-run",
        metavar="RUN_ID",
        help="Print one persisted run timeline instead of starting an Agent.",
    )
    parser.add_argument(
        "--analyze-resume",
        metavar="RUN_ID",
        help="Analyze resumability without calling a Provider or executing tools.",
    )
    parser.add_argument(
        "--resume-run",
        metavar="RUN_ID",
        help="Resume a run only after conservative recovery checks pass.",
    )
    parser.add_argument(
        "--reconcile-run",
        metavar="RUN_ID",
        help="Inspect started-only Patch outcomes without executing anything.",
    )
    parser.add_argument("--no-trace", action="store_true", help="Only print final answer.")
    args = parser.parse_args()

    root = Path.cwd()
    journal_path = Path(args.journal_path)
    if not journal_path.is_absolute():
        journal_path = root / journal_path
    artifacts_path = journal_path.parent / "artifacts"

    inspection_options = [
        option
        for option in (
            args.inspect_run,
            args.analyze_resume,
            args.resume_run,
            args.reconcile_run,
        )
        if option is not None
    ]
    if inspection_options:
        if len(inspection_options) > 1:
            parser.error(
                "inspection, reconciliation and resume options are mutually exclusive"
            )
        if args.task is not None:
            parser.error("task cannot be combined with inspection options")
        try:
            with SQLiteEventStore(journal_path) as event_store:
                target_run = (
                    args.inspect_run
                    or args.analyze_resume
                    or args.resume_run
                    or args.reconcile_run
                )
                if args.analyze_resume:
                    if args.provider == "openai" and not args.model:
                        parser.error(
                            "--model is required when analyzing an OpenAI run"
                        )
                    tools = (
                        create_default_registry(root)
                        if args.provider == "mock"
                        else create_coding_registry(root)
                    )
                    permission_policy, _ = create_permission_policy(
                        args.provider,
                        args.approval_mode,
                    )
                    llm_identity = (
                        "mini_agent.llm.MockLLM"
                        if args.provider == "mock"
                        else f"mini_agent.providers.openai_responses.OpenAIResponsesAdapter:{args.model}"
                    )
                    analyzer = ResumeAnalyzer(
                        event_store,
                        ArtifactSnapshotStore(
                            FileArtifactStore(artifacts_path)
                        ),
                        tools,
                        llm_identity=llm_identity,
                        permission_policy_identity=(
                            f"{type(permission_policy).__module__}."
                            f"{type(permission_policy).__qualname__}"
                        ),
                        max_steps=6,
                        require_verification=True,
                    )
                    print(
                        json.dumps(
                            asdict(analyzer.analyze(target_run)),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    return
                if args.reconcile_run:
                    analyzer = ResumeAnalyzer(
                        event_store,
                        ArtifactSnapshotStore(FileArtifactStore(artifacts_path)),
                        (
                            create_default_registry(root)
                            if args.provider == "mock"
                            else create_coding_registry(root)
                        ),
                        llm_identity="reconciliation-only",
                    )
                    print(
                        json.dumps(
                            [
                                asdict(item)
                                for item in analyzer.reconcile(target_run)
                            ],
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    return
                if args.resume_run:
                    llm = create_llm(args, parser)
                    tools = (
                        create_default_registry(root)
                        if args.provider == "mock"
                        else create_coding_registry(root)
                    )
                    permission_policy, approval_handler = create_permission_policy(
                        args.provider,
                        args.approval_mode,
                    )
                    agent = Agent(
                        llm=llm,
                        tools=tools,
                        permission_policy=permission_policy,
                        approval_handler=approval_handler,
                        retry_policy=ProviderRetryPolicy(
                            max_attempts=args.max_provider_attempts,
                        ),
                        system_prompt=(
                            CODING_SYSTEM_PROMPT if args.provider == "openai" else None
                        ),
                        event_store=event_store,
                        snapshot_store=ArtifactSnapshotStore(
                            FileArtifactStore(artifacts_path)
                        ),
                    )
                    try:
                        result = ResumeExecutor(agent).run(
                            target_run,
                            trace=not args.no_trace,
                        )
                    except ResumeRejectedError as error:
                        print(
                            json.dumps(
                                asdict(error.analysis),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                        parser.exit(2, "resume rejected\n")
                    print(f"\n[run_id]\n{result.run_id}")
                    print("\n[final]")
                    print(result.answer)
                    return
                events = event_store.load(target_run)
        except ProviderError as error:
            parser.exit(1, f"provider error [{error.code}]: {error}\n")
        except RuntimeError as error:
            parser.exit(1, f"resume runtime error: {error}\n")
        except (sqlite3.Error, ValueError) as error:
            parser.exit(1, f"journal error: {error}\n")
        if not events:
            parser.exit(1, f"run not found: {target_run}\n")
        inspection = inspect_run(events)
        print("[run]")
        print(json.dumps(asdict(inspection), ensure_ascii=False, sort_keys=True))
        print("\n[events]")
        for event in events:
            payload = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
            print(
                f"{event.sequence:04d} {event.created_at} "
                f"{event.event_type} {payload}"
            )
        return

    if args.task is None:
        parser.error("task is required unless an inspection or resume option is used")

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
            else create_coding_registry(root)
        )
        permission_policy, approval_handler = create_permission_policy(
            args.provider,
            args.approval_mode,
        )
        with SQLiteEventStore(journal_path) as event_store:
            snapshot_store = ArtifactSnapshotStore(
                FileArtifactStore(artifacts_path)
            )
            agent = Agent(
                llm=llm,
                tools=tools,
                permission_policy=permission_policy,
                approval_handler=approval_handler,
                retry_policy=ProviderRetryPolicy(
                    max_attempts=args.max_provider_attempts,
                ),
                system_prompt=(
                    CODING_SYSTEM_PROMPT if args.provider == "openai" else None
                ),
                event_store=event_store,
                snapshot_store=snapshot_store,
            )
            result = agent.run(args.task, trace=not args.no_trace)
    except ProviderError as error:
        # 不打印 headers、请求体或密钥，只暴露稳定错误码和安全消息。
        parser.exit(1, f"provider error [{error.code}]: {error}\n")
    except sqlite3.Error as error:
        parser.exit(1, f"journal error: {error}\n")

    print(f"\n[run_id]\n{result.run_id}")
    print("\n[final]")
    print(result.answer)


if __name__ == "__main__":
    main()

