from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from mini_agent.journal import (
    EventStore,
    EventType,
    content_summary,
    new_run_id,
)
from mini_agent.recorder import RunRecorder
from mini_agent.llm import LLM
from mini_agent.permissions import (
    ApprovalHandler,
    DenyApprovalHandler,
    PermissionPolicy,
    ReadOnlyPermissionPolicy,
)
from mini_agent.provider_errors import ProviderError
from mini_agent.retry import ProviderRetryPolicy
from mini_agent.schema import Message, ModelRequest, ToolCall, ToolResult
from mini_agent.snapshots import (
    RunSnapshot,
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotStore,
    tool_schema_fingerprint,
    verification_from_snapshot,
    workspace_identity,
    workspace_state_sha256,
)
from mini_agent.resume_types import PatchReconciliation
from mini_agent.tools import Tool, ToolRegistry
from mini_agent.verification import VerificationState


@dataclass(frozen=True)
class AgentResult:
    """Agent 运行结束后的汇总结果。

    messages 被完整保留下来，是为了后续做调试、评估和会话持久化。
    很多 Agent 问题不是出在最终答案，而是出在中间某一步的错误观察或错误决策。
    """

    answer: str
    steps: int
    messages: list[Message]
    run_id: str | None = None


class Agent:
    """把 LLM、工具和消息历史串起来的最小 Agent Runtime。

    一次模型轮次可以返回零到多个工具调用。Runtime 按 Provider 顺序执行，
    并且等这一批结果全部进入历史后才请求下一轮模型。权限、日志和验证状态
    都位于 handler 外层，因此副作用发生前后的关键边界可以被审计。
    """

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        max_steps: int = 6,
        *,
        permission_policy: PermissionPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        retry_policy: ProviderRetryPolicy | None = None,
        require_verification: bool = True,
        system_prompt: str | None = None,
        event_store: EventStore | None = None,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        # 依赖注入：Agent 不关心具体模型是哪家，也不关心工具怎么实现。
        # 这让我们可以先用 MockLLM 学习，之后再替换成真实 Provider Adapter。
        self.llm = llm
        self.tools = tools
        self.permission_policy = permission_policy or ReadOnlyPermissionPolicy()
        self.approval_handler = approval_handler or DenyApprovalHandler()
        self.retry_policy = retry_policy or ProviderRetryPolicy()
        self.require_verification = require_verification
        self.system_prompt = system_prompt
        self.recorder = RunRecorder(event_store, snapshot_store)

        # max_steps 是安全阀。真实模型可能误判、循环调用工具，甚至被 prompt injection 带偏。
        # 没有步数上限，Agent 可能无限运行、疯狂烧钱或不断修改文件。
        self.max_steps = max_steps

    def run(self, task: str, trace: bool = True) -> AgentResult:
        run_id = new_run_id()
        self._record(
            run_id,
            "run_started",
            {
                "task": content_summary(task),
                "max_steps": self.max_steps,
                "require_verification": self.require_verification,
                "tools": [tool.name for tool in self.tools.list_tools()],
            },
        )
        try:
            return self._run_loop(task, trace=trace, run_id=run_id)
        except Exception as error:
            # 不允许记录失败覆盖真正的 Runtime 异常；尤其当原始异常就是日志写入
            # 失败时，第二次写入也可能失败，但仍应把第一次异常交给宿主。
            try:
                self._record(
                    run_id,
                    "run_failed",
                    {
                        "exception_type": type(error).__name__,
                        "provider_error_code": (
                            error.code if isinstance(error, ProviderError) else None
                        ),
                    },
                )
            except Exception:  # noqa: BLE001 - preserve the original Runtime failure
                pass
            raise

    def _run_loop(self, task: str, *, trace: bool, run_id: str) -> AgentResult:
        # 对话历史从用户任务开始。后面每一轮 assistant/tool 消息都会追加进来。
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message("system", self.system_prompt))
        messages.append(Message("user", task))
        seen_call_ids: set[str] = set()
        verification = VerificationState()
        self._save_snapshot(
            run_id,
            messages,
            seen_call_ids,
            verification,
            next_step=1,
            resume_point="model_request",
        )
        return self._continue_loop(
            messages,
            seen_call_ids,
            verification,
            start_step=1,
            trace=trace,
            run_id=run_id,
        )

    def _resume_from_snapshot(
        self,
        snapshot: RunSnapshot,
        *,
        expected_journal_sequence: int,
        reconciliations: tuple[PatchReconciliation, ...] = (),
        trace: bool,
    ) -> AgentResult:
        """仅供 ResumeExecutor 调用；在原子 claim 后恢复已验证状态。"""

        if snapshot.max_steps != self.max_steps:
            raise ValueError("Agent max_steps does not match the persisted run")
        if snapshot.require_verification != self.require_verification:
            raise ValueError("Agent verification policy does not match the persisted run")
        if snapshot.resume_point == "run_completion":
            raise ValueError("an incomplete completion commit cannot be resumed")

        self.recorder.claim_resume(
            snapshot.run_id,
            expected_journal_sequence,
            {
                "from_snapshot_sequence": snapshot.last_event_sequence,
                "previous_journal_sequence": expected_journal_sequence,
                "resume_point": snapshot.resume_point,
                "next_step": snapshot.next_step,
            },
        )
        try:
            self._validate_claimed_snapshot(snapshot, reconciliations)
            return self._continue_from_claimed_snapshot(
                snapshot,
                reconciliations=reconciliations,
                trace=trace,
            )
        except Exception as error:
            try:
                self._record(
                    snapshot.run_id,
                    "run_failed",
                    {
                        "exception_type": type(error).__name__,
                        "provider_error_code": (
                            error.code if isinstance(error, ProviderError) else None
                        ),
                    },
                )
            except Exception:  # noqa: BLE001 - preserve the resumed Runtime failure
                pass
            raise

    def _validate_claimed_snapshot(
        self,
        snapshot: RunSnapshot,
        reconciliations: tuple[PatchReconciliation, ...],
    ) -> None:
        """在 claim 后再次检查易变化的本地身份，缩短竞态窗口。"""

        if snapshot.llm_identity != self.llm_identity():
            raise RuntimeError("LLM identity changed after resume claim")
        if snapshot.permission_policy_identity != self.permission_policy_identity():
            raise RuntimeError("permission policy changed after resume claim")
        if snapshot.tool_schema_sha256 != tool_schema_fingerprint(
            self.tools.list_tools()
        ):
            raise RuntimeError("tool schema changed after resume claim")
        current_workspace = workspace_identity(self.tools.root)
        reconciled_after_states = {
            result.expected_after_workspace_sha256
            for result in reconciliations
            if (
                result.status == "executed"
                and result.expected_before_workspace_sha256
                == snapshot.workspace.state_sha256
            )
        }
        if (
            snapshot.workspace.root != current_workspace.root
            or snapshot.workspace.git_head != current_workspace.git_head
            or (
                snapshot.workspace.state_sha256 != current_workspace.state_sha256
                and current_workspace.state_sha256 not in reconciled_after_states
            )
        ):
            raise RuntimeError("workspace identity changed after resume claim")

    def _continue_from_claimed_snapshot(
        self,
        snapshot: RunSnapshot,
        *,
        reconciliations: tuple[PatchReconciliation, ...],
        trace: bool,
    ) -> AgentResult:
        """恢复 claim 成功后重建内存状态；失败由外层记录 run_failed。"""

        messages = list(snapshot.messages)
        seen_call_ids = set(snapshot.seen_call_ids)
        verification = verification_from_snapshot(snapshot.verification)
        start_step = snapshot.next_step

        if snapshot.resume_point == "tool_batch":
            pending_calls = messages[-1].tool_calls
            reconciliation_by_call_id = {
                result.call_id: result for result in reconciliations
            }
            for call in pending_calls:
                if trace:
                    print(
                        f"\n[resumed tool_call {snapshot.next_step}] "
                        f"{call.name}({call.arguments})"
                    )
                tool = self.tools.get(call.name)
                reconciliation = reconciliation_by_call_id.get(call.call_id)
                if (
                    tool is not None
                    and tool.name == "apply_patch"
                    and reconciliation is not None
                    and reconciliation.status in {"not_executed", "executed"}
                ):
                    result = self._reconciled_patch_result(call, reconciliation)
                    self._record(
                        snapshot.run_id,
                        "tool_execution_reconciled",
                        {
                            "call_id": call.call_id,
                            "tool": call.name,
                            "status": reconciliation.status,
                            "reason": reconciliation.reason,
                        },
                    )
                else:
                    if tool is None or tool.permission != "read" or (
                        tool.effect == "workspace_mutation"
                    ):
                        raise RuntimeError("unsafe tool batch passed resume analysis")
                    result = self._execute_call(call, tool, run_id=snapshot.run_id)
                before_verification = verification.snapshot()
                verification.observe(tool, result)
                after_verification = verification.snapshot()
                if after_verification != before_verification:
                    self._record(
                        snapshot.run_id,
                        "verification_updated",
                        {
                            "call_id": call.call_id,
                            "tool": call.name,
                            "effect": tool.effect,
                            "state": after_verification,
                        },
                    )
                messages.append(
                    Message(
                        "tool",
                        result.to_observation(),
                        tool_call_id=result.call_id,
                        tool_name=result.name,
                    )
                )
            start_step += 1
            self._save_snapshot(
                snapshot.run_id,
                messages,
                seen_call_ids,
                verification,
                next_step=start_step,
                resume_point="model_request",
            )

        return self._continue_loop(
            messages,
            seen_call_ids,
            verification,
            start_step=start_step,
            trace=trace,
            run_id=snapshot.run_id,
        )

    def _continue_loop(
        self,
        messages: list[Message],
        seen_call_ids: set[str],
        verification: VerificationState,
        *,
        start_step: int,
        trace: bool,
        run_id: str,
    ) -> AgentResult:
        """从一个已持久化的 model_request 边界继续公共 Agent Loop。"""

        for step in range(start_step, self.max_steps + 1):
            # 只把 ToolDefinition 交给模型，避免 Provider 看到本地 Python handler。
            request = ModelRequest(
                # 每一轮发送历史快照。若直接传入可变的 messages，后续追加的
                # assistant/tool 消息会篡改已经“发送”的请求，破坏审计和回放。
                messages=list(messages),
                tools=self.tools.list_definitions(),
            )
            request_id = (
                f"{run_id}:step:{step}:request:"
                f"{self.recorder.current_sequence(run_id) + 1}"
            )
            self._record(
                run_id,
                "model_request_started",
                {
                    "request_id": request_id,
                    "step": step,
                    "message_count": len(request.messages),
                    "tool_count": len(request.tools),
                },
            )
            attempt = {"current": 1}

            def on_retry(
                next_attempt: int,
                delay: float,
                error: ProviderError,
            ) -> None:
                self._record(
                    run_id,
                    "model_request_failed",
                    {
                        "request_id": request_id,
                        "step": step,
                        "attempt": attempt["current"],
                        "error_code": error.code,
                        "retrying": True,
                        "retry_delay_seconds": delay,
                        "details": error.details,
                    },
                )
                attempt["current"] = next_attempt
                self._trace_retry(next_attempt, delay, error, trace=trace)

            try:
                response = self.retry_policy.generate(
                    self.llm,
                    request,
                    on_retry=on_retry,
                )
            except ProviderError as error:
                self._record(
                    run_id,
                    "model_request_failed",
                    {
                        "request_id": request_id,
                        "step": step,
                        "attempt": attempt["current"],
                        "error_code": error.code,
                        "retrying": False,
                        "details": error.details,
                    },
                )
                raise

            self._record(
                run_id,
                "model_request_completed",
                {
                    "request_id": request_id,
                    "step": step,
                    "attempt": attempt["current"],
                    "finish_reason": response.finish_reason,
                    "provider_response_id": response.provider_response_id,
                    "content": content_summary(response.content),
                    "tool_calls": [
                        {"call_id": call.call_id, "tool": call.name}
                        for call in response.tool_calls
                    ],
                    "usage": (
                        {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if response.usage is not None
                        else None
                    ),
                },
            )

            # assistant 消息必须持久化整批结构化调用。后续模型需要知道每个
            # observation 对应哪个 call_id 和原始参数，而不能依赖打印日志猜测。
            messages.append(
                Message(
                    "assistant",
                    response.content,
                    tool_calls=response.tool_calls,
                )
            )

            if trace:
                print(f"\n[step {step}] assistant")
                print(response.content)

            # ToolCall 一旦由 Provider 发出，就占用 session 内的 call_id。先验证并
            # 持久化 tool_batch 快照，随后才执行任何 handler；恢复时因此拥有完整参数。
            if response.tool_calls:
                batch_call_ids = [call.call_id for call in response.tool_calls]
                if len(batch_call_ids) != len(set(batch_call_ids)):
                    raise ValueError("duplicate ToolCall.call_id within one tool batch")
                reused = sorted(set(batch_call_ids).intersection(seen_call_ids))
                if reused:
                    raise ValueError(
                        f"duplicate ToolCall.call_id in Agent session: {reused[0]}"
                    )
                seen_call_ids.update(batch_call_ids)
                self._save_snapshot(
                    run_id,
                    messages,
                    seen_call_ids,
                    verification,
                    next_step=step,
                    resume_point="tool_batch",
                )

            # 没有工具调用代表模型给出了最终文本、拒绝或不完整结果；这些状态都
            # 需要交给上层展示，不能因为没有工具就继续盲目循环。
            if not response.tool_calls:
                missing = verification.missing_requirements()
                if (
                    self.require_verification
                    and response.finish_reason == "completed"
                    and missing
                ):
                    blocked = verification.blocking_message()
                    messages.append(Message("system", blocked))
                    if trace:
                        print(f"\n[step {step}] completion_blocked")
                        print(blocked)
                    self._record(
                        run_id,
                        "completion_blocked",
                        {
                            "step": step,
                            "required_tools": missing,
                            "verification": verification.snapshot(),
                        },
                    )
                    self._save_snapshot(
                        run_id,
                        messages,
                        seen_call_ids,
                        verification,
                        next_step=step + 1,
                        resume_point="model_request",
                    )
                    continue
                return self._complete_run(
                    run_id,
                    response.content,
                    step,
                    messages,
                    verification=verification,
                    seen_call_ids=seen_call_ids,
                    finish_reason=response.finish_reason or "completed",
                )

            # 逐个执行同一 ModelTurn 中的调用。一个工具失败不阻止同批次后续调用；
            # 是否 fail-fast 属于未来 Permission/Policy 层，而不是 handler 的职责。
            for call in response.tool_calls:
                if trace:
                    print(
                        f"\n[step {step}] tool_call[{call.call_id}]: "
                        f"{call.name}({call.arguments})"
                    )

                # 权限检查发生在 handler 之前。deny/未批准 ask 会生成普通 ToolResult，
                # 模型可以看到拒绝原因，但任何文件或进程副作用都不会发生。
                tool = self.tools.get(call.name)
                result = self._execute_call(call, tool, run_id=run_id)
                before_verification = verification.snapshot()
                verification.observe(tool, result)
                after_verification = verification.snapshot()
                if after_verification != before_verification:
                    self._record(
                        run_id,
                        "verification_updated",
                        {
                            "call_id": call.call_id,
                            "tool": call.name,
                            "effect": tool.effect if tool is not None else "none",
                            "state": after_verification,
                        },
                    )
                observation = result.to_observation()
                messages.append(
                    Message(
                        "tool",
                        observation,
                        tool_call_id=result.call_id,
                        tool_name=result.name,
                    )
                )

                if trace:
                    print(f"\n[step {step}] observation[{result.call_id}]")
                    print(observation)

            self._save_snapshot(
                run_id,
                messages,
                seen_call_ids,
                verification,
                next_step=step + 1,
                resume_point="model_request",
            )

        # 到达上限时明确停止，并把停止原因也写进消息历史，方便调试。
        final = f"Stopped after reaching max_steps={self.max_steps}."
        messages.append(Message("assistant", final))
        return self._complete_run(
            run_id,
            final,
            self.max_steps,
            messages,
            verification=verification,
            seen_call_ids=seen_call_ids,
            finish_reason="max_steps",
        )

    def _execute_call(
        self,
        call: ToolCall,
        tool: Tool | None,
        *,
        run_id: str,
    ) -> ToolResult:
        if tool is None:
            result = self.tools.run(call)
            self._record(
                run_id,
                "tool_execution_skipped",
                {
                    "call_id": call.call_id,
                    "tool": call.name,
                    "reason": "unknown_tool",
                    "error_code": result.error_code,
                },
            )
            return result

        decision = self.permission_policy.evaluate(tool, call)
        approved: bool | None = None
        if decision.outcome == "ask":
            approved = self.approval_handler.approve(tool, call)
        self._record(
            run_id,
            "permission_decided",
            {
                "call_id": call.call_id,
                "tool": call.name,
                "permission": tool.permission,
                "outcome": decision.outcome,
                "reason": decision.reason,
                "approved": approved,
            },
        )
        if decision.outcome == "deny":
            result = self._permission_denied(call, decision.reason)
            self._record_skipped_tool(run_id, call, tool, result)
            return result
        if decision.outcome == "ask" and not approved:
            result = self._permission_denied(call, "approval was not granted")
            self._record_skipped_tool(run_id, call, tool, result)
            return result

        self._record(
            run_id,
            "tool_execution_started",
            {
                "call_id": call.call_id,
                "tool": call.name,
                "permission": tool.permission,
                "effect": tool.effect,
                "arguments": self._arguments_summary(call),
                "reconciliation": self._reconciliation_evidence(call, tool),
            },
        )
        result = self.tools.run(call)
        self._record(
            run_id,
            "tool_execution_completed",
            {
                "call_id": call.call_id,
                "tool": call.name,
                "ok": result.ok,
                "error_code": result.error_code,
                "content": content_summary(result.content),
                "details": result.details,
            },
        )
        return result

    def _record_skipped_tool(
        self,
        run_id: str,
        call: ToolCall,
        tool: Tool,
        result: ToolResult,
    ) -> None:
        self._record(
            run_id,
            "tool_execution_skipped",
            {
                "call_id": call.call_id,
                "tool": call.name,
                "permission": tool.permission,
                "effect": tool.effect,
                "error_code": result.error_code,
            },
        )

    @staticmethod
    def _arguments_summary(call: ToolCall) -> dict[str, Any]:
        serialized = json.dumps(
            call.arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        target = call.arguments.get("path")
        if target is None:
            target = call.arguments.get("targets")
        return {
            "keys": sorted(call.arguments),
            "target": target,
            "content": content_summary(serialized),
        }

    def _reconciliation_evidence(
        self,
        call: ToolCall,
        tool: Tool,
    ) -> dict[str, Any] | None:
        """为受控 Patch 生成不含正文、可在崩溃后验证的状态证据。"""

        if tool.name != "apply_patch" or tool.effect != "workspace_mutation":
            return None
        try:
            path = self.tools.resolve_workspace_path(str(call.arguments["path"]))
            before = path.read_bytes()
            before_sha256 = hashlib.sha256(before).hexdigest()
            if before_sha256 != str(call.arguments["expected_sha256"]).lower():
                return None
            text = before.decode("utf-8")
            old_text = str(call.arguments["old_text"])
            new_text = str(call.arguments["new_text"])
            if old_text == new_text or text.count(old_text) != 1:
                return None
            after = text.replace(old_text, new_text, 1).encode("utf-8")
            relative = path.relative_to(self.tools.root).as_posix()
            return {
                "kind": "exact_file_patch_v1",
                "path": relative,
                "before_sha256": before_sha256,
                "after_sha256": hashlib.sha256(after).hexdigest(),
                "workspace_before_sha256": workspace_state_sha256(
                    self.tools.root
                ),
                "workspace_after_sha256": workspace_state_sha256(
                    self.tools.root,
                    replacements={relative: after},
                ),
            }
        except (KeyError, OSError, TypeError, UnicodeDecodeError, ValueError):
            return None

    @staticmethod
    def _reconciled_patch_result(
        call: ToolCall,
        reconciliation: PatchReconciliation,
    ) -> ToolResult:
        if reconciliation.status == "executed":
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=True,
                content=(
                    "Patch outcome was reconciled after interruption: the workspace "
                    "matches the expected after state. Inspect the diff and run tests."
                ),
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=(
                "Patch outcome was reconciled after interruption: the workspace "
                "matches the expected before state, so the patch did not execute. "
                "Do not reuse this call ID; inspect state and issue a new call if needed."
            ),
            error_code="reconciled_not_executed",
            details={"reconciliation_status": reconciliation.status},
        )

    def _complete_run(
        self,
        run_id: str,
        answer: str,
        steps: int,
        messages: list[Message],
        *,
        verification: VerificationState,
        seen_call_ids: set[str],
        finish_reason: str,
    ) -> AgentResult:
        self._save_snapshot(
            run_id,
            messages,
            seen_call_ids,
            verification,
            next_step=steps,
            resume_point="run_completion",
        )
        self._record(
            run_id,
            "run_completed",
            {
                "steps": steps,
                "finish_reason": finish_reason,
                "answer": content_summary(answer),
                "message_count": len(messages),
            },
        )
        return AgentResult(answer, steps, messages, run_id=run_id)

    def _record(
        self,
        run_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> None:
        self.recorder.record(run_id, event_type, payload)

    def _save_snapshot(
        self,
        run_id: str,
        messages: list[Message],
        seen_call_ids: set[str],
        verification: VerificationState,
        *,
        next_step: int,
        resume_point: str,
    ) -> None:
        if self.recorder.snapshot_store is None:
            return
        snapshot = RunSnapshot(
            run_id=run_id,
            snapshot_version=SNAPSHOT_SCHEMA_VERSION,
            last_event_sequence=self.recorder.current_sequence(run_id),
            next_step=next_step,
            resume_point=resume_point,
            messages=list(messages),
            seen_call_ids=sorted(seen_call_ids),
            verification=verification.snapshot(),
            max_steps=self.max_steps,
            require_verification=self.require_verification,
            llm_identity=self.llm_identity(),
            tool_schema_sha256=tool_schema_fingerprint(self.tools.list_tools()),
            workspace=workspace_identity(self.tools.root),
            permission_policy_identity=self.permission_policy_identity(),
        )
        self.recorder.save_snapshot(snapshot)

    def llm_identity(self) -> str:
        model = getattr(self.llm, "model", None)
        suffix = f":{model}" if model else ""
        return f"{type(self.llm).__module__}.{type(self.llm).__qualname__}{suffix}"

    def permission_policy_identity(self) -> str:
        return (
            f"{type(self.permission_policy).__module__}."
            f"{type(self.permission_policy).__qualname__}"
        )

    @staticmethod
    def _permission_denied(call: ToolCall, reason: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            content=f"Permission denied for {call.name}: {reason}",
            error_code="permission_denied",
            details={"reason": reason},
        )

    @staticmethod
    def _trace_retry(
        attempt: int,
        delay: float,
        error: ProviderError,
        *,
        trace: bool,
    ) -> None:
        if not trace:
            return
        print(
            f"\n[provider retry] attempt={attempt} delay={delay:.2f}s "
            f"code={error.code}"
        )
