from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from mini_agent.llm import LLM
from mini_agent.provider_errors import ProviderError
from mini_agent.schema import ModelRequest, ModelTurn


RetryObserver = Callable[[int, float, ProviderError], None]


@dataclass(frozen=True)
class ProviderRetryPolicy:
    """只重试尚未形成完整 ModelTurn 的 Provider 请求。

    工具执行不在这个策略内部，因此网络重试永远不会自动重放已经发生的
    文件修改、测试或 Shell 副作用。
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    sleeper: Callable[[float], None] = field(
        default=time.sleep,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")

    def generate(
        self,
        llm: LLM,
        request: ModelRequest,
        *,
        on_retry: RetryObserver | None = None,
    ) -> ModelTurn:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return llm.generate(request)
            except ProviderError as error:
                if not error.retryable or attempt >= self.max_attempts:
                    raise
                delay = self._delay_for(error, attempt)
                if on_retry is not None:
                    on_retry(attempt + 1, delay, error)
                self.sleeper(delay)
        raise AssertionError("retry loop exhausted without returning or raising")

    def _delay_for(self, error: ProviderError, failed_attempt: int) -> float:
        retry_after = error.details.get("retry_after")
        if retry_after is not None:
            try:
                requested = max(0.0, float(retry_after))
            except (TypeError, ValueError):
                requested = -1.0
            if requested >= 0:
                return min(requested, self.max_delay_seconds)
        exponential = self.base_delay_seconds * (2 ** (failed_attempt - 1))
        return min(exponential, self.max_delay_seconds)


__all__ = ["ProviderRetryPolicy"]
