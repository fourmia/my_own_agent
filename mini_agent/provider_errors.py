from __future__ import annotations

from typing import Any


class ProviderError(RuntimeError):
    """模型 Provider 失败的稳定领域错误。

    Provider 错误与 ToolResult 完全分离：前者表示模型请求尚未形成可接受的
    ModelTurn，后者表示某个工具已经被 Agent 执行并得到成功或失败结果。
    """

    code = "provider_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details or {}


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    retryable = True


class ProviderConnectionError(ProviderError):
    code = "provider_connection_error"
    retryable = True


class ProviderRateLimitedError(ProviderError):
    code = "provider_rate_limited"
    retryable = True


class ProviderServerError(ProviderError):
    code = "provider_server_error"
    retryable = True


class ProviderAuthError(ProviderError):
    code = "provider_auth_error"


class ProviderBadRequestError(ProviderError):
    code = "provider_bad_request"


class ProviderProtocolError(ProviderError):
    code = "provider_protocol_error"


__all__ = [
    "ProviderAuthError",
    "ProviderBadRequestError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderProtocolError",
    "ProviderRateLimitedError",
    "ProviderServerError",
    "ProviderTimeoutError",
]
