from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from mini_agent.llm import LLM
from mini_agent.provider_errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderConnectionError,
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitedError,
    ProviderServerError,
    ProviderTimeoutError,
)
from mini_agent.schema import ModelRequest, ModelTurn, ToolCall, ToolDefinition, Usage


class OpenAIResponsesAdapter(LLM):
    """OpenAI Responses API 的非流式 Provider Adapter。

    该模块是业务代码中唯一允许接触 OpenAI Python SDK 的位置。为了让默认
    Mock 模式和单元测试不依赖可选包，SDK 在创建默认客户端和映射异常时才导入。
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        max_output_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI model must be a non-empty string")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = client if client is not None else self._create_default_client(timeout)
        # key 是一个 assistant 工具轮次中的有序 call_id。缓存原始 output items
        # 使后续无状态请求能重放 reasoning/message/function_call，而不污染领域 Schema。
        self._output_item_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    def generate(self, request: ModelRequest) -> ModelTurn:
        payload = self.build_request_payload(request)
        try:
            response = self._client.responses.create(**payload)
        except Exception as exc:  # noqa: BLE001 - SDK exceptions cross the adapter boundary
            mapped = self._map_sdk_exception(exc)
            if mapped is None:
                # 非 SDK 异常通常是测试 Double 或业务代码缺陷，不应伪装成网络错误。
                raise
            raise mapped from exc
        turn = self.parse_response(response)
        self._remember_output_items(response, turn)
        return turn

    def build_request_payload(self, request: ModelRequest) -> dict[str, Any]:
        """把领域请求映射成 Responses API create 参数。"""

        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._map_messages(request.messages),
            "tools": [self._map_tool(tool) for tool in request.tools],
            # Runtime 当前顺序执行工具。即使 Provider 忽略这个提示并返回多调用，
            # Adapter 仍会解析全部调用，Agent 也会按原始顺序处理。
            "parallel_tool_calls": False,
            # 本阶段采用本地历史作为事实来源，不依赖 Provider 保存会话状态。
            "store": False,
        }
        if request.max_output_tokens is not None and request.max_output_tokens <= 0:
            raise ProviderProtocolError("ModelRequest.max_output_tokens must be positive")
        output_limit = (
            request.max_output_tokens
            if request.max_output_tokens is not None
            else self.max_output_tokens
        )
        if output_limit is not None:
            payload["max_output_tokens"] = output_limit
        return payload

    @classmethod
    def parse_response(cls, response: Any) -> ModelTurn:
        """遍历所有 Responses output items，构造 Provider 无关的 ModelTurn。"""

        status = cls._get(response, "status")
        if status == "failed":
            raise cls._map_failed_response(response)
        if status not in {"completed", "incomplete"}:
            raise ProviderProtocolError(f"unsupported OpenAI response status: {status!r}")

        output = cls._get(response, "output")
        if not isinstance(output, list):
            raise ProviderProtocolError("OpenAI response.output must be a list")

        visible_parts: list[str] = []
        calls: list[ToolCall] = []
        saw_refusal = False

        # output[0] 不保证是文本；必须保留服务端 item 顺序逐项解析。
        for item in output:
            item_type = cls._get(item, "type")
            if item_type == "reasoning":
                # 推理 item 不是用户可见文本。3C/Session 持久化阶段再保存 artifact。
                continue
            if item_type == "message":
                parts, refused = cls._parse_message_item(item)
                visible_parts.extend(parts)
                saw_refusal = saw_refusal or refused
                continue
            if item_type == "function_call":
                calls.append(cls._parse_function_call(item))
                continue
            raise ProviderProtocolError(
                f"unsupported OpenAI output item type: {item_type!r}"
            )

        content = "".join(visible_parts)
        if status == "incomplete":
            if calls:
                raise ProviderProtocolError(
                    "incomplete OpenAI response contained executable function calls"
                )
            finish_reason = "incomplete"
        elif saw_refusal:
            if calls:
                raise ProviderProtocolError(
                    "OpenAI response mixed refusal content with function calls"
                )
            finish_reason = "refusal"
        elif calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "completed"

        response_id = cls._get(response, "id")
        if response_id is not None and not isinstance(response_id, str):
            raise ProviderProtocolError("OpenAI response.id must be a string")

        return ModelTurn(
            content=content,
            tool_calls=tuple(calls),
            finish_reason=finish_reason,
            provider_response_id=response_id,
            usage=cls._parse_usage(cls._get(response, "usage")),
        )

    def _map_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role in {"user", "system"}:
                items.append(
                    {
                        "type": "message",
                        "role": message.role,
                        "content": message.content,
                    }
                )
                continue

            if message.role == "assistant":
                cache_key = tuple(call.call_id for call in message.tool_calls)
                cached_items = self._output_item_cache.get(cache_key)
                if cache_key and cached_items is not None:
                    # 原始 Provider item 已包含可见 message、reasoning 和函数调用，
                    # 不能再追加领域降级版本，否则上下文会重复。
                    items.extend(copy.deepcopy(cached_items))
                    continue
                if message.content:
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": message.content,
                        }
                    )
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": call.raw_arguments
                            if call.raw_arguments is not None
                            else json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
                continue

            if message.role == "tool":
                if not message.tool_call_id:
                    raise ProviderProtocolError(
                        "tool message is missing the required tool_call_id"
                    )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
                continue

            raise ProviderProtocolError(f"unsupported message role: {message.role!r}")
        return items

    def _remember_output_items(self, response: Any, turn: ModelTurn) -> None:
        if not turn.tool_calls:
            return
        output = self._get(response, "output")
        if not isinstance(output, list):
            return
        key = tuple(call.call_id for call in turn.tool_calls)
        self._output_item_cache[key] = [
            self._serialize_output_item(item) for item in output
        ]

    @staticmethod
    def _serialize_output_item(item: Any) -> dict[str, Any]:
        if isinstance(item, Mapping):
            return copy.deepcopy(dict(item))
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json", exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        raise ProviderProtocolError(
            f"cannot serialize OpenAI output item of type {type(item).__name__}"
        )

    @staticmethod
    def _map_tool(tool: ToolDefinition) -> dict[str, Any]:
        parameters = copy.deepcopy(tool.parameters)
        if parameters.get("type") != "object":
            raise ProviderProtocolError(
                f"tool {tool.name!r} parameters must have type=object"
            )
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            raise ProviderProtocolError(
                f"tool {tool.name!r} parameters must define properties"
            )
        if parameters.get("additionalProperties") is not False:
            raise ProviderProtocolError(
                f"tool {tool.name!r} must set additionalProperties=false"
            )
        required = parameters.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(name, str) for name in required
        ):
            raise ProviderProtocolError(
                f"tool {tool.name!r} required must be a string list"
            )
        if set(required) != set(properties):
            raise ProviderProtocolError(
                f"tool {tool.name!r} strict schema must require every property"
            )

        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
            "strict": True,
        }

    @classmethod
    def _parse_message_item(cls, item: Any) -> tuple[list[str], bool]:
        content = cls._get(item, "content")
        if not isinstance(content, list):
            raise ProviderProtocolError("OpenAI message item content must be a list")

        parts: list[str] = []
        refused = False
        for block in content:
            block_type = cls._get(block, "type")
            if block_type == "output_text":
                parts.append(cls._require_str(block, "text", "output_text.text"))
            elif block_type == "refusal":
                parts.append(cls._require_str(block, "refusal", "refusal.refusal"))
                refused = True
            else:
                raise ProviderProtocolError(
                    f"unsupported OpenAI message content type: {block_type!r}"
                )
        return parts, refused

    @classmethod
    def _parse_function_call(cls, item: Any) -> ToolCall:
        call_id = cls._require_str(item, "call_id", "function_call.call_id")
        name = cls._require_str(item, "name", "function_call.name")
        raw_arguments = cls._require_str(
            item,
            "arguments",
            "function_call.arguments",
            allow_empty=True,
        )

        argument_error: str | None = None
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            parsed = {}
            argument_error = f"{exc.msg} at line {exc.lineno} column {exc.colno}"
        if argument_error is None and not isinstance(parsed, dict):
            parsed = {}
            argument_error = "top-level JSON value must be an object"

        return ToolCall(
            call_id=call_id,
            name=name,
            arguments=parsed,
            raw_arguments=raw_arguments,
            argument_error=argument_error,
        )

    @classmethod
    def _parse_usage(cls, usage: Any) -> Usage | None:
        if usage is None:
            return None
        return Usage(
            input_tokens=cls._require_int(usage, "input_tokens"),
            output_tokens=cls._require_int(usage, "output_tokens"),
            total_tokens=cls._require_int(usage, "total_tokens"),
        )

    @classmethod
    def _map_failed_response(cls, response: Any) -> ProviderError:
        error = cls._get(response, "error")
        code = cls._get(error, "code") if error is not None else None
        message = cls._get(error, "message") if error is not None else None
        if not isinstance(message, str) or not message:
            message = "OpenAI returned a failed response without an error message"
        details = {"response_id": cls._get(response, "id"), "provider_code": code}
        if code == "server_error":
            return ProviderServerError(message, details=details)
        if code == "rate_limit_exceeded":
            return ProviderRateLimitedError(message, details=details)
        if code is not None:
            return ProviderBadRequestError(message, details=details)
        return ProviderProtocolError(message, details=details)

    @staticmethod
    def _create_default_client(timeout: float) -> Any:
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI provider requires the optional dependency: "
                "pip install -e .[openai]"
            ) from exc

        try:
            # Runtime 将来负责统一重试；先禁用 SDK 自动重试，防止双重退避。
            return OpenAI(max_retries=0, timeout=timeout)
        except OpenAIError as exc:
            raise ProviderAuthError(
                "OpenAI authentication is not configured; set OPENAI_API_KEY."
            ) from exc

    @staticmethod
    def _map_sdk_exception(exc: Exception) -> ProviderError | None:
        try:
            import openai
        except ImportError:
            return None

        if isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(str(exc) or "OpenAI request timed out")
        if isinstance(exc, openai.RateLimitError):
            return ProviderRateLimitedError(
                str(exc) or "OpenAI rate limit exceeded",
                details=OpenAIResponsesAdapter._status_details(exc),
            )
        if isinstance(
            exc,
            (openai.AuthenticationError, openai.PermissionDeniedError),
        ):
            return ProviderAuthError(
                str(exc) or "OpenAI authentication failed",
                details=OpenAIResponsesAdapter._status_details(exc),
            )
        if isinstance(exc, openai.APIConnectionError):
            return ProviderConnectionError(str(exc) or "OpenAI connection failed")
        if isinstance(exc, openai.APIStatusError):
            details = OpenAIResponsesAdapter._status_details(exc)
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int) and status_code >= 500:
                return ProviderServerError(str(exc), details=details)
            if status_code == 429:
                return ProviderRateLimitedError(str(exc), details=details)
            if status_code in {401, 403}:
                return ProviderAuthError(str(exc), details=details)
            return ProviderBadRequestError(str(exc), details=details)
        if isinstance(exc, openai.OpenAIError):
            return ProviderProtocolError(str(exc) or "unexpected OpenAI SDK error")
        return None

    @staticmethod
    def _status_details(exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {}
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            details["status_code"] = status_code
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after")
            if retry_after:
                details["retry_after"] = retry_after
        request_id = getattr(exc, "request_id", None)
        if request_id:
            details["request_id"] = request_id
        return details

    @staticmethod
    def _get(value: Any, name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _require_str(
        cls,
        value: Any,
        name: str,
        label: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        result = cls._get(value, name)
        if not isinstance(result, str) or (not result and not allow_empty):
            raise ProviderProtocolError(f"OpenAI {label} must be a string")
        return result

    @classmethod
    def _require_int(cls, value: Any, name: str) -> int:
        result = cls._get(value, name)
        if not isinstance(result, int):
            raise ProviderProtocolError(f"OpenAI usage.{name} must be an integer")
        return result


__all__ = ["OpenAIResponsesAdapter"]
