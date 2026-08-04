from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from mini_agent.provider_errors import (
    ProviderBadRequestError,
    ProviderProtocolError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from mini_agent.providers.openai_responses import OpenAIResponsesAdapter
from mini_agent.schema import Message, ModelRequest, ToolCall, ToolDefinition
from mini_agent.tools import create_default_registry


class FakeResponses:
    def __init__(self, *, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.payloads: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def test_adapter_maps_stateless_history_tools_and_usage() -> None:
    fake_responses = FakeResponses(
        response={
            "id": "resp-1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
            },
        }
    )
    adapter = OpenAIResponsesAdapter(
        "test-model",
        client=FakeClient(fake_responses),
        max_output_tokens=256,
    )
    call = ToolCall(
        "call-read",
        "read_file",
        {"path": "README.md"},
        raw_arguments='{"path":"README.md"}',
    )
    request = ModelRequest(
        messages=[
            Message("system", "Follow repository rules."),
            Message("user", "Read the file."),
            Message("assistant", "I will inspect it.", tool_calls=(call,)),
            Message(
                "tool",
                "file contents",
                tool_call_id="call-read",
                tool_name="read_file",
            ),
        ],
        tools=[tool_definition()],
    )

    turn = adapter.generate(request)

    assert turn.content == "done"
    assert turn.finish_reason == "completed"
    assert turn.provider_response_id == "resp-1"
    assert turn.usage is not None and turn.usage.total_tokens == 14
    payload = fake_responses.payloads[0]
    assert payload["model"] == "test-model"
    assert payload["parallel_tool_calls"] is False
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 256
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file.",
            "parameters": tool_definition().parameters,
            "strict": True,
        }
    ]
    assert payload["input"][-2:] == [
        {
            "type": "function_call",
            "call_id": "call-read",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-read",
            "output": "file contents",
        },
    ]


def test_adapter_parses_all_output_items_and_invalid_arguments(tmp_path: Path) -> None:
    response = {
        "id": "resp-tools",
        "status": "completed",
        "output": [
            {"type": "reasoning"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Inspecting."}],
            },
            {
                "type": "function_call",
                "call_id": "call-good",
                "name": "list_files",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call-bad",
                "name": "read_file",
                "arguments": '{"path":',
            },
        ],
        "usage": None,
    }

    turn = OpenAIResponsesAdapter.parse_response(response)

    assert turn.content == "Inspecting."
    assert turn.finish_reason == "tool_calls"
    assert [call.call_id for call in turn.tool_calls] == ["call-good", "call-bad"]
    assert turn.tool_calls[0].arguments == {}
    assert turn.tool_calls[1].raw_arguments == '{"path":'
    assert turn.tool_calls[1].argument_error is not None

    result = create_default_registry(tmp_path).run(turn.tool_calls[1])
    assert result.call_id == "call-bad"
    assert result.error_code == "invalid_arguments"


def test_adapter_parses_official_sdk_output_objects() -> None:
    response = SimpleNamespace(
        id="resp-sdk",
        status="completed",
        output=[
            ResponseOutputMessage(
                id="msg-1",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="SDK text",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
            ResponseFunctionToolCall(
                arguments="{}",
                call_id="call-sdk",
                name="list_files",
                type="function_call",
            ),
        ],
        usage=None,
    )

    turn = OpenAIResponsesAdapter.parse_response(response)

    assert turn.content == "SDK text"
    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0].call_id == "call-sdk"


def test_adapter_replays_cached_reasoning_items_before_tool_output() -> None:
    reasoning_item = {"type": "reasoning", "id": "reason-1", "summary": []}
    function_item = {
        "type": "function_call",
        "call_id": "call-list",
        "name": "list_files",
        "arguments": "{}",
    }
    fake_responses = FakeResponses(
        response={
            "id": "resp-tools",
            "status": "completed",
            "output": [reasoning_item, function_item],
            "usage": None,
        }
    )
    adapter = OpenAIResponsesAdapter(
        "test-model",
        client=FakeClient(fake_responses),
    )
    first = adapter.generate(
        ModelRequest(messages=[Message("user", "List files")], tools=[])
    )

    fake_responses.response = {
        "id": "resp-final",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
        "usage": None,
    }
    adapter.generate(
        ModelRequest(
            messages=[
                Message("user", "List files"),
                Message("assistant", first.content, tool_calls=first.tool_calls),
                Message(
                    "tool",
                    "README.md",
                    tool_call_id="call-list",
                    tool_name="list_files",
                ),
            ],
            tools=[],
        )
    )

    assert fake_responses.payloads[1]["input"] == [
        {"type": "message", "role": "user", "content": "List files"},
        reasoning_item,
        function_item,
        {
            "type": "function_call_output",
            "call_id": "call-list",
            "output": "README.md",
        },
    ]


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            {
                "id": "resp-refusal",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "cannot comply"}],
                    }
                ],
                "usage": None,
            },
            "refusal",
        ),
        (
            {
                "id": "resp-incomplete",
                "status": "incomplete",
                "output": [],
                "usage": None,
            },
            "incomplete",
        ),
    ],
)
def test_adapter_preserves_non_completed_finish_reasons(
    response: dict[str, Any],
    reason: str,
) -> None:
    assert OpenAIResponsesAdapter.parse_response(response).finish_reason == reason


def test_adapter_fails_closed_on_unknown_output_item() -> None:
    with pytest.raises(ProviderProtocolError, match="unsupported OpenAI output"):
        OpenAIResponsesAdapter.parse_response(
            {
                "id": "resp-unknown",
                "status": "completed",
                "output": [{"type": "future_item"}],
                "usage": None,
            }
        )


def test_adapter_rejects_non_strict_tool_schema() -> None:
    adapter = OpenAIResponsesAdapter(
        "test-model",
        client=FakeClient(FakeResponses(response=None)),
    )
    request = ModelRequest(
        messages=[Message("user", "test")],
        tools=[
            ToolDefinition(
                name="unsafe_tool",
                description="Missing strict boundary.",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )

    with pytest.raises(ProviderProtocolError, match="additionalProperties"):
        adapter.build_request_payload(request)


def test_failed_response_maps_provider_error() -> None:
    with pytest.raises(ProviderBadRequestError) as captured:
        OpenAIResponsesAdapter.parse_response(
            {
                "id": "resp-failed",
                "status": "failed",
                "error": {"code": "invalid_prompt", "message": "bad prompt"},
                "output": [],
            }
        )

    assert captured.value.code == "provider_bad_request"
    assert captured.value.details["response_id"] == "resp-failed"


def test_sdk_timeout_and_rate_limit_are_classified() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    timeout_adapter = OpenAIResponsesAdapter(
        "test-model",
        client=FakeClient(
            FakeResponses(error=openai.APITimeoutError(request=request))
        ),
    )

    with pytest.raises(ProviderTimeoutError) as timeout:
        timeout_adapter.generate(ModelRequest(messages=[], tools=[]))
    assert timeout.value.retryable is True

    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": "2"},
    )
    rate_adapter = OpenAIResponsesAdapter(
        "test-model",
        client=FakeClient(
            FakeResponses(
                error=openai.RateLimitError(
                    "slow down",
                    response=response,
                    body=None,
                )
            )
        ),
    )

    with pytest.raises(ProviderRateLimitedError) as rate_limited:
        rate_adapter.generate(ModelRequest(messages=[], tools=[]))
    assert rate_limited.value.details["retry_after"] == "2"
