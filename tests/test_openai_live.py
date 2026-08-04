from __future__ import annotations

import os

import pytest

from mini_agent.providers.openai_responses import OpenAIResponsesAdapter
from mini_agent.schema import Message, ModelRequest


pytestmark = pytest.mark.live


def test_openai_live_text_response() -> None:
    """最小只读 smoke test；默认测试套件绝不会访问网络。"""

    if os.environ.get("RUN_OPENAI_LIVE_TEST") != "1":
        pytest.skip("set RUN_OPENAI_LIVE_TEST=1 to enable the live Provider test")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        pytest.skip("OPENAI_MODEL must explicitly select the live test model")

    turn = OpenAIResponsesAdapter(model, max_output_tokens=32).generate(
        ModelRequest(
            messages=[Message("user", "Reply with exactly: OK")],
            tools=[],
        )
    )

    assert turn.finish_reason == "completed"
    assert turn.content.strip() == "OK"
