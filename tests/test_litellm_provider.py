"""Hermetic tests for LiteLLMProvider mocking acompletion with zero network."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from swapcheck.config import ModelConfig
from swapcheck.providers import build_provider
from swapcheck.providers.base import ChatMessage
from swapcheck.providers.litellm_provider import LiteLLMProvider


class MockChoice:
    def __init__(self, message: Any) -> None:
        self.message = message


class MockMessage:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class MockUsage:
    def __init__(self, prompt_tokens: int = 15, completion_tokens: int = 25) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockCompletionResult:
    def __init__(
        self,
        content: str | None = "Agent response",
        tool_calls: list[Any] | None = None,
        prompt_tokens: int = 15,
        completion_tokens: int = 25,
        response_cost: float = 0.00045,
    ) -> None:
        self.choices = [MockChoice(MockMessage(content=content, tool_calls=tool_calls))]
        self.usage = MockUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._hidden_params = {"response_cost": response_cost}


@pytest.mark.asyncio
async def test_litellm_provider_tool_call_parsing_and_latency() -> None:
    """LiteLLMProvider correctly parses tool calls and measures latency > 0."""
    fake_tool_call = {
        "id": "call_123",
        "function": {
            "name": "lookup_booking",
            "arguments": '{"booking_id": "BKG-999", "customer_email": "test@example.com"}',
        },
    }
    mock_result = MockCompletionResult(content="Checking booking", tool_calls=[fake_tool_call])

    provider = LiteLLMProvider(model_name="mock-gpt")
    with patch(
        "swapcheck.providers.litellm_provider.acompletion", new=AsyncMock(return_value=mock_result)
    ):
        response = await provider.complete([ChatMessage(role="user", content="Lookup BKG-999")])

    assert response.content == "Checking booking"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].name == "lookup_booking"
    assert response.tool_calls[0].arguments == {
        "booking_id": "BKG-999",
        "customer_email": "test@example.com",
    }
    assert response.usage.prompt_tokens == 15
    assert response.usage.completion_tokens == 25
    assert response.usage.total_tokens == 40
    assert response.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_litellm_provider_custom_cost_calculation() -> None:
    """LiteLLMProvider computes cost from cost_per_input_token and cost_per_output_token when set."""
    mock_result = MockCompletionResult(prompt_tokens=100, completion_tokens=50, response_cost=0.999)
    provider = LiteLLMProvider(
        model_name="mock-gpt",
        cost_per_input_token=0.00001,
        cost_per_output_token=0.00002,
    )

    with patch(
        "swapcheck.providers.litellm_provider.acompletion", new=AsyncMock(return_value=mock_result)
    ):
        response = await provider.complete([ChatMessage(role="user", content="Hi")])

    # 100 * 0.00001 = 0.001; 50 * 0.00002 = 0.001; total = 0.002
    assert pytest.approx(response.cost, rel=1e-5) == 0.002


@pytest.mark.asyncio
async def test_litellm_provider_fallback_response_cost() -> None:
    """LiteLLMProvider falls back to response_cost from LiteLLM when no manual cost override is set."""
    mock_result = MockCompletionResult(
        prompt_tokens=10, completion_tokens=10, response_cost=0.00035
    )
    provider = LiteLLMProvider(model_name="mock-gpt")

    with patch(
        "swapcheck.providers.litellm_provider.acompletion", new=AsyncMock(return_value=mock_result)
    ):
        response = await provider.complete([ChatMessage(role="user", content="Hi")])

    assert pytest.approx(response.cost, rel=1e-5) == 0.00035


def test_build_provider_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_provider raises ValueError if api_key_env_var is set but missing in environment."""
    monkeypatch.delenv("CUSTOM_MISSING_API_KEY", raising=False)
    config = ModelConfig(
        provider="litellm",
        name="test-litellm",
        model="gpt-test",
        api_key_env_var="CUSTOM_MISSING_API_KEY",
    )
    with pytest.raises(
        ValueError,
        match="API key environment variable 'CUSTOM_MISSING_API_KEY' for model 'gpt-test' is not set.",
    ):
        build_provider(config)


@pytest.mark.asyncio
async def test_internal_bookkeeping_kwargs_are_never_sent_to_the_model_api() -> None:
    """scenario_id / category / repetition_index are harness metadata, not API parameters."""
    provider = LiteLLMProvider(model_name="mock-gpt")
    mock_result = MockCompletionResult(content="ok")
    with patch(
        "swapcheck.providers.litellm_provider.acompletion",
        new=AsyncMock(return_value=mock_result),
    ) as mocked:
        await provider.complete(
            [ChatMessage(role="user", content="hi")],
            scenario_id="BKG-1",
            category="cancellation",
            repetition_index=2,
            top_p=0.9,
        )
    sent = mocked.await_args.kwargs
    assert "scenario_id" not in sent
    assert "category" not in sent
    assert "repetition_index" not in sent
    assert sent["top_p"] == 0.9
