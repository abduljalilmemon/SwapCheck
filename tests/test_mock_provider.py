"""Tests for MockProvider functionality and scriptability."""

import pytest

from swapcheck.providers.base import ChatMessage, LLMResponse, TokenUsage, ToolCall
from swapcheck.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_mock_provider_fallback(mock_provider: MockProvider) -> None:
    """Test that mock provider returns default fallback when unconfigured."""
    messages = [ChatMessage(role="user", content="Hello")]
    resp = await mock_provider.complete(messages)

    assert resp is not None
    assert resp.content == "Mock response: OK."
    assert resp.model == "mock-baseline"
    assert len(mock_provider.call_history) == 1
    assert mock_provider.call_history[0]["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_mock_provider_canned_queue(mock_provider: MockProvider) -> None:
    """Test FIFO ordering of canned responses."""
    mock_provider.enqueue_response(content="First response")
    mock_provider.enqueue_response(
        content="Second response with tool",
        tool_calls=[ToolCall(id="c1", name="test_tool", arguments={"x": 1})],
    )

    messages = [ChatMessage(role="user", content="Test")]
    r1 = await mock_provider.complete(messages)
    r2 = await mock_provider.complete(messages)
    r3 = await mock_provider.complete(messages)  # Should fall back to default

    assert r1.content == "First response"
    assert len(r1.tool_calls) == 0

    assert r2.content == "Second response with tool"
    assert len(r2.tool_calls) == 1
    assert r2.tool_calls[0].name == "test_tool"
    assert r2.tool_calls[0].arguments == {"x": 1}

    assert r3.content == "Mock response: OK."


@pytest.mark.asyncio
async def test_mock_provider_pattern_rules(mock_provider: MockProvider) -> None:
    """Test regex pattern matching rules."""
    mock_provider.add_rule(
        pattern=r"cancel.*booking",
        response=LLMResponse(
            content="Cancelling your reservation.",
            tool_calls=[
                ToolCall(id="c2", name="cancel_booking", arguments={"booking_id": "BKG-1"})
            ],
        ),
    )
    mock_provider.add_rule(
        pattern=r"status\s+(\d+)",
        response=lambda msgs: LLMResponse(
            content=f"Status checked for {msgs[-1].content}",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
    )

    r_cancel = await mock_provider.complete(
        [ChatMessage(role="user", content="I need to cancel my booking please")]
    )
    assert r_cancel.content == "Cancelling your reservation."
    assert len(r_cancel.tool_calls) == 1
    assert r_cancel.tool_calls[0].name == "cancel_booking"

    r_status = await mock_provider.complete(
        [ChatMessage(role="user", content="What is status 999?")]
    )
    assert r_status.content == "Status checked for What is status 999?"


@pytest.mark.asyncio
async def test_mock_provider_cost_calculation(mock_provider: MockProvider) -> None:
    """Test that cost is computed based on tokens and configured rates."""
    mock_provider.cost_per_input_token = 0.001
    mock_provider.cost_per_output_token = 0.002
    mock_provider.enqueue_response(
        content="Testing cost",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )

    resp = await mock_provider.complete([ChatMessage(role="user", content="Hi")])
    # 100 * 0.001 + 50 * 0.002 = 0.10 + 0.10 = 0.20
    assert pytest.approx(resp.cost, rel=1e-5) == 0.20


def test_tool_call_from_raw() -> None:
    """Test ToolCall.from_raw parsing JSON strings and dictionaries."""
    tc1 = ToolCall.from_raw("1", "test", '{"key": "value", "num": 42}')
    assert tc1.arguments == {"key": "value", "num": 42}

    tc2 = ToolCall.from_raw("2", "test", {"already": "dict"})
    assert tc2.arguments == {"already": "dict"}

    tc3 = ToolCall.from_raw("3", "test", "not-json-string")
    assert tc3.arguments == {"__raw_string__": "not-json-string"}


@pytest.mark.asyncio
async def test_mock_provider_reset(mock_provider: MockProvider) -> None:
    """Test that reset clears queue, rules, and history."""
    mock_provider.enqueue_response(content="Queued")
    mock_provider.add_rule(r"test", LLMResponse(content="Rule"))
    await mock_provider.complete([ChatMessage(role="user", content="Hello")])

    assert len(mock_provider.call_history) == 1
    assert len(mock_provider.canned_responses) == 0

    mock_provider.reset()
    assert len(mock_provider.call_history) == 0
    assert len(mock_provider.canned_responses) == 0
    assert len(mock_provider.rules) == 0
