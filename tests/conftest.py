"""Shared pytest fixtures for SwapCheck tests."""

from collections.abc import Generator
from pathlib import Path

import pytest

from swapcheck.cache import ResponseCache
from swapcheck.providers.base import ChatMessage, LLMResponse, TokenUsage, ToolCall
from swapcheck.providers.mock_provider import MockProvider


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Generator[ResponseCache, None, None]:
    """Provide an isolated, temporary SQLite response cache."""
    cache_path = tmp_path / "test_cache.db"
    cache = ResponseCache(db_path=cache_path)
    yield cache
    cache.clear()


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """Provide a standard user conversation history."""
    return [
        ChatMessage(role="system", content="You are a helpful customer support agent."),
        ChatMessage(role="user", content="Hello, I need help with booking BKG-101."),
    ]


@pytest.fixture
def sample_response() -> LLMResponse:
    """Provide a sample LLM response with tool call."""
    return LLMResponse(
        content="Looking up your booking details now.",
        tool_calls=[
            ToolCall(
                id="call_test_123",
                name="lookup_booking",
                arguments={"booking_id": "BKG-101", "customer_email": "customer@example.com"},
            )
        ],
        usage=TokenUsage(prompt_tokens=40, completion_tokens=25, total_tokens=65),
        latency_ms=120.5,
        model="test-model",
    )


@pytest.fixture
def mock_provider() -> MockProvider:
    """Provide a clean MockProvider instance."""
    return MockProvider(model_name="mock-baseline")
