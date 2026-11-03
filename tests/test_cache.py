"""Tests for persistent SQLite response cache."""

from swapcheck.cache import ResponseCache
from swapcheck.providers.base import ChatMessage, LLMResponse


def test_cache_miss_returns_none(tmp_cache: ResponseCache) -> None:
    """Test that querying an unknown key returns None."""
    assert tmp_cache.get("non_existent_key") is None


def test_cache_set_and_get(
    tmp_cache: ResponseCache,
    sample_messages: list[ChatMessage],
    sample_response: LLMResponse,
) -> None:
    """Test storing and retrieving a response."""
    key = tmp_cache.compute_key(
        model="test-model",
        messages=sample_messages,
        temperature=0.0,
    )

    tmp_cache.set(cache_key=key, model="test-model", response=sample_response)

    cached_resp = tmp_cache.get(key)
    assert cached_resp is not None
    assert cached_resp.cached is True
    assert cached_resp.content == sample_response.content
    assert cached_resp.model == sample_response.model
    assert len(cached_resp.tool_calls) == len(sample_response.tool_calls)
    assert cached_resp.tool_calls[0].name == "lookup_booking"
    assert cached_resp.tool_calls[0].arguments == {
        "booking_id": "BKG-101",
        "customer_email": "customer@example.com",
    }
    assert cached_resp.usage.total_tokens == sample_response.usage.total_tokens


def test_cache_hit_counter_and_stats(
    tmp_cache: ResponseCache,
    sample_messages: list[ChatMessage],
    sample_response: LLMResponse,
) -> None:
    """Test that get operations increment hit counts and stats reflect them."""
    key = tmp_cache.compute_key(model="test-model", messages=sample_messages)
    tmp_cache.set(cache_key=key, model="test-model", response=sample_response)

    stats_init = tmp_cache.stats()
    assert stats_init["entries"] == 1
    assert stats_init["hits"] == 0

    tmp_cache.get(key)
    tmp_cache.get(key)
    tmp_cache.get(key)

    stats_after = tmp_cache.stats()
    assert stats_after["entries"] == 1
    assert stats_after["hits"] == 3


def test_deterministic_key_computation() -> None:
    """Test that key computation is stable and invariant to dictionary ordering."""
    m1 = [
        {"role": "user", "content": "Hello", "extra": {"b": 2, "a": 1}},
    ]
    m2 = [
        {"role": "user", "content": "Hello", "extra": {"a": 1, "b": 2}},
    ]
    k1 = ResponseCache.compute_key(model="gpt", messages=m1, temperature=0.5)
    k2 = ResponseCache.compute_key(model="gpt", messages=m2, temperature=0.5)
    assert k1 == k2

    # Different model yields different key
    k3 = ResponseCache.compute_key(model="other-gpt", messages=m1, temperature=0.5)
    assert k1 != k3


def test_cache_clear(
    tmp_cache: ResponseCache,
    sample_messages: list[ChatMessage],
    sample_response: LLMResponse,
) -> None:
    """Test that clear wipes all entries."""
    key = tmp_cache.compute_key(model="test-model", messages=sample_messages)
    tmp_cache.set(cache_key=key, model="test-model", response=sample_response)
    assert tmp_cache.stats()["entries"] == 1

    tmp_cache.clear()
    assert tmp_cache.stats()["entries"] == 0
    assert tmp_cache.get(key) is None
