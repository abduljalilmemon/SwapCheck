"""Scriptable offline MockProvider for deterministic testing and CI."""

from __future__ import annotations

import re
from collections.abc import Callable
from re import Pattern
from typing import Any

from swapcheck.providers.base import (
    BaseProvider,
    ChatMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
)


class MockRule:
    """Pattern matching rule to produce a canned response."""

    def __init__(
        self,
        pattern: str | Pattern[str],
        response: LLMResponse | Callable[[list[ChatMessage]], LLMResponse],
    ) -> None:
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.response = response

    def matches(self, messages: list[ChatMessage]) -> bool:
        """Check if any message content matches the pattern."""
        return any(m.content and self.pattern.search(m.content) for m in reversed(messages))

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        """Produce the response for this rule."""
        if callable(self.response):
            return self.response(messages)
        return self.response.model_copy(deep=True)


class MockProvider(BaseProvider):
    """Offline, deterministic mock LLM provider for hermetic testing.

    Supports:
    1. Sequential canned responses (FIFO queue).
    2. Regex pattern-matching rules.
    3. Trajectory / call history inspection.
    4. Deterministic token counting and latency simulation.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        temperature: float = 0.0,
        cost_per_input_token: float = 0.000001,
        cost_per_output_token: float = 0.000002,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature)
        self.canned_responses: list[LLMResponse] = []
        self.rules: list[MockRule] = []
        self.call_history: list[dict[str, Any]] = []
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.default_fallback = LLMResponse(
            content="Mock response: OK.",
            model=model_name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def enqueue_response(
        self,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        usage: TokenUsage | None = None,
        latency_ms: float = 10.0,
    ) -> None:
        """Add a response to the FIFO queue."""
        resp = LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            usage=usage or TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            latency_ms=latency_ms,
            model=self.model_name,
        )
        self.canned_responses.append(resp)

    def add_rule(
        self,
        pattern: str | Pattern[str],
        response: LLMResponse | Callable[[list[ChatMessage]], LLMResponse],
    ) -> None:
        """Add a regex pattern-matching rule."""
        self.rules.append(MockRule(pattern=pattern, response=response))

    def reset(self) -> None:
        """Clear call history, queue, and rules."""
        self.canned_responses.clear()
        self.rules.clear()
        self.call_history.clear()

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate deterministic response based on queue, rules, or fallback."""
        self.call_history.append(
            {
                "messages": [m.to_dict() for m in messages],
                "tools": tools,
                "kwargs": kwargs,
            }
        )

        resp: LLMResponse
        # 1. Check FIFO queue first
        if self.canned_responses:
            resp = self.canned_responses.pop(0)
        else:
            # 2. Check pattern matching rules
            matched_rule = next((r for r in self.rules if r.matches(messages)), None)
            if matched_rule:
                resp = matched_rule.generate(messages)
            else:
                # 3. Default fallback
                resp = self.default_fallback.model_copy(deep=True)

        resp.model = self.model_name
        if resp.cost == 0.0 and resp.usage.total_tokens > 0:
            resp.cost = (
                resp.usage.prompt_tokens * self.cost_per_input_token
                + resp.usage.completion_tokens * self.cost_per_output_token
            )
        return resp
