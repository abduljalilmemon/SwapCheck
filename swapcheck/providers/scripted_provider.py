"""Scripted mock provider with optional degradation controls for offline evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from swapcheck.providers.base import ChatMessage, LLMResponse
from swapcheck.providers.mock_provider import MockProvider


class ScriptedMockProvider(MockProvider):
    """MockProvider variant that supports optional deterministic degradation hints."""

    def __init__(
        self,
        model_name: str = "scripted-mock",
        temperature: float = 0.0,
        cost_per_input_token: float = 0.000001,
        cost_per_output_token: float = 0.000002,
        *,
        degrade_categories: Sequence[str] | None = None,
        degrade_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
        )
        self.degrade_categories = {str(item).lower() for item in (degrade_categories or [])}
        self.degrade_rate = float(degrade_rate)
        self.extra_kwargs = kwargs

    def maybe_degrade(self, messages: list[ChatMessage]) -> LLMResponse | None:
        if self.degrade_rate <= 0.0 or not self.degrade_categories:
            return None
        text = " ".join((msg.content or "") for msg in messages if msg.content).lower()
        if any(category in text for category in self.degrade_categories):
            degraded = self.default_fallback.model_copy(deep=True)
            degraded.content = "I need a bit more context before I can safely proceed."
            degraded.usage = degraded.usage.model_copy(
                update={"total_tokens": max(degraded.usage.total_tokens, 1)}
            )
            return degraded
        return None


ScriptedProvider = ScriptedMockProvider
__all__ = ["ScriptedMockProvider", "ScriptedProvider"]
