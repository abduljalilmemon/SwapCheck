"""LLM Provider abstractions and implementations."""

from swapcheck.config import ModelConfig
from swapcheck.providers.base import (
    BaseProvider,
    ChatMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from swapcheck.providers.litellm_provider import LiteLLMProvider
from swapcheck.providers.mock_provider import MockProvider, MockRule
from swapcheck.providers.scripted_provider import ScriptedMockProvider


def build_provider(config: ModelConfig) -> BaseProvider:
    """Construct a provider from the configured backend name."""
    provider_name = (config.provider or "mock").lower()
    if provider_name in {"mock", "mock_provider"}:
        return ScriptedMockProvider(
            model_name=config.model,
            temperature=config.temperature,
            cost_per_input_token=config.cost_per_input_token or 0.0,
            cost_per_output_token=config.cost_per_output_token or 0.0,
            degrade_categories=config.extra_params.get("degrade_categories", []),
            degrade_rate=float(config.extra_params.get("degrade_rate", 0.0)),
        )
    if provider_name in {"litellm", "openai", "azure"}:
        return LiteLLMProvider(
            model_name=config.model,
            temperature=config.temperature,
            api_key=config.extra_params.get("api_key") or None,
            base_url=config.extra_params.get("base_url"),
            max_tokens=config.max_tokens,
            **{k: v for k, v in config.extra_params.items() if k not in {"api_key", "base_url"}},
        )
    raise ValueError(f"Unsupported provider backend: {config.provider!r}")


__all__ = [
    "BaseProvider",
    "ChatMessage",
    "LLMResponse",
    "LiteLLMProvider",
    "MockProvider",
    "MockRule",
    "ScriptedMockProvider",
    "TokenUsage",
    "ToolCall",
    "build_provider",
]
