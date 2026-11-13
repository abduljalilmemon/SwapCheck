"""LLM Provider abstractions and implementations."""

import os

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
            unsafe_mode=bool(config.extra_params.get("unsafe_mode", False)),
        )
    if provider_name in {"litellm", "openai", "azure"}:
        api_key: str | None = None
        if config.api_key_env_var:
            api_key = os.environ.get(config.api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"API key environment variable '{config.api_key_env_var}' for model '{config.model}' is not set."
                )

        extra_kwargs = {
            k: v for k, v in config.extra_params.items() if k not in {"api_key", "base_url"}
        }
        return LiteLLMProvider(
            model_name=config.model,
            temperature=config.temperature,
            api_key=api_key,
            base_url=config.extra_params.get("base_url"),
            max_tokens=config.max_tokens,
            cost_per_input_token=config.cost_per_input_token,
            cost_per_output_token=config.cost_per_output_token,
            **extra_kwargs,
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
