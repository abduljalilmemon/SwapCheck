"""LiteLLM-backed provider adapter for real model comparisons."""

from __future__ import annotations

import os
import time
from typing import Any

# Ensure LiteLLM uses local backup model cost map and doesn't hang fetching remote json
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"

from litellm import acompletion

from swapcheck.providers.base import BaseProvider, ChatMessage, LLMResponse, TokenUsage, ToolCall


class LiteLLMProvider(BaseProvider):
    """Thin adapter around LiteLLM for production provider execution."""

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        cost_per_input_token: float | None = None,
        cost_per_output_token: float | None = None,
        **extra_kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.extra_kwargs = extra_kwargs

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [message.to_dict() for message in messages],
            "temperature": self.temperature,
        }
        if self.api_key is not None:
            payload["api_key"] = self.api_key
        if self.base_url is not None:
            payload["base_url"] = self.base_url
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
        payload.update(self.extra_kwargs)
        payload.update(kwargs)

        start_time = time.perf_counter()
        result = await acompletion(**payload)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        choice = result.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        tool_calls: list[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", []) or []:
            function_payload = getattr(raw_call, "function", None)
            if isinstance(raw_call, dict):
                function_payload = raw_call.get("function", {})
            if function_payload is None:
                continue
            call_name = getattr(function_payload, "name", None)
            if isinstance(function_payload, dict):
                call_name = function_payload.get("name")
            arguments = getattr(function_payload, "arguments", None)
            if isinstance(function_payload, dict):
                arguments = function_payload.get("arguments")
            call_id = getattr(raw_call, "id", "tool_call")
            if isinstance(raw_call, dict):
                call_id = raw_call.get("id", "tool_call")
            if isinstance(arguments, str):
                tool_calls.append(
                    ToolCall.from_raw(
                        call_id=str(call_id),
                        name=str(call_name or "unknown_tool"),
                        raw_args=arguments,
                    )
                )
            else:
                tool_calls.append(
                    ToolCall(
                        id=str(call_id),
                        name=str(call_name or "unknown_tool"),
                        arguments=dict(arguments or {}),
                    )
                )

        usage = getattr(result, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

        if self.cost_per_input_token is not None or self.cost_per_output_token is not None:
            cost = (prompt_tokens * (self.cost_per_input_token or 0.0)) + (
                completion_tokens * (self.cost_per_output_token or 0.0)
            )
        else:
            raw_cost = (
                getattr(result, "_hidden_params", {}).get("response_cost", 0.0)
                if hasattr(result, "_hidden_params")
                else 0.0
            )
            cost = float(raw_cost or 0.0)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            latency_ms=latency_ms,
            cost=float(cost),
            model=self.model_name,
        )
