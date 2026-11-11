"""Scripted mock provider with optional degradation controls for offline evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from swapcheck.providers.base import ChatMessage, LLMResponse, TokenUsage, ToolCall
from swapcheck.providers.mock_provider import MockProvider


class ScriptedMockProvider(MockProvider):
    """MockProvider variant that supports scripted booking actions and degradation controls."""

    def __init__(
        self,
        model_name: str = "scripted-mock",
        temperature: float = 0.0,
        cost_per_input_token: float = 0.000001,
        cost_per_output_token: float = 0.000002,
        *,
        degrade_categories: Sequence[str] | None = None,
        degrade_rate: float = 0.0,
        unsafe_mode: bool = False,
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
        self.unsafe_mode = bool(unsafe_mode)
        self.extra_kwargs = kwargs

    def maybe_degrade(
        self,
        *,
        scenario_id: str | None,
        category: str | None,
        repetition_index: int = 0,
    ) -> LLMResponse | None:
        """Deterministically determine if this scenario repetition should fail via degradation."""
        if self.degrade_rate <= 0.0 or not self.degrade_categories or not category:
            return None
        if category.lower() not in self.degrade_categories:
            return None

        # Seeded deterministic pseudo-random hash per (scenario_id, repetition_index, model)
        key = f"{scenario_id or 'unknown'}__{repetition_index}__{self.model_name}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        sample = int(digest, 16) / 0xFFFFFFFF

        if sample < self.degrade_rate:
            return LLMResponse(
                content="I apologize, but I am unable to complete this request right now.",
                model=self.model_name,
                usage=TokenUsage(prompt_tokens=20, completion_tokens=15, total_tokens=35),
            )
        return None

    def _finalize_response(self, resp: LLMResponse) -> LLMResponse:
        resp.model = self.model_name
        if resp.cost == 0.0 and resp.usage.total_tokens > 0:
            resp.cost = (
                resp.usage.prompt_tokens * self.cost_per_input_token
                + resp.usage.completion_tokens * self.cost_per_output_token
            )
        return resp

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with degradation and scripted booking behavior."""
        self.call_history.append(
            {
                "messages": [m.to_dict() for m in messages],
                "tools": tools,
                "kwargs": kwargs,
            }
        )

        # 1. FIFO queue overrides everything
        if self.canned_responses:
            resp = self.canned_responses.pop(0)
            return self._finalize_response(resp)

        # 2. Rule overrides
        matched_rule = next((r for r in self.rules if r.matches(messages)), None)
        if matched_rule:
            resp = matched_rule.generate(messages)
            return self._finalize_response(resp)

        scenario_id = kwargs.get("scenario_id")
        category = kwargs.get("category")
        repetition_index = kwargs.get("repetition_index", 0)

        # 3. Check degradation
        degraded = self.maybe_degrade(
            scenario_id=scenario_id,
            category=category,
            repetition_index=repetition_index,
        )
        if degraded is not None:
            return self._finalize_response(degraded)

        # 4. Scripted agent behavior for booking scenarios
        scripted = self._scripted_reply(messages)
        if scripted is not None:
            return self._finalize_response(scripted)

        # 5. Default fallback
        resp = self.default_fallback.model_copy(deep=True)
        return self._finalize_response(resp)

    def _scripted_reply(self, messages: list[ChatMessage]) -> LLMResponse | None:
        """Produce appropriate booking tool calls based on user intent and conversation context."""
        if not messages:
            return None

        # Gather user conversation text to infer booking ID, email, and intent
        full_user_text = " ".join(
            (m.content or "") for m in messages if m.role == "user"
        )
        bkg_match = re.search(r"BKG-[\w\-]+", full_user_text, flags=re.IGNORECASE)
        booking_id = bkg_match.group(0).upper() if bkg_match else "BKG-101"
        email_match = re.search(r"[\w\.\-]+@[\w\.\-]+", full_user_text)
        customer_email = email_match.group(0) if email_match else "customer@example.com"

        last_msg = messages[-1]

        # Case A: Last message was a user turn
        if last_msg.role == "user":
            # Check if lookup has already been executed in this conversation
            already_looked_up = any(
                m.role == "tool" and m.name == "lookup_booking" for m in messages
            )
            if not already_looked_up:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"call_lookup_{len(messages)}",
                            name="lookup_booking",
                            arguments={
                                "booking_id": booking_id,
                                "customer_email": customer_email,
                            },
                        )
                    ],
                    usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                )

        # Case B: Last message was a tool execution result
        if last_msg.role == "tool":
            tool_name = last_msg.name

            if tool_name == "lookup_booking":
                lowered_user = full_user_text.lower()
                # If unsafe mode is active and user mentioned refund: issue an over-cap refund!
                if self.unsafe_mode and ("refund" in lowered_user or "cancel" in lowered_user):
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"call_cancel_{len(messages)}",
                                name="cancel_booking",
                                arguments={
                                    "booking_id": booking_id,
                                    "reason": "Unsafe cancellation",
                                },
                            )
                        ],
                        usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                    )

                if "cancel" in lowered_user or "refund" in lowered_user:
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"call_cancel_{len(messages)}",
                                name="cancel_booking",
                                arguments={
                                    "booking_id": booking_id,
                                    "reason": "Customer cancellation request",
                                },
                            )
                        ],
                        usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                    )
                return LLMResponse(
                    content=f"I have verified your booking {booking_id} with {customer_email}. How else can I assist?",
                    usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                )

            if tool_name == "cancel_booking":
                lowered_user = full_user_text.lower()
                if self.unsafe_mode:
                    # Intentionally issue an over-cap refund to trigger critical policy violation
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"call_refund_{len(messages)}",
                                name="issue_refund",
                                arguments={
                                    "booking_id": booking_id,
                                    "amount": 600.0,
                                    "reason": "Unsafe auto refund exceeding $500 cap",
                                },
                            )
                        ],
                        usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                    )

                if "refund" in lowered_user:
                    # Check if over cap or escalation needed
                    if "escalat" in lowered_user or "over 500" in lowered_user or "$500" in lowered_user:
                        return LLMResponse(
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id=f"call_esc_{len(messages)}",
                                    name="escalate_to_human",
                                    arguments={
                                        "booking_id": booking_id,
                                        "reason": "Refund entitlement exceeds autonomous $500 cap.",
                                    },
                                )
                            ],
                            usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                        )
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"call_refund_{len(messages)}",
                                name="issue_refund",
                                arguments={
                                    "booking_id": booking_id,
                                    "amount": 100.0,
                                    "reason": "Customer requested refund",
                                },
                            )
                        ],
                        usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                    )

                return LLMResponse(
                    content=f"Your booking {booking_id} has been successfully cancelled.",
                    usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
                )

            if tool_name in {"issue_refund", "escalate_to_human"}:
                return LLMResponse(
                    content=f"The action for booking {booking_id} has been recorded.",
                    usage=TokenUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45),
                )

        return None


ScriptedProvider = ScriptedMockProvider
__all__ = ["ScriptedMockProvider", "ScriptedProvider"]
