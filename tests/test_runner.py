"""Tests for scenario execution runner and budget/cache behavior."""

from __future__ import annotations

import pytest

from swapcheck.providers.base import TokenUsage, ToolCall
from swapcheck.providers.mock_provider import MockProvider
from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.runner import ScenarioRunner


@pytest.mark.asyncio
async def test_runner_repeats_scenarios_and_caches_calls() -> None:
    """The runner should repeat scenario executions and avoid re-sending identical cache hits."""
    scenario = Scenario(
        id="runner-lookup-1",
        category="simple_lookup",
        description="Look up a booking.",
        initial_db_state=[
            {
                "id": "BKG-1001",
                "customer_name": "Alice Smith",
                "customer_email": "alice@example.com",
                "item_type": "flight",
                "details": "SFO to JFK",
                "purchase_date": "2026-01-10T12:00:00Z",
                "departure_date": "2026-01-20T08:00:00Z",
                "price": 240.0,
                "status": "CONFIRMED",
                "refunded_amount": 0.0,
                "notes": [],
                "escalated": False,
                "escalation_reason": None,
            }
        ],
        user_turns=["Please look up BKG-1001 with alice@example.com."],
        expected_final_db_state=[{"id": "BKG-1001", "status": "CONFIRMED", "refunded_amount": 0.0}],
        forbidden_tool_calls=["issue_refund"],
        expected_tool_calls=["lookup_booking"],
        tags=["lookup"],
    )

    provider = MockProvider(model_name="mock-runner")
    provider.enqueue_response(
        content=None,
        tool_calls=[
            ToolCall(
                id="lookup_1",
                name="lookup_booking",
                arguments={"booking_id": "BKG-1001", "customer_email": "alice@example.com"},
            )
        ],
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    provider.enqueue_response(
        content="I found your booking and it is confirmed.",
        usage=TokenUsage(prompt_tokens=80, completion_tokens=30, total_tokens=110),
    )

    runner = ScenarioRunner(
        scenarios=[scenario],
        repetitions=2,
        max_concurrency=1,
        provider_factory=lambda: provider,
    )

    trajectories = await runner.run_all()
    assert len(trajectories) == 2
    assert trajectories[0].scenario_id == "runner-lookup-1"
    assert trajectories[0].turns[0].user_turn == "Please look up BKG-1001 with alice@example.com."
    assert trajectories[0].final_db_snapshot["BKG-1001"]["status"] == "CONFIRMED"
    # Repetitions are independent samples, so each repetition reaches the provider
    # (the cache key includes the repetition index): 2 repetitions -> 3 provider calls here,
    # because the two queued responses are consumed by repetition 0 and repetition 1 falls
    # back to the default response.
    calls_after_first_run = len(provider.call_history)
    assert calls_after_first_run == 3

    # Re-running the same suite with the same cache must not hit the provider again.
    trajectories_again = await runner.run_all()
    assert len(trajectories_again) == 2
    assert len(provider.call_history) == calls_after_first_run
    assert [t.turns[0].assistant_text for t in trajectories_again] == [
        t.turns[0].assistant_text for t in trajectories
    ]


@pytest.mark.asyncio
async def test_runner_stops_when_cost_budget_is_exceeded() -> None:
    """When the max_cost budget is hit, the runner should stop without crashing."""
    scenario = Scenario(
        id="runner-budget-1",
        category="cancellation",
        description="Try to cancel a booking.",
        initial_db_state=[
            {
                "id": "BKG-202",
                "customer_name": "Bob Jones",
                "customer_email": "bob@example.com",
                "item_type": "hotel",
                "details": "Hotel in NYC",
                "purchase_date": "2026-02-01T12:00:00Z",
                "departure_date": "2026-03-01T10:00:00Z",
                "price": 300.0,
                "status": "CONFIRMED",
                "refunded_amount": 0.0,
                "notes": [],
                "escalated": False,
                "escalation_reason": None,
            }
        ],
        user_turns=["Cancel booking BKG-202 for bob@example.com."],
        expected_final_db_state=[{"id": "BKG-202", "status": "CANCELLED", "refunded_amount": 0.0}],
        forbidden_tool_calls=["issue_refund"],
        expected_tool_calls=["lookup_booking", "cancel_booking"],
        tags=["cancellation"],
    )

    provider = MockProvider(model_name="mock-budget")
    provider.cost_per_input_token = 0.001
    provider.cost_per_output_token = 0.002
    provider.enqueue_response(
        content=None,
        tool_calls=[
            ToolCall(
                id="lookup_2",
                name="lookup_booking",
                arguments={"booking_id": "BKG-202", "customer_email": "bob@example.com"},
            )
        ],
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    provider.enqueue_response(
        content=None,
        tool_calls=[
            ToolCall(
                id="cancel_2",
                name="cancel_booking",
                arguments={"booking_id": "BKG-202", "reason": "Requested by customer"},
            )
        ],
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    provider.enqueue_response(content="Your booking has been cancelled.")

    runner = ScenarioRunner(
        scenarios=[scenario],
        repetitions=2,
        max_concurrency=1,
        provider_factory=lambda: provider,
        max_cost=0.0003,
    )

    result = await runner.run_all()
    assert runner.budget_exceeded is True
    assert len(result) <= 1
    assert result[0].budget_exceeded is True
