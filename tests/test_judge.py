"""Tests for the LLM judge scorer."""

from __future__ import annotations

import pytest

from swapcheck.providers.mock_provider import MockProvider
from swapcheck.scenarios.trajectory import Trajectory, TurnRecord
from swapcheck.scoring.judge import JudgeScorer


@pytest.mark.asyncio
async def test_judge_scorer_parses_structured_json() -> None:
    """The judge scorer should parse structured JSON and score a trajectory."""
    provider = MockProvider(model_name="judge-mock")
    provider.enqueue_response(
        content=(
            '{"helpfulness": 5, "groundedness": 4, "tone": 5, '
            '"off_policy_promises": false, "passed": true, '
            '"explanation": "The agent acted helpfully and stayed within policy."}'
        )
    )

    trajectory = Trajectory(
        scenario_id="SCN-1",
        category="simple_lookup",
        repetition_index=0,
        turns=[
            TurnRecord(
                turn_index=0,
                user_turn="Hi, I want to cancel a booking.",
                assistant_text="I need to verify your booking and request your email.",
                tool_calls=[],
                tool_results=[],
                latency_ms=10.0,
                tokens=20,
                cost=0.0,
            )
        ],
        total_latency_ms=10.0,
        total_tokens=20,
        total_cost=0.0,
        final_db_snapshot={},
    )

    score = await JudgeScorer(provider=provider).score(trajectory)
    assert score.helpfulness == 5
    assert score.groundedness == 4
    assert score.tone == 5
    assert score.off_policy_promises is False
    assert score.passed is True
    assert score.model == "judge-mock"


@pytest.mark.asyncio
async def test_judge_scorer_falls_back_on_invalid_json() -> None:
    """The judge scorer should gracefully handle broken JSON output."""
    provider = MockProvider(model_name="judge-mock")
    provider.enqueue_response(content="Not valid JSON at all")

    trajectory = Trajectory(
        scenario_id="SCN-2",
        category="refund_policy_edge_cases",
        repetition_index=0,
        turns=[
            TurnRecord(
                turn_index=0,
                user_turn="I need a refund.",
                assistant_text="I can look up the booking and verify the refund policy.",
                latency_ms=5.0,
                tokens=15,
            )
        ],
        total_latency_ms=5.0,
        total_tokens=15,
        total_cost=0.0,
        final_db_snapshot={},
    )

    score = await JudgeScorer(provider=provider).score(trajectory)
    assert 1 <= score.helpfulness <= 5
    assert 1 <= score.groundedness <= 5
    assert 1 <= score.tone <= 5
    assert score.model == "judge-mock"
