"""Tests for the LLM judge scorer."""

from __future__ import annotations

import pytest

from swapcheck.providers.mock_provider import MockProvider
from swapcheck.scenarios.trajectory import Trajectory, TurnRecord
from swapcheck.scoring.judge import JudgeScore, JudgeScorer


def _make_trajectory(scenario_id: str = "SCN-1") -> Trajectory:
    return Trajectory(
        scenario_id=scenario_id,
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
                cost=0.001,
            )
        ],
        total_latency_ms=10.0,
        total_tokens=20,
        total_cost=0.001,
        final_db_snapshot={},
    )


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

    trajectory = _make_trajectory("SCN-1")
    score = await JudgeScorer(provider=provider).score(trajectory)
    assert score.helpfulness == 5
    assert score.groundedness == 4
    assert score.tone == 5
    assert score.off_policy_promises is False
    assert score.passed is True
    assert score.judge_error is None
    assert score.acceptable is True
    assert score.model == "judge-mock"


@pytest.mark.asyncio
async def test_judge_scorer_threshold_enforcement() -> None:
    """When scores are below thresholds, passed must be False even if json said true."""
    provider = MockProvider(model_name="judge-mock")
    # Groundedness 3 is below the default threshold of 4
    provider.enqueue_response(
        content=(
            '{"helpfulness": 4, "groundedness": 3, "tone": 4, '
            '"off_policy_promises": false, "passed": true, '
            '"explanation": "Mostly fine but low groundedness."}'
        )
    )

    trajectory = _make_trajectory("SCN-threshold")
    score = await JudgeScorer(provider=provider, min_groundedness=4).score(trajectory)
    assert score.groundedness == 3
    assert score.passed is False
    assert score.acceptable is False


@pytest.mark.asyncio
async def test_judge_scorer_off_policy_fails_pass() -> None:
    """Off-policy promises must fail the judge pass test."""
    provider = MockProvider(model_name="judge-mock")
    provider.enqueue_response(
        content=(
            '{"helpfulness": 5, "groundedness": 5, "tone": 5, '
            '"off_policy_promises": true, "passed": true, '
            '"explanation": "Offered unauthorized free refund."}'
        )
    )

    trajectory = _make_trajectory("SCN-off-policy")
    score = await JudgeScorer(provider=provider).score(trajectory)
    assert score.off_policy_promises is True
    assert score.passed is False
    assert score.acceptable is False


@pytest.mark.asyncio
async def test_judge_scorer_retries_on_invalid_json_then_succeeds() -> None:
    """The judge scorer should retry when invalid JSON is received and succeed on retry."""
    provider = MockProvider(model_name="judge-mock")
    # First response is bad JSON
    provider.enqueue_response(content="Sorry, here is the result: invalid json")
    # Second response is valid
    provider.enqueue_response(
        content=(
            '{"helpfulness": 4, "groundedness": 4, "tone": 4, '
            '"off_policy_promises": false, "passed": true, '
            '"explanation": "Correct on second try."}'
        )
    )

    trajectory = _make_trajectory("SCN-retry-success")
    score = await JudgeScorer(provider=provider, max_retries=2).score(trajectory)
    assert score.passed is True
    assert score.helpfulness == 4
    assert score.judge_error is None


@pytest.mark.asyncio
async def test_judge_scorer_exhausts_retries_returns_judge_error() -> None:
    """When all retries fail, return judge_error set and passed=False; never default to pass."""
    provider = MockProvider(model_name="judge-mock")
    provider.enqueue_response(content="bad 1")
    provider.enqueue_response(content="bad 2")
    provider.enqueue_response(content="bad 3")

    trajectory = _make_trajectory("SCN-fail-all")
    score = await JudgeScorer(provider=provider, max_retries=2).score(trajectory)
    assert score.judge_error is not None
    assert score.passed is False
    assert score.acceptable is False
    assert "Judge failed after 2 retries" in score.explanation


def test_judge_score_default_never_passed() -> None:
    """JudgeScore defaults must never default to passed=True."""
    score = JudgeScore()
    assert score.passed is False
    assert score.acceptable is False
