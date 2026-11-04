"""Tests for deterministic trajectory scoring."""

from __future__ import annotations

from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.trajectory import Trajectory, TurnRecord
from swapcheck.scoring.deterministic import DeterministicScorer


def _trajectory(*, tool_calls: list[dict[str, object]], final_snapshot: dict[str, dict[str, object]]) -> Trajectory:
    return Trajectory(
        scenario_id="scenario-1",
        category="simple_lookup",
        turns=[TurnRecord(turn_index=0, user_turn="Hello", tool_calls=list(tool_calls))],
        final_db_snapshot=dict(final_snapshot),
    )


def test_deterministic_scorer_accepts_valid_run() -> None:
    """A valid lookup + cancel flow should pass deterministic scoring."""
    scenario = Scenario(
        id="scenario-valid",
        category="simple_lookup",
        description="Valid run.",
        initial_db_state=[{"id": "BKG-1", "customer_email": "alice@example.com", "status": "CONFIRMED", "price": 300.0}],
        user_turns=["Please cancel BKG-1."],
        expected_final_db_state=[{"id": "BKG-1", "status": "CANCELLED"}],
        forbidden_tool_calls=["issue_refund"],
        expected_tool_calls=["lookup_booking", "cancel_booking"],
    )
    trajectory = _trajectory(
        tool_calls=[
            {
                "id": "lookup-1",
                "name": "lookup_booking",
                "arguments": {"booking_id": "BKG-1", "customer_email": "alice@example.com"},
            },
            {
                "id": "cancel-1",
                "name": "cancel_booking",
                "arguments": {"booking_id": "BKG-1", "reason": "Requested by customer"},
            },
        ],
        final_snapshot={
            "BKG-1": {
                "id": "BKG-1",
                "customer_email": "alice@example.com",
                "status": "CANCELLED",
                "price": 300.0,
            }
        },
    )

    score = DeterministicScorer().score(trajectory, scenario)

    assert score.passed is True
    assert score.tool_selection_ok is True
    assert score.args_valid is True
    assert score.db_state_matches is True
    assert score.forbidden_action_violations == []


def test_deterministic_scorer_flags_invalid_args_and_forbidden_tools() -> None:
    """Bad arguments and forbidden refund calls should fail the deterministic gate."""
    scenario = Scenario(
        id="scenario-invalid",
        category="refund_policy_edge_cases",
        description="Invalid run.",
        initial_db_state=[{"id": "BKG-2", "customer_email": "bob@example.com", "status": "CANCELLED", "price": 1000.0}],
        user_turns=["Please refund BKG-2."],
        expected_final_db_state=[{"id": "BKG-2", "status": "CANCELLED"}],
        forbidden_tool_calls=["issue_refund"],
        expected_tool_calls=["issue_refund"],
    )
    trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-1",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-2", "reason": "Refund approved"},
            },
        ],
        final_snapshot={
            "BKG-2": {
                "id": "BKG-2",
                "customer_email": "bob@example.com",
                "status": "CANCELLED",
                "price": 1000.0,
            }
        },
    )

    score = DeterministicScorer().score(trajectory, scenario)

    assert score.tool_selection_ok is True
    assert score.args_valid is False
    assert score.db_state_matches is True
    assert score.forbidden_action_violations == ["issue_refund"]
    assert any("missing required arguments" in reason for reason in score.failure_reasons)


def test_deterministic_scorer_reports_db_state_diff() -> None:
    """A mismatched final DB snapshot should be surfaced as a readable diff."""
    scenario = Scenario(
        id="scenario-db-mismatch",
        category="simple_lookup",
        description="Database expectation mismatch.",
        initial_db_state=[{"id": "BKG-3", "customer_email": "carol@example.com", "status": "CONFIRMED", "price": 50.0}],
        user_turns=["Check BKG-3."],
        expected_final_db_state=[{"id": "BKG-3", "status": "CANCELLED"}],
        forbidden_tool_calls=[],
        expected_tool_calls=["lookup_booking"],
    )
    trajectory = _trajectory(
        tool_calls=[
            {
                "id": "lookup-3",
                "name": "lookup_booking",
                "arguments": {"booking_id": "BKG-3", "customer_email": "carol@example.com"},
            }
        ],
        final_snapshot={
            "BKG-3": {
                "id": "BKG-3",
                "customer_email": "carol@example.com",
                "status": "CONFIRMED",
                "price": 50.0,
            }
        },
    )

    score = DeterministicScorer().score(trajectory, scenario)

    assert score.db_state_matches is False
    assert any("DB mismatch" in reason for reason in score.failure_reasons)
