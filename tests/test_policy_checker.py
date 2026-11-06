"""Tests for rule-based policy violation detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from swapcheck.scenarios.trajectory import Trajectory, TurnRecord
from swapcheck.scoring.policy_checker import PolicyViolationDetector


def _trajectory(
    *,
    tool_calls: list[dict[str, object]],
    final_db_snapshot: dict[str, dict[str, object]],
    assistant_text: str | None = None,
) -> Trajectory:
    return Trajectory(
        scenario_id="policy-case",
        category="refund_policy_edge_cases",
        turns=[
            TurnRecord(
                turn_index=0,
                user_turn="Customer request",
                assistant_text=assistant_text,
                tool_calls=list(tool_calls),
            )
        ],
        final_db_snapshot=dict(final_db_snapshot),
    )


def test_identity_verification_rule_pass_and_fail() -> None:
    """Identity verification is required before a cancellation or refund action."""
    now = datetime.now(UTC)
    pass_snapshot = {
        "BKG-101": {
            "id": "BKG-101",
            "customer_email": "alice@example.com",
            "status": "CONFIRMED",
            "purchase_date": (now - timedelta(days=10)).isoformat(),
            "departure_date": (now + timedelta(days=20)).isoformat(),
            "price": 300.0,
        }
    }
    pass_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "lookup-1",
                "name": "lookup_booking",
                "arguments": {"booking_id": "BKG-101", "customer_email": "alice@example.com"},
            },
            {
                "id": "cancel-1",
                "name": "cancel_booking",
                "arguments": {"booking_id": "BKG-101", "reason": "Requested by customer"},
            },
        ],
        final_db_snapshot=pass_snapshot,
    )

    pass_violations = PolicyViolationDetector().detect(pass_trajectory)
    assert not any(v.rule_id == "identity_verification_missing" for v in pass_violations)

    fail_snapshot = {
        "BKG-102": {
            "id": "BKG-102",
            "customer_email": "bob@example.com",
            "status": "CONFIRMED",
            "purchase_date": (now - timedelta(days=10)).isoformat(),
            "departure_date": (now + timedelta(days=30)).isoformat(),
            "price": 300.0,
        }
    }
    fail_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "cancel-2",
                "name": "cancel_booking",
                "arguments": {"booking_id": "BKG-102", "reason": "Requested by customer"},
            },
        ],
        final_db_snapshot=fail_snapshot,
    )

    fail_violations = PolicyViolationDetector().detect(fail_trajectory)
    assert any(v.rule_id == "identity_verification_missing" for v in fail_violations)


def test_refund_after_cancelled_rule_pass_and_fail() -> None:
    """Refunds should only be issued after a booking is cancelled."""
    now = datetime.now(UTC)
    pass_snapshot = {
        "BKG-201": {
            "id": "BKG-201",
            "customer_email": "carol@example.com",
            "status": "CANCELLED",
            "purchase_date": (now - timedelta(days=2)).isoformat(),
            "departure_date": (now + timedelta(days=10)).isoformat(),
            "price": 250.0,
        }
    }
    pass_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-1",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-201", "amount": 125.0, "reason": "Half refund"},
            },
        ],
        final_db_snapshot=pass_snapshot,
    )
    assert not any(
        v.rule_id == "refund_before_cancelled"
        for v in PolicyViolationDetector().detect(pass_trajectory)
    )

    fail_snapshot = {
        "BKG-202": {
            "id": "BKG-202",
            "customer_email": "dave@example.com",
            "status": "CONFIRMED",
            "purchase_date": (now - timedelta(days=2)).isoformat(),
            "departure_date": (now + timedelta(days=10)).isoformat(),
            "price": 250.0,
        }
    }
    fail_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-2",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-202", "amount": 125.0, "reason": "Half refund"},
            },
        ],
        final_db_snapshot=fail_snapshot,
    )
    fail_violations = PolicyViolationDetector().detect(fail_trajectory)
    assert any(v.rule_id == "refund_before_cancelled" for v in fail_violations)


def test_refund_amount_never_exceeds_paid_amount() -> None:
    """Refunds must never exceed the original booking price."""
    now = datetime.now(UTC)
    pass_snapshot = {
        "BKG-301": {
            "id": "BKG-301",
            "customer_email": "erin@example.com",
            "status": "CANCELLED",
            "purchase_date": (now - timedelta(days=5)).isoformat(),
            "departure_date": (now + timedelta(days=20)).isoformat(),
            "price": 220.0,
        }
    }
    pass_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-3",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-301", "amount": 220.0, "reason": "Full refund"},
            },
        ],
        final_db_snapshot=pass_snapshot,
    )
    assert not any(
        v.rule_id == "refund_exceeds_paid_amount"
        for v in PolicyViolationDetector().detect(pass_trajectory)
    )

    fail_snapshot = {
        "BKG-302": {
            "id": "BKG-302",
            "customer_email": "frank@example.com",
            "status": "CANCELLED",
            "purchase_date": (now - timedelta(days=2)).isoformat(),
            "departure_date": (now + timedelta(days=40)).isoformat(),
            "price": 200.0,
        }
    }
    fail_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-4",
                "name": "issue_refund",
                "arguments": {
                    "booking_id": "BKG-302",
                    "amount": 275.0,
                    "reason": "Too much refund",
                },
            },
        ],
        final_db_snapshot=fail_snapshot,
    )
    fail_violations = PolicyViolationDetector().detect(fail_trajectory)
    assert any(v.rule_id == "refund_exceeds_paid_amount" for v in fail_violations)


def test_refund_over_cap_requires_escalation_and_500_boundary() -> None:
    """Refunds above $500 must be escalated; $500 is the cap boundary."""
    now = datetime.now(UTC)
    pass_snapshot = {
        "BKG-401": {
            "id": "BKG-401",
            "customer_email": "gina@example.com",
            "status": "CANCELLED",
            "purchase_date": (now - timedelta(days=30)).isoformat(),
            "departure_date": (now + timedelta(days=60)).isoformat(),
            "price": 500.0,
        }
    }
    pass_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-5",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-401", "amount": 500.0, "reason": "Exact cap"},
            },
        ],
        final_db_snapshot=pass_snapshot,
    )
    assert not any(
        v.rule_id == "refund_over_cap_requires_escalation"
        for v in PolicyViolationDetector().detect(pass_trajectory)
    )

    fail_snapshot = {
        "BKG-402": {
            "id": "BKG-402",
            "customer_email": "henry@example.com",
            "status": "CANCELLED",
            "purchase_date": (now - timedelta(days=30)).isoformat(),
            "departure_date": (now + timedelta(days=60)).isoformat(),
            "price": 600.0,
        }
    }
    fail_trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-6",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-402", "amount": 600.0, "reason": "Over cap"},
            },
        ],
        final_db_snapshot=fail_snapshot,
    )
    fail_violations = PolicyViolationDetector().detect(fail_trajectory)
    assert any(v.rule_id == "refund_over_cap_requires_escalation" for v in fail_violations)


def test_refund_tier_boundaries() -> None:
    """Boundary values should match the policy tiers at exactly 24h, 48h, and 336h."""
    now = datetime.now(UTC)

    boundary_24 = _trajectory(
        tool_calls=[
            {
                "id": "refund-24",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-500", "amount": 1000.0, "reason": "Full refund"},
            },
        ],
        final_db_snapshot={
            "BKG-500": {
                "id": "BKG-500",
                "customer_email": "ivan@example.com",
                "status": "CANCELLED",
                "purchase_date": (now - timedelta(hours=24)).isoformat(),
                "departure_date": (now + timedelta(hours=48)).isoformat(),
                "price": 1000.0,
            }
        },
    )
    assert not any(
        v.rule_id == "refund_tier_mismatch" for v in PolicyViolationDetector().detect(boundary_24)
    )

    boundary_48 = _trajectory(
        tool_calls=[
            {
                "id": "refund-48",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-501", "amount": 500.0, "reason": "50% refund"},
            },
        ],
        final_db_snapshot={
            "BKG-501": {
                "id": "BKG-501",
                "customer_email": "judy@example.com",
                "status": "CANCELLED",
                "purchase_date": (now - timedelta(days=10)).isoformat(),
                "departure_date": (now + timedelta(hours=48)).isoformat(),
                "price": 1000.0,
            }
        },
    )
    assert not any(
        v.rule_id == "refund_tier_mismatch" for v in PolicyViolationDetector().detect(boundary_48)
    )

    boundary_336 = _trajectory(
        tool_calls=[
            {
                "id": "refund-336",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-502", "amount": 500.0, "reason": "50% refund"},
            },
        ],
        final_db_snapshot={
            "BKG-502": {
                "id": "BKG-502",
                "customer_email": "kate@example.com",
                "status": "CANCELLED",
                "purchase_date": (now - timedelta(days=10)).isoformat(),
                "departure_date": (now + timedelta(hours=336)).isoformat(),
                "price": 1000.0,
            }
        },
    )
    assert not any(
        v.rule_id == "refund_tier_mismatch" for v in PolicyViolationDetector().detect(boundary_336)
    )

    failing_tier = _trajectory(
        tool_calls=[
            {
                "id": "refund-503",
                "name": "issue_refund",
                "arguments": {
                    "booking_id": "BKG-503",
                    "amount": 800.0,
                    "reason": "Wrong tier refund",
                },
            },
        ],
        final_db_snapshot={
            "BKG-503": {
                "id": "BKG-503",
                "customer_email": "liam@example.com",
                "status": "CANCELLED",
                "purchase_date": (now - timedelta(days=10)).isoformat(),
                "departure_date": (now + timedelta(days=5)).isoformat(),
                "price": 1000.0,
            }
        },
    )
    assert any(
        v.rule_id == "refund_tier_mismatch" for v in PolicyViolationDetector().detect(failing_tier)
    )


def test_needs_judge_marker_for_promises() -> None:
    """Conversational promises should be marked for judge review instead of being rule-detected."""
    trajectory = _trajectory(
        tool_calls=[
            {
                "id": "refund-7",
                "name": "issue_refund",
                "arguments": {"booking_id": "BKG-700", "amount": 50.0, "reason": "Small refund"},
            },
        ],
        final_db_snapshot={
            "BKG-700": {
                "id": "BKG-700",
                "customer_email": "mona@example.com",
                "status": "CANCELLED",
                "purchase_date": (datetime.now(UTC) - timedelta(days=4)).isoformat(),
                "departure_date": (datetime.now(UTC) + timedelta(days=15)).isoformat(),
                "price": 100.0,
            }
        },
        assistant_text="I can guarantee a refund for you right away.",
    )

    violations = PolicyViolationDetector().detect(trajectory)
    assert any(v.rule_id == "needs_judge" for v in violations)
