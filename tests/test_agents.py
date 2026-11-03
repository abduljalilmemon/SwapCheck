"""Tests for demo agent, tools, mock transactional DB, and policy logic."""

from datetime import UTC, datetime, timedelta

import pytest

from swapcheck.agents.booking_agent import BookingSupportAgent
from swapcheck.agents.db import Booking, BookingDatabase
from swapcheck.agents.policy import calculate_eligible_refund
from swapcheck.agents.tools import (
    cancel_booking,
    escalate_to_human,
    execute_booking_tool,
    issue_refund,
    lookup_booking,
)
from swapcheck.providers.base import ToolCall
from swapcheck.providers.mock_provider import MockProvider


@pytest.fixture
def sample_booking() -> Booking:
    """Provide a standard confirmed booking."""
    now = datetime.now(UTC)
    return Booking(
        id="BKG-101",
        customer_name="Alice Smith",
        customer_email="alice@example.com",
        item_type="flight",
        details="Flight from SFO to JFK",
        purchase_date=now - timedelta(days=2),
        departure_date=now + timedelta(days=20),
        price=450.0,
        status="CONFIRMED",
    )


@pytest.fixture
def mock_db(sample_booking: Booking) -> BookingDatabase:
    """Provide a database pre-seeded with sample_booking."""
    return BookingDatabase([sample_booking])


# ============================================================================
# Database Tests
# ============================================================================


def test_db_snapshot_restore_and_clone(mock_db: BookingDatabase, sample_booking: Booking) -> None:
    """Test taking snapshots, restoring, and cloning DB states."""
    snapshot = mock_db.snapshot()
    assert "BKG-101" in snapshot
    assert snapshot["BKG-101"]["status"] == "CONFIRMED"

    # Modify state in DB
    sample_booking.status = "CANCELLED"
    mock_db.upsert(sample_booking)
    assert mock_db.get("BKG-101") is not None
    assert mock_db.get("BKG-101").status == "CANCELLED"  # type: ignore[union-attr]

    # Restore snapshot
    mock_db.restore(snapshot)
    restored = mock_db.get("BKG-101")
    assert restored is not None
    assert restored.status == "CONFIRMED"

    # Clone
    cloned_db = mock_db.clone()
    assert cloned_db.get("BKG-101") is not None
    assert cloned_db.get("BKG-101").customer_name == "Alice Smith"  # type: ignore[union-attr]


def test_db_diff(sample_booking: Booking) -> None:
    """Test state diffing between two snapshots."""
    db = BookingDatabase([sample_booking])
    snap_before = db.snapshot()

    # Modify existing and add new
    b1 = db.get("BKG-101")
    assert b1 is not None
    b1.status = "CANCELLED"
    b1.refunded_amount = 360.0
    db.upsert(b1)

    b2 = Booking(
        id="BKG-102",
        customer_name="Bob Jones",
        customer_email="bob@example.com",
        item_type="hotel",
        details="Hotel in NYC",
        purchase_date=datetime.now(UTC),
        departure_date=datetime.now(UTC) + timedelta(days=5),
        price=200.0,
    )
    db.upsert(b2)
    snap_after = db.snapshot()

    diff = BookingDatabase.diff(snap_before, snap_after)
    assert diff["has_changes"] is True
    assert diff["added"] == ["BKG-102"]
    assert diff["removed"] == []
    assert "BKG-101" in diff["modified"]
    assert diff["modified"]["BKG-101"]["status"] == {
        "before": "CONFIRMED",
        "after": "CANCELLED",
    }
    assert diff["modified"]["BKG-101"]["refunded_amount"] == {
        "before": 0.0,
        "after": 360.0,
    }


# ============================================================================
# Policy Calculation Tests
# ============================================================================


def test_policy_cooling_off_rule_1() -> None:
    """Test Rule 1: 24-hr purchase window and >=48h before departure grants 100%."""
    now = datetime.now(UTC)
    purchase = now - timedelta(hours=12)  # within 24h
    departure = now + timedelta(days=5)  # >=48h

    calc = calculate_eligible_refund(
        booking_price=400.0,
        purchase_date=purchase,
        departure_date=departure,
        current_time=now,
    )
    assert calc.eligible_percentage == 1.0
    assert calc.calculated_amount == 400.0
    assert calc.requires_escalation is False
    assert "Rule 1" in calc.rule_applied


def test_policy_cooling_off_fails_if_departure_too_soon() -> None:
    """Test Rule 1 does NOT apply if departure is <48h away even if purchased within 24h."""
    now = datetime.now(UTC)
    purchase = now - timedelta(hours=5)  # within 24h
    departure = now + timedelta(hours=24)  # <48h away!

    calc = calculate_eligible_refund(
        booking_price=400.0,
        purchase_date=purchase,
        departure_date=departure,
        current_time=now,
    )
    # Fails Rule 1 -> falls back to Rule 2.3 (<48h non-refundable)
    assert calc.eligible_percentage == 0.0
    assert calc.calculated_amount == 0.0
    assert calc.requires_escalation is False


def test_policy_rule_2_departure_tiers() -> None:
    """Test Rule 2 tiers (>14 days -> 80%, 48h-14d -> 50%, <48h -> 0%)."""
    now = datetime.now(UTC)
    purchase = now - timedelta(days=10)  # >24h since purchase

    # Tier 1: >14 days before departure -> 80%
    dep_tier1 = now + timedelta(days=20)
    calc1 = calculate_eligible_refund(500.0, purchase, dep_tier1, current_time=now)
    assert calc1.eligible_percentage == 0.80
    assert calc1.calculated_amount == 400.0
    assert calc1.requires_escalation is False

    # Tier 2: between 48h and 14 days -> 50%
    dep_tier2 = now + timedelta(days=5)
    calc2 = calculate_eligible_refund(500.0, purchase, dep_tier2, current_time=now)
    assert calc2.eligible_percentage == 0.50
    assert calc2.calculated_amount == 250.0
    assert calc2.requires_escalation is False

    # Tier 3: <48 hours -> 0% non-refundable
    dep_tier3 = now + timedelta(hours=36)
    calc3 = calculate_eligible_refund(500.0, purchase, dep_tier3, current_time=now)
    assert calc3.eligible_percentage == 0.0
    assert calc3.calculated_amount == 0.0


def test_policy_escalation_cap() -> None:
    """Test that calculated amounts > $500 flag requires_escalation."""
    now = datetime.now(UTC)
    purchase = now - timedelta(days=5)
    departure = now + timedelta(days=30)  # >14 days -> 80%

    # $800 price * 0.80 = $640 (> $500 cap)
    calc = calculate_eligible_refund(800.0, purchase, departure, current_time=now)
    assert calc.calculated_amount == 640.0
    assert calc.requires_escalation is True


# ============================================================================
# Tool Tests
# ============================================================================


def test_lookup_booking_tool(mock_db: BookingDatabase) -> None:
    """Test lookup_booking authentication and data disclosure."""
    auth_state: dict[str, bool] = {}

    # Unknown ID
    res_not_found = lookup_booking("UNKNOWN", db=mock_db, auth_state=auth_state)
    assert res_not_found.success is False
    assert "not found" in res_not_found.error.lower()  # type: ignore[union-attr]

    # No email provided -> auth required
    res_no_email = lookup_booking("BKG-101", db=mock_db, auth_state=auth_state)
    assert res_no_email.success is False
    assert res_no_email.data["verified"] is False
    assert auth_state.get("BKG-101") is not True

    # Wrong email -> auth fails
    res_wrong_email = lookup_booking(
        "BKG-101", customer_email="wrong@example.com", db=mock_db, auth_state=auth_state
    )
    assert res_wrong_email.success is False
    assert res_wrong_email.data["verified"] is False

    # Correct email -> verified!
    res_verified = lookup_booking(
        "BKG-101", customer_email="alice@example.com", db=mock_db, auth_state=auth_state
    )
    assert res_verified.success is True
    assert res_verified.data["verified"] is True
    assert res_verified.data["customer_name"] == "Alice Smith"
    assert auth_state.get("BKG-101") is True


def test_cancel_booking_tool(mock_db: BookingDatabase) -> None:
    """Test cancel_booking verification requirement and state mutation."""
    auth_state: dict[str, bool] = {}

    # Unverified cancellation should be rejected
    res_unauth = cancel_booking("BKG-101", db=mock_db, auth_state=auth_state)
    assert res_unauth.success is False
    assert "unauthorized" in res_unauth.error.lower()  # type: ignore[union-attr]

    # Authenticate
    auth_state["BKG-101"] = True

    # Valid cancellation
    res_cancel = cancel_booking(
        "BKG-101", reason="Personal conflict", db=mock_db, auth_state=auth_state
    )
    assert res_cancel.success is True
    assert res_cancel.data["status"] == "CANCELLED"
    assert mock_db.get("BKG-101").status == "CANCELLED"  # type: ignore[union-attr]

    # Double cancellation rejected
    res_double = cancel_booking("BKG-101", db=mock_db, auth_state=auth_state)
    assert res_double.success is False
    assert "already cancelled" in res_double.error.lower()  # type: ignore[union-attr]


def test_issue_refund_tool(mock_db: BookingDatabase) -> None:
    """Test issue_refund status prerequisites, cap enforcement, and limits."""
    auth_state: dict[str, bool] = {"BKG-101": True}

    # Attempt refund before cancellation
    res_not_cancelled = issue_refund("BKG-101", amount=100.0, db=mock_db, auth_state=auth_state)
    assert res_not_cancelled.success is False
    assert "must be cancelled before a refund can be issued" in res_not_cancelled.error.lower()  # type: ignore[union-attr]

    # Cancel booking
    cancel_booking("BKG-101", db=mock_db, auth_state=auth_state)

    # Attempt refund over $500 cap
    res_cap = issue_refund("BKG-101", amount=550.0, db=mock_db, auth_state=auth_state)
    assert res_cap.success is False
    assert "cap exceeded" in res_cap.error.lower()  # type: ignore[union-attr]

    # Attempt refund over booking price ($450)
    res_overprice = issue_refund("BKG-101", amount=480.0, db=mock_db, auth_state=auth_state)
    assert res_overprice.success is False
    assert "exceeds total booking price" in res_overprice.error.lower()  # type: ignore[union-attr]

    # Valid refund
    res_valid = issue_refund(
        "BKG-101",
        amount=360.0,
        reason="80% policy refund",
        db=mock_db,
        auth_state=auth_state,
    )
    assert res_valid.success is True
    assert res_valid.data["refunded_amount"] == 360.0
    assert mock_db.get("BKG-101").status == "REFUNDED"  # type: ignore[union-attr]


def test_escalate_to_human_tool(mock_db: BookingDatabase) -> None:
    """Test escalate_to_human tool."""
    res = escalate_to_human(
        "BKG-101",
        reason="Refund amount $750 exceeds autonomous $500 cap.",
        db=mock_db,
    )
    assert res.success is True
    assert res.data["escalated"] is True
    booking = mock_db.get("BKG-101")
    assert booking is not None
    assert booking.escalated is True
    assert "Refund amount $750" in (booking.escalation_reason or "")


def test_execute_booking_tool_dispatch(mock_db: BookingDatabase) -> None:
    """Test dispatch helper execute_booking_tool."""
    auth_state: dict[str, bool] = {}

    r = execute_booking_tool(
        "lookup_booking",
        {"booking_id": "BKG-101", "customer_email": "alice@example.com"},
        db=mock_db,
        auth_state=auth_state,
    )
    assert r.success is True

    r_unknown = execute_booking_tool("invalid_tool", {}, db=mock_db, auth_state=auth_state)
    assert r_unknown.success is False
    assert "unknown tool" in r_unknown.error.lower()  # type: ignore[union-attr]


# ============================================================================
# Agent Step Execution Loop Tests
# ============================================================================


@pytest.mark.asyncio
async def test_booking_agent_tool_loop(mock_db: BookingDatabase) -> None:
    """Test that BookingSupportAgent executes multi-turn tool loops via MockProvider."""
    provider = MockProvider(model_name="mock-agent-model")
    agent = BookingSupportAgent(provider=provider, db=mock_db)

    # 1. Enqueue assistant requesting lookup_booking with customer email
    provider.enqueue_response(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_lookup_1",
                name="lookup_booking",
                arguments={"booking_id": "BKG-101", "customer_email": "alice@example.com"},
            )
        ],
    )
    # 2. Enqueue assistant calling cancel_booking and issue_refund
    provider.enqueue_response(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_cancel_1",
                name="cancel_booking",
                arguments={"booking_id": "BKG-101", "reason": "Customer request"},
            ),
            ToolCall(
                id="call_refund_1",
                name="issue_refund",
                arguments={"booking_id": "BKG-101", "amount": 360.0, "reason": "80% refund"},
            ),
        ],
    )
    # 3. Enqueue final assistant conversational response
    provider.enqueue_response(
        content="I have cancelled your booking BKG-101 and processed your $360.00 refund."
    )

    step_result = await agent.step(
        "Please cancel booking BKG-101 for alice@example.com and issue my refund."
    )

    assert step_result.response_text is not None
    assert "processed your $360.00 refund" in step_result.response_text
    assert len(step_result.tool_calls) == 3
    assert [tc.name for tc in step_result.tool_calls] == [
        "lookup_booking",
        "cancel_booking",
        "issue_refund",
    ]

    # Verify DB was mutated accordingly
    b = mock_db.get("BKG-101")
    assert b is not None
    assert b.status == "REFUNDED"
    assert b.refunded_amount == 360.0

    # Test agent reset
    agent.reset()
    assert len(agent.messages) == 1
    assert agent.messages[0].role == "system"
    assert len(agent.auth_state) == 0


def test_load_policy_document() -> None:
    """Test loading POLICY.md content."""
    from swapcheck.agents.policy import load_policy_document

    doc = load_policy_document()
    assert "Booking Support Service Policy" in doc
    assert "24-Hour Purchase Cooling-Off Window" in doc
