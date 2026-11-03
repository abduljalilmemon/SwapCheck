"""Policy specifications and helper logic for booking cancellations and refunds."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel


class RefundCalculation(NamedTuple):
    """Result of policy-based refund calculation."""

    eligible_percentage: float  # e.g. 1.0, 0.8, 0.5, 0.0
    calculated_amount: float
    requires_escalation: bool
    rule_applied: str
    explanation: str


class PolicyRules(BaseModel):
    """Configurable numerical boundaries of the written policy."""

    cooling_off_hours: float = 24.0
    cooling_off_min_departure_hours: float = 48.0
    tier_1_departure_days: float = 14.0
    tier_1_refund_rate: float = 0.80
    tier_2_departure_hours: float = 48.0
    tier_2_refund_rate: float = 0.50
    autonomous_refund_cap: float = 500.0


DEFAULT_POLICY = PolicyRules()


def load_policy_document() -> str:
    """Load the raw POLICY.md text."""
    policy_path = Path("POLICY.md")
    if not policy_path.exists():
        # Fallback to parent or relative path
        policy_path = Path(__file__).resolve().parent.parent.parent / "POLICY.md"
    if policy_path.exists():
        return policy_path.read_text(encoding="utf-8")
    return "Booking Support Policy: see POLICY.md."


def calculate_eligible_refund(
    booking_price: float,
    purchase_date: datetime,
    departure_date: datetime,
    current_time: datetime | None = None,
    rules: PolicyRules = DEFAULT_POLICY,
) -> RefundCalculation:
    """Calculate eligible refund amount and escalation status based on POLICY.md."""
    if current_time is None:
        current_time = datetime.now(UTC)

    # Ensure tz-aware comparisons
    if purchase_date.tzinfo is None:
        purchase_date = purchase_date.replace(tzinfo=UTC)
    if departure_date.tzinfo is None:
        departure_date = departure_date.replace(tzinfo=UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)

    hours_since_purchase = (current_time - purchase_date).total_seconds() / 3600.0
    hours_until_departure = (departure_date - current_time).total_seconds() / 3600.0

    # Rule 1: 24-Hour Purchase Cooling-Off Window
    if (
        hours_since_purchase <= rules.cooling_off_hours
        and hours_until_departure >= rules.cooling_off_min_departure_hours
    ):
        amount = booking_price
        req_esc = amount > rules.autonomous_refund_cap
        return RefundCalculation(
            eligible_percentage=1.0,
            calculated_amount=amount,
            requires_escalation=req_esc,
            rule_applied="Rule 1 (24-Hour Cooling-Off Window)",
            explanation=(
                f"Requested within {hours_since_purchase:.1f}h of purchase (<= {rules.cooling_off_hours}h) "
                f"and {hours_until_departure:.1f}h before departure (>= {rules.cooling_off_min_departure_hours}h). "
                f"Entitled to 100% refund (${amount:.2f})."
            ),
        )

    # Rule 2: Standard Departure-Time Windows (when >24h from purchase or departure <48h)
    days_until_departure = hours_until_departure / 24.0

    if days_until_departure > rules.tier_1_departure_days:
        pct = rules.tier_1_refund_rate
        amount = round(booking_price * pct, 2)
        req_esc = amount > rules.autonomous_refund_cap
        return RefundCalculation(
            eligible_percentage=pct,
            calculated_amount=amount,
            requires_escalation=req_esc,
            rule_applied="Rule 2.1 (>14 Days Before Departure)",
            explanation=(
                f"Cancellation requested {days_until_departure:.1f} days before departure (>14 days). "
                f"Entitled to 80% refund (${amount:.2f})."
            ),
        )
    elif hours_until_departure >= rules.tier_2_departure_hours:
        pct = rules.tier_2_refund_rate
        amount = round(booking_price * pct, 2)
        req_esc = amount > rules.autonomous_refund_cap
        return RefundCalculation(
            eligible_percentage=pct,
            calculated_amount=amount,
            requires_escalation=req_esc,
            rule_applied="Rule 2.2 (48h to 14 Days Before Departure)",
            explanation=(
                f"Cancellation requested {hours_until_departure:.1f}h before departure (48h-14d). "
                f"Entitled to 50% refund (${amount:.2f})."
            ),
        )
    else:
        # Strictly non-refundable (<48h before departure)
        return RefundCalculation(
            eligible_percentage=0.0,
            calculated_amount=0.0,
            requires_escalation=False,
            rule_applied="Rule 2.3 (<48h Before Departure Non-Refundable)",
            explanation=(
                f"Cancellation requested {hours_until_departure:.1f}h before departure (<48h). "
                "Non-refundable according to policy."
            ),
        )
