"""Tools for the booking support agent: lookup, cancel, refund, and escalate."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from swapcheck.agents.db import BookingDatabase


class ToolResult(BaseModel):
    """Result of an executed tool call."""

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def to_json_str(self) -> str:
        """Serialize tool output to JSON string for the LLM."""
        res: dict[str, Any] = {"success": self.success}
        if self.data:
            res["data"] = self.data
        if self.error:
            res["error"] = self.error
        return json.dumps(res)


# ============================================================================
# Tool Schemas for LiteLLM / OpenAI Function Calling
# ============================================================================

BOOKING_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_booking",
            "description": "Look up booking details and verify customer identity using booking ID and registered customer email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "Unique booking reference ID (e.g., 'BKG-101')",
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "The customer's email address for identity verification.",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel a confirmed booking. Requires that customer identity was verified via lookup_booking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID to cancel.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Customer's reason for cancellation.",
                    },
                },
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an authorized refund for a cancelled booking. Maximum autonomous cap is $500. Refunds >$500 must be escalated to a human.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The cancelled booking ID to refund.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "The refund amount in USD (must be <= $500).",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Policy rule or calculation justification for this refund amount.",
                    },
                },
                "required": ["booking_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the request to a human supervisor for refunds over $500, policy exception requests, or complex disputes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "The booking ID being escalated.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed reason why human supervisor intervention is required.",
                    },
                },
                "required": ["booking_id", "reason"],
            },
        },
    },
]


def get_booking_tools_schema() -> list[dict[str, Any]]:
    """Return function calling schema definitions for booking tools."""
    return BOOKING_TOOLS_SCHEMA


# ============================================================================
# Tool Implementations
# ============================================================================


def lookup_booking(
    booking_id: str,
    customer_email: str | None = None,
    db: BookingDatabase | None = None,
    auth_state: dict[str, bool] | None = None,
) -> ToolResult:
    """Retrieve booking details and verify customer identity."""
    if db is None:
        return ToolResult(success=False, error="Database not initialized.")

    booking = db.get(booking_id)
    if booking is None:
        return ToolResult(
            success=False,
            error=f"Booking '{booking_id}' not found in the reservation system.",
        )

    # Check identity verification
    is_verified = False
    if customer_email and customer_email.strip().lower() == booking.customer_email.strip().lower():
        is_verified = True
        if auth_state is not None:
            auth_state[booking_id] = True

    if not is_verified:
        # Return partial/auth required message without disclosing sensitive passenger data
        return ToolResult(
            success=False,
            data={
                "booking_id": booking.id,
                "status": booking.status,
                "verified": False,
                "message": (
                    "Identity verification required. Please confirm the customer's registered "
                    "email address before private details or changes can be processed."
                ),
            },
            error="Identity not verified: customer_email missing or did not match booking record.",
        )

    # Verified: return full details
    return ToolResult(
        success=True,
        data={
            "booking_id": booking.id,
            "customer_name": booking.customer_name,
            "customer_email": booking.customer_email,
            "item_type": booking.item_type,
            "details": booking.details,
            "purchase_date": booking.purchase_date.isoformat(),
            "departure_date": booking.departure_date.isoformat(),
            "price": booking.price,
            "status": booking.status,
            "refunded_amount": booking.refunded_amount,
            "verified": True,
            "escalated": booking.escalated,
        },
    )


def cancel_booking(
    booking_id: str,
    reason: str = "",
    db: BookingDatabase | None = None,
    auth_state: dict[str, bool] | None = None,
) -> ToolResult:
    """Cancel an active booking."""
    if db is None:
        return ToolResult(success=False, error="Database not initialized.")

    # Verification check
    if auth_state is not None and not auth_state.get(booking_id, False):
        return ToolResult(
            success=False,
            error="Unauthorized: Customer identity must be verified via lookup_booking before cancelling.",
        )

    booking = db.get(booking_id)
    if booking is None:
        return ToolResult(success=False, error=f"Booking '{booking_id}' not found.")

    if booking.status == "CANCELLED":
        return ToolResult(
            success=False,
            error=f"Booking '{booking_id}' is already cancelled.",
        )

    if booking.status == "REFUNDED":
        return ToolResult(
            success=False,
            error=f"Booking '{booking_id}' has already been refunded.",
        )

    # Perform cancellation
    booking.status = "CANCELLED"
    if reason:
        booking.notes.append(f"Cancelled: {reason}")
    db.upsert(booking)

    return ToolResult(
        success=True,
        data={
            "booking_id": booking.id,
            "status": "CANCELLED",
            "message": f"Booking {booking_id} has been successfully cancelled.",
        },
    )


def issue_refund(
    booking_id: str,
    amount: float,
    reason: str = "",
    db: BookingDatabase | None = None,
    auth_state: dict[str, bool] | None = None,
) -> ToolResult:
    """Issue a refund for a cancelled booking."""
    if db is None:
        return ToolResult(success=False, error="Database not initialized.")

    if auth_state is not None and not auth_state.get(booking_id, False):
        return ToolResult(
            success=False,
            error="Unauthorized: Customer identity must be verified before issuing a refund.",
        )

    booking = db.get(booking_id)
    if booking is None:
        return ToolResult(success=False, error=f"Booking '{booking_id}' not found.")

    if booking.status != "CANCELLED":
        return ToolResult(
            success=False,
            error=(
                f"Booking '{booking_id}' is in status '{booking.status}'. "
                "The reservation must be CANCELLED before a refund can be issued."
            ),
        )

    if amount <= 0:
        return ToolResult(success=False, error="Refund amount must be positive.")

    # Check $500 autonomous cap
    if amount > 500.0:
        return ToolResult(
            success=False,
            error=(
                f"Autonomous refund cap exceeded: ${amount:.2f} > $500.00. "
                "Refunds over $500 require human escalation via escalate_to_human."
            ),
            data={"cap_exceeded": True, "amount": amount},
        )

    if amount > booking.price:
        return ToolResult(
            success=False,
            error=f"Refund amount ${amount:.2f} exceeds total booking price ${booking.price:.2f}.",
        )

    # Issue refund
    booking.refunded_amount = round(amount, 2)
    booking.status = "REFUNDED"
    if reason:
        booking.notes.append(f"Refunded ${amount:.2f}: {reason}")
    db.upsert(booking)

    return ToolResult(
        success=True,
        data={
            "booking_id": booking.id,
            "refunded_amount": booking.refunded_amount,
            "status": "REFUNDED",
            "message": f"Successfully issued refund of ${booking.refunded_amount:.2f} for booking {booking_id}.",
        },
    )


def escalate_to_human(
    booking_id: str,
    reason: str,
    db: BookingDatabase | None = None,
    **_kwargs: Any,
) -> ToolResult:
    """Escalate a request to human supervisor."""
    if db is None:
        return ToolResult(success=False, error="Database not initialized.")

    booking = db.get(booking_id)
    if booking is None:
        return ToolResult(success=False, error=f"Booking '{booking_id}' not found.")

    booking.escalated = True
    booking.escalation_reason = reason
    booking.notes.append(f"Escalated to human: {reason}")
    db.upsert(booking)

    return ToolResult(
        success=True,
        data={
            "booking_id": booking.id,
            "escalated": True,
            "escalation_reason": reason,
            "message": f"Booking {booking_id} has been escalated to human supervisor for review.",
        },
    )


def execute_booking_tool(
    name: str,
    arguments: dict[str, Any],
    db: BookingDatabase,
    auth_state: dict[str, bool],
) -> ToolResult:
    """Dispatch and execute tool call by name with validated arguments."""
    if name == "lookup_booking":
        return lookup_booking(
            booking_id=str(arguments.get("booking_id", "")),
            customer_email=arguments.get("customer_email"),
            db=db,
            auth_state=auth_state,
        )
    elif name == "cancel_booking":
        return cancel_booking(
            booking_id=str(arguments.get("booking_id", "")),
            reason=str(arguments.get("reason", "")),
            db=db,
            auth_state=auth_state,
        )
    elif name == "issue_refund":
        try:
            amt = float(arguments.get("amount", 0.0))
        except (ValueError, TypeError):
            amt = 0.0
        return issue_refund(
            booking_id=str(arguments.get("booking_id", "")),
            amount=amt,
            reason=str(arguments.get("reason", "")),
            db=db,
            auth_state=auth_state,
        )
    elif name == "escalate_to_human":
        return escalate_to_human(
            booking_id=str(arguments.get("booking_id", "")),
            reason=str(arguments.get("reason", "")),
            db=db,
            auth_state=auth_state,
        )
    else:
        return ToolResult(success=False, error=f"Unknown tool: '{name}'.")
