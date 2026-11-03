"""In-memory transactional booking database with snapshotting, rollback, and state diffing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Booking(BaseModel):
    """Booking entity representing a customer's flight or hotel reservation."""

    id: str = Field(..., description="Unique booking ID (e.g. BKG-101)")
    customer_name: str = Field(..., description="Full name of customer")
    customer_email: str = Field(..., description="Email address for identity verification")
    item_type: Literal["flight", "hotel"] = Field(..., description="Reservation type")
    details: str = Field(..., description="Flight route, hotel room, or itinerary details")
    purchase_date: datetime = Field(..., description="Timestamp when booking was purchased")
    departure_date: datetime = Field(..., description="Scheduled departure or check-in timestamp")
    price: float = Field(..., ge=0.0, description="Total amount paid in USD")
    status: Literal["CONFIRMED", "CANCELLED", "REFUNDED", "COMPLETED"] = Field(
        default="CONFIRMED", description="Current lifecycle status"
    )
    refunded_amount: float = Field(default=0.0, ge=0.0, description="Total refunded amount in USD")
    notes: list[str] = Field(default_factory=list, description="Audit log of actions on booking")
    escalated: bool = Field(default=False, description="Whether booking was escalated to human")
    escalation_reason: str | None = Field(default=None, description="Reason for human escalation")

    def to_dict(self) -> dict[str, Any]:
        """Convert booking to JSON-serializable dict."""
        data = self.model_dump()
        data["purchase_date"] = self.purchase_date.isoformat()
        data["departure_date"] = self.departure_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Booking:
        """Instantiate booking from dictionary, parsing dates if needed."""
        d = dict(data)
        if isinstance(d.get("purchase_date"), str):
            d["purchase_date"] = datetime.fromisoformat(d["purchase_date"])
        if isinstance(d.get("departure_date"), str):
            d["departure_date"] = datetime.fromisoformat(d["departure_date"])
        return cls.model_validate(d)


class BookingDatabase:
    """In-memory transactional database for demo agent reservations."""

    def __init__(
        self,
        initial_data: dict[str, Booking] | list[Booking] | list[dict[str, Any]] | None = None,
    ) -> None:
        self._bookings: dict[str, Booking] = {}
        if initial_data:
            self.load(initial_data)

    def load(
        self,
        data: dict[str, Booking] | list[Booking] | list[dict[str, Any]],
    ) -> None:
        """Load bookings into database."""
        self._bookings.clear()
        if isinstance(data, dict):
            for k, v in data.items():
                self._bookings[k] = v if isinstance(v, Booking) else Booking.from_dict(v)
        elif isinstance(data, list):
            for item in data:
                b = item if isinstance(item, Booking) else Booking.from_dict(item)
                self._bookings[b.id] = b

    def get(self, booking_id: str) -> Booking | None:
        """Retrieve a booking by ID."""
        booking = self._bookings.get(booking_id)
        if booking is None:
            return None
        # Return a copy to prevent accidental uncommitted in-place mutations
        return booking.model_copy(deep=True)

    def upsert(self, booking: Booking) -> None:
        """Add or update a booking."""
        self._bookings[booking.id] = booking.model_copy(deep=True)

    def all(self) -> list[Booking]:
        """Return all bookings."""
        return [b.model_copy(deep=True) for b in self._bookings.values()]

    def snapshot(self) -> dict[str, Any]:
        """Create a deep JSON-serializable snapshot of current database state."""
        return {b_id: b.to_dict() for b_id, b in self._bookings.items()}

    def restore(self, snapshot_data: dict[str, Any]) -> None:
        """Restore database state from a snapshot."""
        self.load(snapshot_data)

    def clone(self) -> BookingDatabase:
        """Create a fully independent clone of the database."""
        new_db = BookingDatabase()
        new_db._bookings = {k: v.model_copy(deep=True) for k, v in self._bookings.items()}
        return new_db

    @staticmethod
    def diff(
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute the difference between two database snapshots.

        Returns a dict:
        - "added": list of booking IDs added
        - "removed": list of booking IDs removed
        - "modified": dict mapping booking ID to dict of field changes {field: {"before": v1, "after": v2}}
        """
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added = sorted(after_keys - before_keys)
        removed = sorted(before_keys - after_keys)
        common = before_keys & after_keys

        modified: dict[str, dict[str, dict[str, Any]]] = {}

        for b_id in common:
            b_before = before[b_id]
            b_after = after[b_id]
            field_diffs: dict[str, dict[str, Any]] = {}

            all_fields = set(b_before.keys()) | set(b_after.keys())
            for field in all_fields:
                val_before = b_before.get(field)
                val_after = b_after.get(field)
                if val_before != val_after:
                    field_diffs[field] = {"before": val_before, "after": val_after}

            if field_diffs:
                modified[b_id] = field_diffs

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "has_changes": bool(added or removed or modified),
        }
