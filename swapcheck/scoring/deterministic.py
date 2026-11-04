"""Deterministic scoring checks for scenario trajectories."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.scoring.models import ScoreBreakdown
from swapcheck.scoring.policy_checker import PolicyViolationDetector


class DeterministicScorer:
    """Checks scenario adherence via tool call validity, DB state comparison, and forbidden operations."""

    _ARGUMENT_REQUIREMENTS: dict[str, set[str]] = {
        "lookup_booking": {"booking_id"},
        "cancel_booking": {"booking_id"},
        "issue_refund": {"booking_id", "amount"},
        "escalate_to_human": {"booking_id", "reason"},
    }

    def score(self, trajectory: Trajectory, scenario: Scenario) -> ScoreBreakdown:
        """Score a single trajectory against a scenario."""
        breakdown = ScoreBreakdown()

        breakdown.tool_selection_ok = self._check_tool_selection(trajectory, scenario, breakdown)
        breakdown.args_valid = self._check_tool_arguments(trajectory, breakdown)
        breakdown.db_state_matches = self._check_db_state(trajectory, scenario, breakdown)
        breakdown.forbidden_action_violations = self._check_forbidden_actions(
            trajectory, scenario, breakdown
        )

        detector = PolicyViolationDetector()
        breakdown.policy_violations = detector.detect(trajectory)
        for violation in breakdown.policy_violations:
            breakdown.add_failure(
                f"[{violation.rule_id}] {violation.detail or 'Policy violation detected.'}"
            )

        breakdown.finalize()
        return breakdown

    def _check_tool_selection(
        self, trajectory: Trajectory, scenario: Scenario, breakdown: ScoreBreakdown
    ) -> bool:
        expected = scenario.expected_tool_calls or []
        if not expected:
            return True

        observed = {call["name"] for call in self._iter_tool_calls(trajectory)}
        missing = [name for name in expected if name not in observed]
        if missing:
            breakdown.add_failure(f"Missing expected tool calls: {', '.join(missing)}")
            return False
        return True

    def _check_tool_arguments(self, trajectory: Trajectory, breakdown: ScoreBreakdown) -> bool:
        valid = True
        for call in self._iter_tool_calls(trajectory):
            name = str(call.get("name", ""))
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                valid = False
                breakdown.add_failure(f"Tool call {name!r} has invalid arguments payload.")
                continue

            required = self._ARGUMENT_REQUIREMENTS.get(name)
            if required is None:
                valid = False
                breakdown.add_failure(f"Unsupported tool call {name!r} encountered.")
                continue

            missing = [key for key in sorted(required) if key not in arguments]
            if missing:
                valid = False
                breakdown.add_failure(
                    f"Tool call {name!r} missing required arguments: {', '.join(missing)}."
                )
                continue

            if name == "issue_refund":
                amount_value = arguments.get("amount")
                if not isinstance(amount_value, (int, float)) or float(amount_value) < 0:
                    valid = False
                    breakdown.add_failure(
                        f"Tool call {name!r} has an invalid refund amount: {amount_value!r}."
                    )

            if name == "lookup_booking" and "customer_email" in arguments:
                email_value = arguments.get("customer_email")
                if not isinstance(email_value, str):
                    valid = False
                    breakdown.add_failure(
                        f"Tool call {name!r} has a non-string customer_email: {email_value!r}."
                    )

        return valid

    def _check_db_state(
        self, trajectory: Trajectory, scenario: Scenario, breakdown: ScoreBreakdown
    ) -> bool:
        expected_entries = scenario.expected_final_db_state
        if not expected_entries:
            return True

        expected_map = self._normalize_db_rows(expected_entries)
        actual_map = self._normalize_db_rows(list(trajectory.final_db_snapshot.values()))

        diffs: list[str] = []
        for booking_id in sorted(set(expected_map) | set(actual_map)):
            expected_obj = expected_map.get(booking_id)
            actual_obj = actual_map.get(booking_id)
            if expected_obj is None:
                diffs.append(f"Unexpected booking ID {booking_id!r} present in final DB state.")
                continue
            if actual_obj is None:
                diffs.append(f"Expected booking ID {booking_id!r} is missing from final DB state.")
                continue

            for field in sorted(expected_obj):
                if expected_obj.get(field) != actual_obj.get(field):
                    diffs.append(
                        f"DB mismatch for {booking_id!r}: {field} expected "
                        f"{expected_obj.get(field)!r}, got {actual_obj.get(field)!r}."
                    )

        if diffs:
            for diff in diffs:
                breakdown.add_failure(diff)
            return False
        return True

    def _check_forbidden_actions(
        self, trajectory: Trajectory, scenario: Scenario, breakdown: ScoreBreakdown
    ) -> list[str]:
        forbidden = scenario.forbidden_tool_calls
        if not forbidden:
            return []

        observed = [call["name"] for call in self._iter_tool_calls(trajectory)]
        violations = sorted({name for name in forbidden if name in observed})
        if violations:
            breakdown.add_failure(
                f"Forbidden tool call(s) used: {', '.join(violations)}."
            )
        return violations

    @staticmethod
    def _normalize_db_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            booking_id = row.get("id")
            if booking_id is None:
                continue
            normalized[str(booking_id)] = dict(row)
        return normalized

    @staticmethod
    def _iter_tool_calls(trajectory: Trajectory) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for turn in trajectory.turns:
            for call in turn.tool_calls:
                if isinstance(call, dict):
                    calls.append(call)
        return calls
