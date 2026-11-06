"""Pydantic models for per-trajectory score breakdowns and rule violations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "major", "minor"]


class PolicyViolation(BaseModel):
    """A structured policy violation with rule metadata."""

    rule_id: str = Field(..., description="Stable policy rule identifier.")
    severity: Severity = Field(..., description="Rule severity level.")
    detail: str = Field(default="", description="Human-readable explanation of the violation.")


class ScoreBreakdown(BaseModel):
    """Per-trajectory deterministic and policy scoring summary."""

    passed: bool = Field(default=False, description="Whether the trajectory passes all checks.")
    tool_selection_ok: bool = Field(default=True, description="Expected tool calls were observed.")
    args_valid: bool = Field(default=True, description="Tool arguments match the supported schema.")
    db_state_matches: bool = Field(
        default=True, description="Final DB state matches the scenario expectation."
    )
    forbidden_action_violations: list[str] = Field(
        default_factory=list,
        description="Forbidden tool names that were invoked.",
    )
    policy_violations: list[PolicyViolation] = Field(
        default_factory=list,
        description="Structured policy violations detected from the trajectory.",
    )
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reason strings for all failures.",
    )

    def add_failure(self, reason: str) -> None:
        """Append a failure string if it is not already present."""
        if reason not in self.failure_reasons:
            self.failure_reasons.append(reason)

    def finalize(self) -> None:
        """Compute the final pass/fail state from the collected evidence."""
        self.passed = (
            self.tool_selection_ok
            and self.args_valid
            and self.db_state_matches
            and not self.forbidden_action_violations
            and not self.policy_violations
        )
