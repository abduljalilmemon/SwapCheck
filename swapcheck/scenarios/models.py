"""Scenario schema and YAML parsing for the SwapCheck evaluation suite."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Scenario(BaseModel):
    """Definition of a single booking-support evaluation scenario."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique scenario identifier")
    category: str = Field(..., description="Scenario category, e.g. simple_lookup")
    description: str = Field(..., description="Human-readable scenario description")
    initial_db_state: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Initial booking database snapshot for the scenario.",
    )
    user_turns: list[str] = Field(
        default_factory=list,
        description="Scripted user turns to play in sequence.",
    )
    expected_final_db_state: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Expected database state after the run completes.",
    )
    forbidden_tool_calls: list[str] = Field(
        default_factory=list,
        description="Tool names that must never appear in the run.",
    )
    expected_tool_calls: list[str] | None = Field(
        default=None,
        description="Optional list of tool names expected to appear at least once.",
    )
    current_time: datetime | None = Field(
        default=None,
        description=(
            "Simulated current time (UTC) for this scenario. It is shown to the agent in its "
            "system prompt and used by the policy checker for refund-tier calculations, so "
            "results never depend on the wall clock. If omitted, refund-tier checks are skipped."
        ),
    )
    rubric_notes: str | None = Field(default=None, description="Optional rubric guidance")
    tags: list[str] = Field(default_factory=list, description="Classification tags")

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> Scenario:
        """Load and validate a scenario from a YAML file."""
        scenario_path = Path(path)
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        raw_data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        if raw_data is None:
            raise ValueError(f"Scenario file '{scenario_path}' is empty.")
        if not isinstance(raw_data, dict):
            raise ValueError(
                f"Scenario file '{scenario_path}' must contain a mapping at the top level."
            )
        return cls.model_validate(raw_data)

    @classmethod
    def from_yaml_text(cls, text: str) -> Scenario:
        """Load and validate a scenario from YAML text."""
        raw_data = yaml.safe_load(text)
        if raw_data is None:
            raise ValueError("Scenario YAML text is empty.")
        if not isinstance(raw_data, dict):
            raise ValueError("Scenario YAML text must contain a mapping at the top level.")
        return cls.model_validate(raw_data)

    def initial_db_mapping(self) -> dict[str, dict[str, Any]]:
        """Normalize initial state to a dict keyed by booking ID."""
        return self._normalize_db_state(self.initial_db_state)

    def expected_final_db_mapping(self) -> dict[str, dict[str, Any]]:
        """Normalize expected final state to a dict keyed by booking ID."""
        return self._normalize_db_state(self.expected_final_db_state)

    @staticmethod
    def _normalize_db_state(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each booking entry in a scenario must be a mapping.")
            booking_id = str(entry.get("id"))
            if not booking_id:
                raise ValueError("Each booking entry must include a non-empty 'id'.")
            normalized[booking_id] = entry
        return normalized
