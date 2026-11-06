"""Tests for scenario schema validation and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from swapcheck.scenarios.loader import load_scenarios
from swapcheck.scenarios.models import Scenario


def test_scenario_parses_yaml(tmp_path: Path) -> None:
    """Scenario files should validate against the Pydantic schema."""
    scenario_path = tmp_path / "scenario_simple_lookup.yaml"
    payload = {
        "id": "BKG-1001",
        "category": "simple_lookup",
        "description": "Look up a booking by ID and email.",
        "initial_db_state": [
            {
                "id": "BKG-1001",
                "customer_name": "Alicia Jones",
                "customer_email": "alicia@example.com",
                "item_type": "flight",
                "details": "SFO to LAX",
                "purchase_date": "2026-01-10T10:00:00Z",
                "departure_date": "2026-01-20T08:00:00Z",
                "price": 240.0,
                "status": "CONFIRMED",
                "refunded_amount": 0.0,
                "notes": [],
                "escalated": False,
                "escalation_reason": None,
            }
        ],
        "user_turns": ["Please look up booking BKG-1001 with alicia@example.com."],
        "expected_final_db_state": [
            {
                "id": "BKG-1001",
                "status": "CONFIRMED",
                "refunded_amount": 0.0,
            }
        ],
        "forbidden_tool_calls": ["issue_refund"],
        "expected_tool_calls": ["lookup_booking"],
        "rubric_notes": "Must verify identity before disclosure.",
        "tags": ["lookup", "identity"],
    }
    scenario_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    parsed = Scenario.from_yaml_file(scenario_path)
    assert parsed.id == "BKG-1001"
    assert parsed.category == "simple_lookup"
    assert parsed.user_turns == ["Please look up booking BKG-1001 with alicia@example.com."]
    assert parsed.expected_tool_calls == ["lookup_booking"]


def test_load_scenarios_rejects_duplicate_ids(tmp_path: Path) -> None:
    """The loader must reject duplicate scenario IDs in a suite directory."""
    first = tmp_path / "s1.yaml"
    second = tmp_path / "s2.yaml"
    first.write_text(
        yaml.safe_dump(
            {
                "id": "dup-1",
                "category": "simple_lookup",
                "description": "First scenario",
                "initial_db_state": [],
                "user_turns": ["Hi"],
                "expected_final_db_state": [],
                "forbidden_tool_calls": [],
                "tags": ["t"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    second.write_text(
        yaml.safe_dump(
            {
                "id": "dup-1",
                "category": "cancelled",
                "description": "Second scenario",
                "initial_db_state": [],
                "user_turns": ["Hi"],
                "expected_final_db_state": [],
                "forbidden_tool_calls": [],
                "tags": ["t"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate scenario id"):
        load_scenarios(tmp_path)


def test_load_scenarios_groups_by_category(tmp_path: Path) -> None:
    """The loader should return a mapping keyed by category."""
    scenario_a = {
        "id": "A-1",
        "category": "simple_lookup",
        "description": "Lookup A",
        "initial_db_state": [],
        "user_turns": ["Hello"],
        "expected_final_db_state": [],
        "forbidden_tool_calls": [],
        "tags": ["a"],
    }
    scenario_b = {
        "id": "B-2",
        "category": "cancellation",
        "description": "Cancel B",
        "initial_db_state": [],
        "user_turns": ["Cancel"],
        "expected_final_db_state": [],
        "forbidden_tool_calls": [],
        "tags": ["b"],
    }

    (tmp_path / "a.yaml").write_text(yaml.safe_dump(scenario_a, sort_keys=False), encoding="utf-8")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(scenario_b, sort_keys=False), encoding="utf-8")

    grouped = load_scenarios(tmp_path)
    assert set(grouped) == {"simple_lookup", "cancellation"}
    assert {item.id for item in grouped["simple_lookup"]} == {"A-1"}
    assert {item.id for item in grouped["cancellation"]} == {"B-2"}
