"""Scenario suite loader and validation utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from swapcheck.scenarios.models import Scenario


def load_scenario_file(path: str | Path) -> Scenario:
    """Load one scenario YAML file and validate it with the Pydantic model."""
    return Scenario.from_yaml_file(path)


def load_scenarios(path: str | Path) -> dict[str, list[Scenario]]:
    """Load all scenario files in a directory and group them by category."""
    suite_path = Path(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Scenario suite directory not found: {suite_path}")

    yaml_files = sorted({*suite_path.rglob("*.yaml"), *suite_path.rglob("*.yml")})
    if not yaml_files:
        raise ValueError(f"No scenario YAML files found under '{suite_path}'.")

    seen_ids: set[str] = set()
    grouped: dict[str, list[Scenario]] = defaultdict(list)

    for yaml_path in yaml_files:
        try:
            scenario = load_scenario_file(yaml_path)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid scenario file '{yaml_path}': {exc}") from exc

        if scenario.id in seen_ids:
            raise ValueError(f"Duplicate scenario id '{scenario.id}' in '{yaml_path}'.")
        seen_ids.add(scenario.id)
        grouped[scenario.category].append(scenario)

    return dict(sorted(grouped.items()))
