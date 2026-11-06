from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from .models import BootstrapResult, ScenarioResult


def _scenario_to_rates(row: ScenarioResult | Mapping[str, Any]) -> tuple[float, float]:
    if isinstance(row, ScenarioResult):
        return float(row.baseline_pass_rate), float(row.candidate_pass_rate)
    if "baseline_pass_rate" in row and "candidate_pass_rate" in row:
        return float(row["baseline_pass_rate"]), float(row["candidate_pass_rate"])
    if "baseline" in row and "candidate" in row:
        return float(row["baseline"]), float(row["candidate"])
    raise ValueError(f"Scenario row is missing baseline/candidate pass rates: {row!r}")


def compute_scenario_differences(
    baseline_rates: Sequence[float] | Sequence[ScenarioResult | Mapping[str, Any]],
    candidate_rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Return paired scenario deltas as candidate_rate - baseline_rate."""
    if candidate_rates is not None:
        baseline_array = np.asarray(baseline_rates, dtype=float)
        candidate_array = np.asarray(candidate_rates, dtype=float)
        if baseline_array.shape != candidate_array.shape:
            raise ValueError("Baseline and candidate arrays must have the same length.")
        return cast(np.ndarray, candidate_array - baseline_array)

    rows = list(baseline_rates)
    if not rows:
        raise ValueError("No scenario rows provided.")
    diffs: list[float] = []
    for row in rows:
        if isinstance(row, (float, int)):
            raise TypeError("Scenario rows must be mapping-like or ScenarioResult objects.")
        baseline, candidate = _scenario_to_rates(row)
        diffs.append(float(candidate - baseline))
    return cast(np.ndarray, np.asarray(diffs, dtype=float))


def paired_scenario_bootstrap(
    baseline_rates: Sequence[float] | Sequence[ScenarioResult | Mapping[str, Any]],
    candidate_rates: Sequence[float] | None = None,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the 95% percentile CI of the mean paired scenario pass-rate delta."""
    if resamples <= 0:
        raise ValueError("resamples must be positive.")

    deltas = compute_scenario_differences(baseline_rates, candidate_rates)
    if deltas.size == 0:
        raise ValueError("At least one scenario is required.")

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, deltas.size, size=(resamples, deltas.size))
    bootstrap_means = deltas[sample_idx].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])

    overall = BootstrapResult(
        estimate=float(np.mean(deltas)),
        lower=float(lower),
        upper=float(upper),
        resamples=resamples,
    )

    if candidate_rates is not None:
        return {"overall": overall, "categories": {}}

    categories: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in baseline_rates:
        if isinstance(row, (float, int)):
            raise TypeError("Scenario rows must be mapping-like or ScenarioResult objects.")
        baseline, candidate = _scenario_to_rates(row)
        category = (
            row.get("category", "uncategorized") if isinstance(row, Mapping) else row.category
        )
        categories[str(category)].append((baseline, candidate))

    category_summary: dict[str, BootstrapResult] = {}
    for cat_index, (name, items) in enumerate(sorted(categories.items())):
        baseline_vals = np.asarray([item[0] for item in items], dtype=float)
        candidate_vals = np.asarray([item[1] for item in items], dtype=float)
        diffs_for_cat = candidate_vals - baseline_vals
        cat_rng = np.random.default_rng(seed + cat_index * 997)
        cat_idx = cat_rng.integers(0, diffs_for_cat.size, size=(resamples, diffs_for_cat.size))
        cat_bootstrap = diffs_for_cat[cat_idx].mean(axis=1)
        lower_cat, upper_cat = np.percentile(cat_bootstrap, [2.5, 97.5])
        category_summary[name] = BootstrapResult(
            estimate=float(np.mean(diffs_for_cat)),
            lower=float(lower_cat),
            upper=float(upper_cat),
            resamples=resamples,
        )

    return {"overall": overall, "categories": category_summary}


def paired_bootstrap_by_category(
    scenario_rows: Sequence[ScenarioResult | Mapping[str, Any]],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, BootstrapResult]:
    """Compute the category-level paired bootstrap CI for scenario deltas."""
    rows_by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in scenario_rows:
        baseline, candidate = _scenario_to_rates(row)
        category = (
            row.get("category", "uncategorized") if isinstance(row, Mapping) else row.category
        )
        rows_by_category[str(category)].append((baseline, candidate))

    summary: dict[str, BootstrapResult] = {}
    for category_index, (name, values) in enumerate(sorted(rows_by_category.items())):
        baseline_vals = np.asarray([item[0] for item in values], dtype=float)
        candidate_vals = np.asarray([item[1] for item in values], dtype=float)
        diffs = candidate_vals - baseline_vals
        rng = np.random.default_rng(seed + category_index * 997)
        indices = rng.integers(0, diffs.size, size=(resamples, diffs.size))
        bootstrap_means = diffs[indices].mean(axis=1)
        lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
        summary[name] = BootstrapResult(
            estimate=float(np.mean(diffs)),
            lower=float(lower),
            upper=float(upper),
            resamples=resamples,
        )
    return summary
