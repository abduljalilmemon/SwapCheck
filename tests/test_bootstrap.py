from __future__ import annotations

import numpy as np

from swapcheck.verdict.bootstrap import paired_scenario_bootstrap


def test_bootstrap_reproduces_known_difference_in_most_datasets() -> None:
    rng = np.random.default_rng(12345)
    covered = 0

    for idx in range(200):
        true_delta = 0.12
        baseline = rng.random(60)
        candidate = np.clip(baseline + true_delta + rng.normal(0.0, 0.10, 60), 0.0, 1.0)
        result = paired_scenario_bootstrap(baseline, candidate, resamples=2000, seed=idx)
        ci = result["overall"]
        if ci[0] <= true_delta <= ci[1]:
            covered += 1

    assert covered >= 150


def test_bootstrap_by_category_returns_category_interval() -> None:
    scenarios = [
        {"scenario_id": "s1", "category": "refunds", "baseline_pass_rate": 0.7, "candidate_pass_rate": 0.8},
        {"scenario_id": "s2", "category": "refunds", "baseline_pass_rate": 0.6, "candidate_pass_rate": 0.9},
        {"scenario_id": "s3", "category": "policy", "baseline_pass_rate": 0.5, "candidate_pass_rate": 0.55},
        {"scenario_id": "s4", "category": "policy", "baseline_pass_rate": 0.4, "candidate_pass_rate": 0.5},
    ]

    result = paired_scenario_bootstrap(scenarios, resamples=1000, seed=4)

    assert "overall" in result
    assert "categories" in result
    assert "refunds" in result["categories"]
    assert "policy" in result["categories"]
    assert result["categories"]["refunds"].estimate > 0.0
