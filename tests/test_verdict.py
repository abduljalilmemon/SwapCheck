from __future__ import annotations

from swapcheck.verdict.engine import VerdictEngine
from swapcheck.verdict.models import ScenarioResult


def _scenario_rows(
    *,
    baseline: float,
    candidate: float,
    category: str = "booking",
    count: int = 60,
    baseline_violations: int = 0,
    candidate_violations: int = 0,
    new_critical_policy_violations: int = 0,
) -> list[ScenarioResult]:
    rows: list[ScenarioResult] = []
    for idx in range(count):
        rows.append(
            ScenarioResult(
                scenario_id=f"{category}-{idx}",
                category=category,
                baseline_pass_rate=baseline,
                candidate_pass_rate=candidate,
                baseline_repetitions=3,
                candidate_repetitions=3,
                baseline_cost=1.0,
                candidate_cost=1.0,
                baseline_successes=int(round(baseline * 3)),
                candidate_successes=int(round(candidate * 3)),
                baseline_latency_ms=[100.0, 110.0, 120.0],
                candidate_latency_ms=[105.0, 115.0, 125.0],
                baseline_prompt_tokens=300,
                baseline_completion_tokens=150,
                candidate_prompt_tokens=300,
                candidate_completion_tokens=150,
                baseline_critical_policy_violations=baseline_violations,
                candidate_critical_policy_violations=candidate_violations,
                new_critical_policy_violations=new_critical_policy_violations,
            )
        )
    return rows


def test_ship_for_identical_models() -> None:
    rows = _scenario_rows(baseline=0.90, candidate=0.90, category="booking", count=60)

    result = VerdictEngine().evaluate(rows)

    assert result.verdict == "SHIP"
    assert result.reasons


def test_hold_for_clearly_worse_candidate() -> None:
    rows = _scenario_rows(baseline=0.90, candidate=0.30, category="booking", count=60)

    result = VerdictEngine().evaluate(rows)

    assert result.verdict == "HOLD"
    assert any("overall" in reason.lower() for reason in result.reasons)


def test_inconclusive_for_noisy_borderline_data() -> None:
    rows = _scenario_rows(baseline=0.52, candidate=0.50, category="booking", count=30)

    result = VerdictEngine().evaluate(rows)

    assert result.verdict == "INCONCLUSIVE"
    assert result.required_sample_size is not None


def test_hold_when_new_critical_policy_violation_is_detected() -> None:
    rows = _scenario_rows(
        baseline=0.85,
        candidate=0.85,
        category="booking",
        count=60,
        new_critical_policy_violations=1,
    )

    result = VerdictEngine().evaluate(rows)

    assert result.verdict == "HOLD"
    assert any("critical policy violation" in reason.lower() for reason in result.reasons)


def test_underpowered_category_is_flagged_without_forcing_hold() -> None:
    rows = [
        * _scenario_rows(baseline=0.90, candidate=0.92, category="booking", count=60),
        * _scenario_rows(baseline=0.72, candidate=0.74, category="refunds", count=3),
    ]

    result = VerdictEngine(min_scenarios=50, min_scenarios_per_category=8).evaluate(rows)

    assert result.verdict == "SHIP"
    assert "refunds" in result.underpowered_categories


def test_verdict_reports_cost_latency_and_tokens() -> None:
    rows = _scenario_rows(baseline=0.90, candidate=0.90, category="booking", count=60)

    result = VerdictEngine().evaluate(rows)

    assert "baseline" in result.cost_summary
    assert "candidate" in result.cost_summary
    assert "baseline" in result.latency_summary
    assert "candidate" in result.latency_summary
    assert "baseline" in result.token_totals
    assert "candidate" in result.token_totals
