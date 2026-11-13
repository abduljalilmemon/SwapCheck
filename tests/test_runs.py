"""Unit tests for stored-run helpers: ids, pass decisions, aggregation and judge metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swapcheck.providers.base import ChatMessage
from swapcheck.providers.scripted_provider import ScriptedMockProvider
from swapcheck.runs import (
    TrajectoryRecord,
    build_scenario_results,
    build_summary,
    decide_pass,
    judge_metrics,
    load_manifest,
    load_records,
    make_trajectory_id,
    verdict_from_dict,
    verdict_to_dict,
    write_run,
)
from swapcheck.scenarios.trajectory import Trajectory, TurnRecord
from swapcheck.scoring.models import PolicyViolation, ScoreBreakdown
from swapcheck.verdict.engine import VerdictEngine


def _score(passed: bool = True, *, critical: int = 0, minor: int = 0) -> ScoreBreakdown:
    violations = [PolicyViolation(rule_id="r", severity="critical") for _ in range(critical)]
    violations += [PolicyViolation(rule_id="needs_judge", severity="minor") for _ in range(minor)]
    breakdown = ScoreBreakdown(policy_violations=violations)
    breakdown.finalize()
    if not passed:
        breakdown.passed = False
    return breakdown


def _record(
    scenario: str,
    role: str,
    rep: int,
    *,
    passed: bool = True,
    critical: int = 0,
    judge: dict[str, Any] | None = None,
    category: str = "cancellation",
    cost: float = 0.01,
) -> TrajectoryRecord:
    trajectory = Trajectory(
        scenario_id=scenario,
        category=category,
        repetition_index=rep,
        turns=[TurnRecord(turn_index=0, user_turn="hi", assistant_text="hello")],
        total_cost=cost,
        total_latency_ms=100.0,
        prompt_tokens=30,
        completion_tokens=10,
        total_tokens=40,
    )
    return TrajectoryRecord(
        trajectory_id=make_trajectory_id(scenario, role, rep),
        role=role,
        config_name=role,
        scenario_id=scenario,
        category=category,
        repetition_index=rep,
        trajectory=trajectory,
        score=_score(passed, critical=critical),
        judge_result=judge,
        passed=passed,
    )


def test_trajectory_ids_are_stable_and_filesystem_safe() -> None:
    assert make_trajectory_id("BKG-101", "baseline", 2) == "BKG-101__baseline__rep2"
    unsafe = make_trajectory_id("a/b c:d", "candidate", 0)
    assert "/" not in unsafe and " " not in unsafe and ":" not in unsafe


def test_minor_violations_do_not_fail_a_trajectory() -> None:
    assert _score(minor=2).passed is True
    assert _score(critical=1).passed is False


def test_decide_pass_ignores_judge_unless_enabled() -> None:
    good = _score()
    assert decide_pass(good, None, judge_affects_pass=False) is True
    assert decide_pass(_score(passed=False), {"passed": True}, judge_affects_pass=False) is False


def test_decide_pass_with_judge_enabled_never_treats_errors_as_pass() -> None:
    good = _score()
    assert decide_pass(good, {"passed": True}, judge_affects_pass=True) is True
    assert decide_pass(good, {"passed": False}, judge_affects_pass=True) is False
    assert (
        decide_pass(good, {"passed": True, "judge_error": "boom"}, judge_affects_pass=True) is False
    )
    assert decide_pass(good, None, judge_affects_pass=True) is False


def test_build_scenario_results_pairs_scenarios_and_counts_new_violations() -> None:
    records = [
        _record("s1", "baseline", 0),
        _record("s1", "baseline", 1),
        _record("s1", "candidate", 0, passed=False, critical=1),
        _record("s1", "candidate", 1),
        # s2: the baseline already violates, so the candidate's violation is not "new"
        _record("s2", "baseline", 0, passed=False, critical=1),
        _record("s2", "candidate", 0, passed=False, critical=2),
        # s3 exists only under the baseline and cannot be paired
        _record("s3", "baseline", 0),
    ]
    results = {r.scenario_id: r for r in build_scenario_results(records)}
    assert set(results) == {"s1", "s2"}
    assert results["s1"].baseline_pass_rate == 1.0
    assert results["s1"].candidate_pass_rate == 0.5
    assert results["s1"].new_critical_policy_violations == 1
    assert results["s2"].new_critical_policy_violations == 0
    assert results["s2"].candidate_critical_policy_violations == 2
    assert results["s1"].baseline_prompt_tokens == 60
    assert results["s1"].baseline_completion_tokens == 20
    assert results["s1"].baseline_latency_ms == [100.0, 100.0]


def test_budget_exceeded_trajectories_are_excluded() -> None:
    skipped = _record("s1", "candidate", 0)
    skipped.trajectory.budget_exceeded = True
    results = build_scenario_results([_record("s1", "baseline", 0), skipped])
    assert results == []


def test_judge_metrics_exclude_errors_and_count_them() -> None:
    ok = {
        "helpfulness": 4,
        "groundedness": 5,
        "tone": 4,
        "off_policy_promises": False,
        "passed": True,
        "model": "judge-x",
        "cost": 0.5,
    }
    bad = {"judge_error": "invalid json", "passed": False, "model": "judge-x", "cost": 0.25}
    records = [
        _record("s1", "baseline", 0, judge=ok),
        _record("s1", "baseline", 1, judge=bad),
        _record("s2", "baseline", 0, judge={**ok, "helpfulness": 2, "passed": False}),
    ]
    metrics = judge_metrics(records)
    assert metrics is not None
    assert metrics["total_errors"] == 1
    assert metrics["total_cost"] == pytest.approx(1.25)
    overall = next(r for r in metrics["rows"] if r["category"] == "ALL")
    assert overall["errors"] == 1
    assert overall["helpfulness"] == pytest.approx(3.0)  # the error row is not averaged in
    assert overall["judge_pass_rate"] == pytest.approx(0.5)
    assert judge_metrics([_record("s1", "baseline", 0)]) is None


def test_summary_lists_worst_failures_first_by_critical_violations() -> None:
    records = [
        _record("s1", "baseline", 0),
        _record("s1", "candidate", 0, passed=False, critical=0),
        _record("s2", "baseline", 0),
        _record("s2", "candidate", 0, passed=False, critical=3),
    ]
    summary = build_summary(records, build_scenario_results(records), worst_n=1)
    assert summary["pass_rates"] == {"baseline": 1.0, "candidate": 0.0}
    assert summary["worst_failures"][0]["scenario_id"] == "s2"


def test_run_directory_round_trip_and_verdict_serialisation(tmp_path: Path) -> None:
    records = [_record("s1", "baseline", 0), _record("s1", "candidate", 0)]
    results = build_scenario_results(records)
    verdict = VerdictEngine(bootstrap_resamples=200, min_scenarios=1, min_repetitions=1).evaluate(
        results
    )
    write_run(tmp_path, records, {"verdict": verdict_to_dict(verdict)})

    manifest = load_manifest(tmp_path)
    assert [t["trajectory_id"] for t in manifest["trajectories"]] == [
        "s1__baseline__rep0",
        "s1__candidate__rep0",
    ]
    loaded = load_records(tmp_path)
    assert {r.trajectory_id for r in loaded} == {r.trajectory_id for r in records}

    restored = verdict_from_dict(manifest["verdict"])
    assert restored.verdict == verdict.verdict
    assert restored.overall_ci == verdict.overall_ci
    assert restored.cost_summary == verdict.cost_summary


def test_load_helpers_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path)
    (tmp_path / "trajectories").mkdir()
    with pytest.raises(ValueError):
        load_records(tmp_path)


# ---------------------------------------------------------------------------
# offline mock judge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_judge_flags_promises_and_unhelpful_replies() -> None:
    provider = ScriptedMockProvider(model_name="judge-mock")

    def prompt(assistant_text: str) -> list[ChatMessage]:
        body = (
            "You are an impartial judge for a customer-support agent run.\n\n"
            f"Transcript:\nUser 1: hi\nAssistant 1: {assistant_text}\n\n"
            "Return JSON only in this exact schema:"
        )
        return [ChatMessage(role="user", content=body)]

    import json

    good = json.loads((await provider.complete(prompt("Your booking is cancelled."))).content or "")
    assert good["passed"] is True and good["off_policy_promises"] is False

    promise = json.loads(
        (await provider.complete(prompt("I guarantee a full refund."))).content or ""
    )
    assert promise["off_policy_promises"] is True and promise["passed"] is False

    degraded = json.loads(
        (
            await provider.complete(
                prompt("I apologize, but I am unable to complete this request.")
            )
        ).content
        or ""
    )
    assert degraded["passed"] is False
