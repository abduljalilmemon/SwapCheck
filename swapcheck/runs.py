"""Stored evaluation runs: trajectory records, manifest, summaries and judge metrics.

A run directory looks like::

    <output_dir>/
        manifest.json                  # configs, CLI settings, verdict, trajectory index
        trajectories/<id>.json         # one TrajectoryRecord per trajectory
        report.md / report.json        # rendered reports

``swapcheck label`` and ``swapcheck calibrate`` work from this directory.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.scoring.models import ScoreBreakdown
from swapcheck.verdict.models import ScenarioResult

ROLES = ("baseline", "candidate")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


class TrajectoryRecord(BaseModel):
    """One scored trajectory, as stored on disk."""

    trajectory_id: str
    role: str = Field(..., description="'baseline' or 'candidate'")
    config_name: str
    scenario_id: str
    category: str
    repetition_index: int
    trajectory: Trajectory
    score: ScoreBreakdown
    judge_result: dict[str, Any] | None = None
    passed: bool = Field(
        ...,
        description="Pass decision used for the verdict (deterministic, plus judge if enabled).",
    )


def make_trajectory_id(scenario_id: str, role: str, repetition_index: int) -> str:
    """Stable, filesystem-safe trajectory id."""
    return f"{_UNSAFE_CHARS.sub('_', scenario_id)}__{role}__rep{repetition_index}"


def write_run(
    output_dir: Path,
    records: Sequence[TrajectoryRecord],
    manifest: dict[str, Any],
) -> None:
    """Write trajectories and the manifest into ``output_dir``."""
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        path = trajectories_dir / f"{record.trajectory_id}.json"
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    manifest = dict(manifest)
    manifest["trajectories"] = [
        {
            "trajectory_id": record.trajectory_id,
            "role": record.role,
            "scenario_id": record.scenario_id,
            "category": record.category,
            "repetition_index": record.repetition_index,
            "passed": record.passed,
            "judge_result": record.judge_result,
        }
        for record in records
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json found in run directory: {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest.json in {run_dir} is not a JSON object.")
    return data


def load_records(run_dir: Path) -> list[TrajectoryRecord]:
    """Load all stored trajectory records from a run directory."""
    trajectories_dir = run_dir / "trajectories"
    if not trajectories_dir.is_dir():
        raise FileNotFoundError(f"No trajectories directory found in run directory: {run_dir}")
    records = [
        TrajectoryRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(trajectories_dir.glob("*.json"))
    ]
    if not records:
        raise ValueError(f"No trajectory files found in {trajectories_dir}.")
    return records


def decide_pass(
    score: ScoreBreakdown,
    judge_result: dict[str, Any] | None,
    *,
    judge_affects_pass: bool,
) -> bool:
    """Pass decision: deterministic and policy checks, plus the judge only if enabled.

    A judge error never counts as a pass when the judge is allowed to affect the verdict.
    """
    if not score.passed:
        return False
    if not judge_affects_pass:
        return True
    if judge_result is None or judge_result.get("judge_error"):
        return False
    return bool(judge_result.get("passed"))


def _critical_count(record: TrajectoryRecord) -> int:
    return sum(
        1 for violation in record.score.policy_violations if violation.severity == "critical"
    )


def build_scenario_results(records: Iterable[TrajectoryRecord]) -> list[ScenarioResult]:
    """Aggregate trajectory records into one paired ScenarioResult per scenario."""
    grouped: dict[str, dict[str, list[TrajectoryRecord]]] = defaultdict(lambda: defaultdict(list))
    categories: dict[str, str] = {}
    for record in records:
        if record.trajectory.budget_exceeded:
            continue
        grouped[record.scenario_id][record.role].append(record)
        categories[record.scenario_id] = record.category

    results: list[ScenarioResult] = []
    for scenario_id in sorted(grouped):
        by_role = grouped[scenario_id]
        baseline = by_role.get("baseline", [])
        candidate = by_role.get("candidate", [])
        if not baseline or not candidate:
            # A scenario that was not completed under both configs cannot be paired.
            continue
        baseline_pass = sum(1 for r in baseline if r.passed)
        candidate_pass = sum(1 for r in candidate if r.passed)
        baseline_crit = sum(_critical_count(r) for r in baseline)
        candidate_crit = sum(_critical_count(r) for r in candidate)
        results.append(
            ScenarioResult(
                scenario_id=scenario_id,
                category=categories[scenario_id],
                baseline_pass_rate=baseline_pass / len(baseline),
                candidate_pass_rate=candidate_pass / len(candidate),
                baseline_repetitions=len(baseline),
                candidate_repetitions=len(candidate),
                baseline_cost=sum(r.trajectory.total_cost for r in baseline),
                candidate_cost=sum(r.trajectory.total_cost for r in candidate),
                baseline_successes=baseline_pass,
                candidate_successes=candidate_pass,
                baseline_latency_ms=[r.trajectory.total_latency_ms for r in baseline],
                candidate_latency_ms=[r.trajectory.total_latency_ms for r in candidate],
                baseline_prompt_tokens=sum(r.trajectory.prompt_tokens for r in baseline),
                baseline_completion_tokens=sum(r.trajectory.completion_tokens for r in baseline),
                candidate_prompt_tokens=sum(r.trajectory.prompt_tokens for r in candidate),
                candidate_completion_tokens=sum(r.trajectory.completion_tokens for r in candidate),
                baseline_critical_policy_violations=baseline_crit,
                candidate_critical_policy_violations=candidate_crit,
                # A "new" violation is a critical violation in a candidate trajectory of a
                # scenario where the baseline had none (counted per scenario, never negative).
                new_critical_policy_violations=(
                    candidate_crit if candidate_crit > 0 and baseline_crit == 0 else 0
                ),
            )
        )
    return results


def judge_metrics(records: Sequence[TrajectoryRecord]) -> dict[str, Any] | None:
    """Aggregate judge scores per config and category. Judge errors are excluded and counted."""
    judged = [r for r in records if r.judge_result is not None]
    if not judged:
        return None

    models = sorted({str(r.judge_result.get("model", "")) for r in judged if r.judge_result})
    total_cost = sum(float((r.judge_result or {}).get("cost", 0.0)) for r in judged)
    total_errors = sum(1 for r in judged if (r.judge_result or {}).get("judge_error"))

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: dict[tuple[str, str], int] = defaultdict(int)
    for record in judged:
        result = record.judge_result or {}
        for category in (record.category, "ALL"):
            key = (record.role, category)
            if result.get("judge_error"):
                errors[key] += 1
            else:
                buckets[key].append(result)

    rows: list[dict[str, Any]] = []
    for role, category in sorted(set(buckets) | set(errors)):
        items = buckets.get((role, category), [])
        n = len(items)
        rows.append(
            {
                "config": role,
                "category": category,
                "helpfulness": float(np.mean([i["helpfulness"] for i in items])) if n else 0.0,
                "groundedness": float(np.mean([i["groundedness"] for i in items])) if n else 0.0,
                "tone": float(np.mean([i["tone"] for i in items])) if n else 0.0,
                "off_policy_rate": (
                    sum(1 for i in items if i.get("off_policy_promises")) / n if n else 0.0
                ),
                "judge_pass_rate": sum(1 for i in items if i.get("passed")) / n if n else 0.0,
                "errors": errors.get((role, category), 0),
            }
        )
    return {
        "model": ", ".join(m for m in models if m),
        "total_cost": float(total_cost),
        "total_errors": int(total_errors),
        "rows": rows,
    }


def build_summary(
    records: Sequence[TrajectoryRecord],
    results: Sequence[ScenarioResult],
    *,
    worst_n: int = 3,
) -> dict[str, Any]:
    """Pass rates (overall and per category) and the worst failing candidate trajectories."""
    summary: dict[str, Any] = {"pass_rates": {}, "category_pass_rates": {}, "worst_failures": []}
    for role in ROLES:
        rates = [getattr(r, f"{role}_pass_rate") for r in results]
        summary["pass_rates"][role] = float(np.mean(rates)) if rates else 0.0

    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in results:
        per_category[row.category]["baseline"].append(row.baseline_pass_rate)
        per_category[row.category]["candidate"].append(row.candidate_pass_rate)
    summary["category_pass_rates"] = {
        category: {role: float(np.mean(values)) for role, values in roles.items()}
        for category, roles in sorted(per_category.items())
    }

    failures = [r for r in records if r.role == "candidate" and not r.passed]
    failures.sort(key=lambda r: (-_critical_count(r), r.scenario_id, r.repetition_index))
    summary["worst_failures"] = [
        {
            "trajectory_id": r.trajectory_id,
            "scenario_id": r.scenario_id,
            "category": r.category,
            "critical_violations": _critical_count(r),
            "reasons": list(r.score.failure_reasons)
            or (["Judge rated the trajectory as failing."] if r.judge_result else []),
        }
        for r in failures[:worst_n]
    ]
    return summary


def verdict_to_dict(verdict: Any) -> dict[str, Any]:
    """Serialize a VerdictResult dataclass to plain JSON-friendly data."""
    from dataclasses import asdict

    return asdict(verdict)


def verdict_from_dict(data: dict[str, Any]) -> Any:
    """Rebuild a VerdictResult from the dictionary stored in manifest.json."""
    from swapcheck.verdict.models import CategoryResult, VerdictResult

    category_results = {
        name: CategoryResult(
            category=str(item["category"]),
            scenario_count=int(item["scenario_count"]),
            estimate=float(item["estimate"]),
            ci=(float(item["ci"][0]), float(item["ci"][1])),
            powered=bool(item["powered"]),
            status=item["status"],
            reasons=list(item.get("reasons", [])),
            blockers=list(item.get("blockers", [])),
            notes=list(item.get("notes", [])),
        )
        for name, item in data.get("category_results", {}).items()
    }
    return VerdictResult(
        verdict=data["verdict"],
        reasons=list(data.get("reasons", [])),
        overall_delta=float(data["overall_delta"]),
        overall_ci=(float(data["overall_ci"][0]), float(data["overall_ci"][1])),
        required_sample_size=data.get("required_sample_size"),
        underpowered_categories=list(data.get("underpowered_categories", [])),
        category_results=category_results,
        cost_summary=data.get("cost_summary", {}),
        latency_summary=data.get("latency_summary", {}),
        token_totals=data.get("token_totals", {}),
        bootstrap_resamples=int(data.get("bootstrap_resamples", 2000)),
        metadata=dict(data.get("metadata", {})),
        blockers=list(data.get("blockers", [])),
        notes=list(data.get("notes", [])),
    )
