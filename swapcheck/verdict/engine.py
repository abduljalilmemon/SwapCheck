from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import ceil
from typing import Literal

import numpy as np

from .bootstrap import paired_bootstrap_by_category, paired_scenario_bootstrap
from .models import CategoryResult, ScenarioResult, VerdictResult


class VerdictEngine:
    """Build a SHIP / HOLD / INCONCLUSIVE verdict from paired scenario pass-rate deltas."""

    def __init__(
        self,
        *,
        epsilon_overall: float = 0.05,
        epsilon_category: float = 0.10,
        min_scenarios: int = 50,
        min_scenarios_per_category: int = 8,
        min_repetitions: int = 3,
        bootstrap_resamples: int = 2000,
        seed: int = 0,
    ) -> None:
        self.epsilon_overall = float(epsilon_overall)
        self.epsilon_category = float(epsilon_category)
        self.min_scenarios = int(min_scenarios)
        self.min_scenarios_per_category = int(min_scenarios_per_category)
        self.min_repetitions = int(min_repetitions)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.seed = int(seed)

    def evaluate(self, scenario_results: Sequence[ScenarioResult]) -> VerdictResult:
        if not scenario_results:
            raise ValueError("At least one scenario result is required.")

        overall_bootstrap = paired_scenario_bootstrap(
            [row.model_dump() for row in scenario_results],
            resamples=self.bootstrap_resamples,
            seed=self.seed,
        )
        overall = overall_bootstrap["overall"]

        category_bootstraps = paired_bootstrap_by_category(
            [row.model_dump() for row in scenario_results],
            resamples=self.bootstrap_resamples,
            seed=self.seed,
        )

        reasons: list[str] = []
        blockers: list[str] = []
        notes: list[str] = []
        category_results: dict[str, CategoryResult] = {}
        underpowered_categories: list[str] = []
        categories: dict[str, list[ScenarioResult]] = defaultdict(list)
        for row in scenario_results:
            categories[row.category].append(row)

        for category_name in sorted(categories):
            rows = categories[category_name]
            count = len(rows)
            category_delta = float(
                np.mean([row.candidate_pass_rate - row.baseline_pass_rate for row in rows])
            )
            if count < self.min_scenarios_per_category:
                underpowered_categories.append(category_name)
                category_results[category_name] = CategoryResult(
                    category=category_name,
                    scenario_count=count,
                    estimate=category_delta,
                    ci=(0.0, 0.0),
                    powered=False,
                    status="UNDERPOWERED",
                    reasons=[
                        f"Category '{category_name}' has {count} scenarios, below the minimum of {self.min_scenarios_per_category}."
                    ],
                    blockers=[],
                    notes=[
                        f"Category '{category_name}' has {count} scenarios, below the minimum of {self.min_scenarios_per_category}."
                    ],
                )
                continue

            summary = category_bootstraps[category_name]
            lower, upper = float(summary.lower), float(summary.upper)
            category_reasons: list[str] = []
            if lower >= -self.epsilon_category:
                status: Literal["PASS", "FAIL", "INCONCLUSIVE"] = "PASS"
                category_reasons.append(
                    f"Category '{category_name}' is non-inferior: lower CI {lower:.3f} >= -{self.epsilon_category:.2f}."
                )
            elif upper < -self.epsilon_category:
                status = "FAIL"
                category_reasons.append(
                    f"Category '{category_name}' shows statistically significant degradation: upper CI {upper:.3f} < -{self.epsilon_category:.2f}."
                )
            else:
                status = "INCONCLUSIVE"
                category_reasons.append(
                    f"Category '{category_name}' interval spans the category margin: [{lower:.3f}, {upper:.3f}]."
                )

            category_blockers = [
                reason
                for reason in category_reasons
                if "degradation" in reason.lower() or "significant" in reason.lower()
            ]
            category_notes = [
                reason for reason in category_reasons if reason not in category_blockers
            ]
            category_results[category_name] = CategoryResult(
                category=category_name,
                scenario_count=count,
                estimate=category_delta,
                ci=(lower, upper),
                powered=True,
                status=status,
                reasons=category_reasons,
                blockers=category_blockers,
                notes=category_notes,
            )

        if len(scenario_results) < self.min_scenarios:
            blockers.append(
                f"Overall sample size below minimum ({len(scenario_results)} < {self.min_scenarios})."
            )

        low_repetitions = any(
            row.baseline_repetitions < self.min_repetitions
            or row.candidate_repetitions < self.min_repetitions
            for row in scenario_results
        )
        if low_repetitions:
            blockers.append(
                f"Repetitions below minimum: at least one scenario has fewer than "
                f"{self.min_repetitions} repetitions under one config."
            )

        delta_values = np.asarray(
            [float(row.candidate_pass_rate - row.baseline_pass_rate) for row in scenario_results],
            dtype=float,
        )
        total_new_violations = sum(
            int(row.new_critical_policy_violations) for row in scenario_results
        )
        if total_new_violations > 0:
            blockers.append(f"New critical policy violations detected: {total_new_violations}.")

        if overall.upper < -self.epsilon_overall:
            blockers.append(
                "Overall candidate is definitively worse than baseline: upper CI "
                f"{overall.upper:.3f} < -{self.epsilon_overall:.2f}."
            )
        elif overall.lower < -self.epsilon_overall <= overall.upper:
            blockers.append(
                "Overall interval spans the decision margin and remains inconclusive: "
                f"[{overall.lower:.3f}, {overall.upper:.3f}] crosses -{self.epsilon_overall:.2f}."
            )
        elif overall.lower >= -self.epsilon_overall:
            notes.append(
                "Overall non-inferiority holds: lower CI "
                f"{overall.lower:.3f} >= -{self.epsilon_overall:.2f}."
            )

        if any(result.status == "FAIL" for result in category_results.values()):
            blockers.append(
                "At least one powered category has a statistically significant degradation."
            )

        verdict: Literal["SHIP", "HOLD", "INCONCLUSIVE"] = "SHIP"
        powered_failures = [
            result
            for result in category_results.values()
            if result.powered and result.status == "FAIL"
        ]
        if overall.upper < -self.epsilon_overall or total_new_violations > 0 or powered_failures:
            verdict = "HOLD"
        elif len(scenario_results) < self.min_scenarios or (
            overall.lower < -self.epsilon_overall <= overall.upper
        ):
            verdict = "INCONCLUSIVE"
        elif (
            len(scenario_results) >= self.min_scenarios
            and overall.lower >= -self.epsilon_overall
            and overall.width <= 2 * self.epsilon_overall
            and all(
                result.status in {"PASS", "UNDERPOWERED"}
                for result in category_results.values()
                if result.powered
            )
            and total_new_violations == 0
            and not low_repetitions
        ):
            verdict = "SHIP"
        else:
            verdict = "INCONCLUSIVE"
            for category_result in category_results.values():
                if category_result.powered and category_result.status == "INCONCLUSIVE":
                    blockers.append(
                        f"Category '{category_result.category}' is inconclusive: its interval "
                        f"[{category_result.ci[0]:.3f}, {category_result.ci[1]:.3f}] spans the "
                        f"category margin -{self.epsilon_category:.2f}."
                    )
            if (
                len(scenario_results) >= self.min_scenarios
                and overall.lower >= -self.epsilon_overall
                and overall.width > 2 * self.epsilon_overall
            ):
                blockers.append(
                    f"Overall interval too wide to establish non-inferiority: width "
                    f"{overall.width:.3f} > {2 * self.epsilon_overall:.2f}."
                )

        required_sample_size: int | None = None
        if verdict == "INCONCLUSIVE":
            variance = float(np.var(delta_values, ddof=1)) if len(delta_values) > 1 else 0.0
            numerator = (1.96 + 0.84) ** 2 * variance
            if self.epsilon_overall > 0.0:
                calc_n = int(ceil(numerator / (self.epsilon_overall**2)))
                required_sample_size = max(calc_n, self.min_scenarios)
            else:
                required_sample_size = self.min_scenarios

        cost_summary = {
            "baseline": self._cost_summary(scenario_results, config="baseline"),
            "candidate": self._cost_summary(scenario_results, config="candidate"),
        }
        latency_summary = {
            "baseline": self._latency_summary(scenario_results, config="baseline"),
            "candidate": self._latency_summary(scenario_results, config="candidate"),
        }
        token_totals = {
            "baseline": self._token_totals(scenario_results, config="baseline"),
            "candidate": self._token_totals(scenario_results, config="candidate"),
        }

        reasons = blockers + notes

        return VerdictResult(
            verdict=verdict,
            reasons=reasons,
            overall_delta=overall.estimate,
            overall_ci=(overall.lower, overall.upper),
            required_sample_size=required_sample_size,
            underpowered_categories=underpowered_categories,
            category_results=category_results,
            cost_summary=cost_summary,
            latency_summary=latency_summary,
            token_totals=token_totals,
            bootstrap_resamples=self.bootstrap_resamples,
            metadata={
                "epsilon_overall": self.epsilon_overall,
                "epsilon_category": self.epsilon_category,
                "min_scenarios": self.min_scenarios,
                "min_repetitions": self.min_repetitions,
                "min_scenarios_per_category": self.min_scenarios_per_category,
            },
            blockers=blockers,
            notes=notes,
        )

    @staticmethod
    def _cost_summary(
        scenario_results: Sequence[ScenarioResult], *, config: str
    ) -> dict[str, float | int]:
        total_cost = sum(float(getattr(row, f"{config}_cost")) for row in scenario_results)
        successes = sum(int(getattr(row, f"{config}_successes")) for row in scenario_results)
        cost_per_success = total_cost / max(successes, 1)
        return {
            "total_cost": float(total_cost),
            "successful_tasks": int(successes),
            "cost_per_successful_task": float(cost_per_success),
        }

    @staticmethod
    def _latency_summary(
        scenario_results: Sequence[ScenarioResult], *, config: str
    ) -> dict[str, float]:
        latencies: list[float] = []
        for row in scenario_results:
            latencies.extend(float(value) for value in getattr(row, f"{config}_latency_ms"))
        if not latencies:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0}
        arr = np.asarray(latencies, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
        }

    @staticmethod
    def _token_totals(scenario_results: Sequence[ScenarioResult], *, config: str) -> dict[str, int]:
        prompt = sum(int(getattr(row, f"{config}_prompt_tokens")) for row in scenario_results)
        completion = sum(
            int(getattr(row, f"{config}_completion_tokens")) for row in scenario_results
        )
        return {
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(prompt + completion),
        }
