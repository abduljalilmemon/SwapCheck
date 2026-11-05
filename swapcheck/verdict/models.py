from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class ScenarioResult(BaseModel):
    """One scenario observed under a baseline and candidate configuration."""

    scenario_id: str = Field(..., description="Stable scenario identifier.")
    category: str = Field(default="uncategorized", description="Scenario category.")
    baseline_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    baseline_repetitions: int = Field(default=1, ge=1)
    candidate_repetitions: int = Field(default=1, ge=1)
    baseline_cost: float = Field(default=0.0, ge=0.0)
    candidate_cost: float = Field(default=0.0, ge=0.0)
    baseline_successes: int = Field(default=0, ge=0)
    candidate_successes: int = Field(default=0, ge=0)
    baseline_latency_ms: list[float] = Field(default_factory=list)
    candidate_latency_ms: list[float] = Field(default_factory=list)
    baseline_prompt_tokens: int = Field(default=0, ge=0)
    baseline_completion_tokens: int = Field(default=0, ge=0)
    candidate_prompt_tokens: int = Field(default=0, ge=0)
    candidate_completion_tokens: int = Field(default=0, ge=0)
    baseline_critical_policy_violations: int = Field(default=0, ge=0)
    candidate_critical_policy_violations: int = Field(default=0, ge=0)
    new_critical_policy_violations: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class BootstrapResult:
    """A percentile bootstrap summary for the scenario-level mean difference."""

    estimate: float
    lower: float
    upper: float
    resamples: int = 2000

    def __iter__(self) -> Iterator[float]:
        yield self.lower
        yield self.upper

    def __getitem__(self, index: int) -> float:
        return (self.lower, self.upper)[index]

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass
class CategoryResult:
    """Per-category verdict metadata."""

    category: str
    scenario_count: int
    estimate: float
    ci: tuple[float, float]
    powered: bool
    status: Literal["PASS", "FAIL", "INCONCLUSIVE", "UNDERPOWERED"]
    reasons: list[str] = field(default_factory=list)


@dataclass
class VerdictResult:
    """Overall verdict for a model comparison run."""

    verdict: Literal["SHIP", "HOLD", "INCONCLUSIVE"]
    reasons: list[str]
    overall_delta: float
    overall_ci: tuple[float, float]
    required_sample_size: int | None = None
    underpowered_categories: list[str] = field(default_factory=list)
    category_results: dict[str, CategoryResult] = field(default_factory=dict)
    cost_summary: dict[str, dict[str, float | int]] = field(default_factory=dict)
    latency_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    token_totals: dict[str, dict[str, int]] = field(default_factory=dict)
    bootstrap_resamples: int = 2000
    metadata: dict[str, Any] = field(default_factory=dict)
