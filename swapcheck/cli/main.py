"""Typer command-line interface for SwapCheck."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swapcheck.config import ModelConfig, load_model_config
from swapcheck.providers import build_provider
from swapcheck.reports.json_export import export_json_report
from swapcheck.reports.markdown import render_markdown_report
from swapcheck.scenarios.loader import load_scenarios
from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.runner import ScenarioRunner
from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.scoring.deterministic import DeterministicScorer
from swapcheck.verdict.engine import VerdictEngine
from swapcheck.verdict.models import ScenarioResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Statistically grounded evaluation harness for tool-using LLM agents.",
)
console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_candidate = _project_root() / candidate
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _evaluate_scenario(
    scenario: Scenario,
    *,
    baseline_trajectories: list[Trajectory],
    candidate_trajectories: list[Trajectory],
) -> ScenarioResult:
    scorer = DeterministicScorer()
    baseline_passes = 0
    candidate_passes = 0
    baseline_critical = 0
    candidate_critical = 0

    for trajectory in baseline_trajectories:
        if trajectory.budget_exceeded:
            continue
        breakdown = scorer.score(trajectory, scenario)
        if breakdown.passed:
            baseline_passes += 1
        baseline_critical += sum(
            1 for violation in breakdown.policy_violations if violation.severity == "critical"
        )

    for trajectory in candidate_trajectories:
        if trajectory.budget_exceeded:
            continue
        breakdown = scorer.score(trajectory, scenario)
        if breakdown.passed:
            candidate_passes += 1
        candidate_critical += sum(
            1 for violation in breakdown.policy_violations if violation.severity == "critical"
        )

    baseline_count = max(len(baseline_trajectories), 1)
    candidate_count = max(len(candidate_trajectories), 1)
    return ScenarioResult(
        scenario_id=scenario.id,
        category=scenario.category,
        baseline_pass_rate=baseline_passes / baseline_count,
        candidate_pass_rate=candidate_passes / candidate_count,
        baseline_repetitions=len(baseline_trajectories),
        candidate_repetitions=len(candidate_trajectories),
        baseline_cost=sum(float(trajectory.total_cost) for trajectory in baseline_trajectories),
        candidate_cost=sum(float(trajectory.total_cost) for trajectory in candidate_trajectories),
        baseline_successes=baseline_passes,
        candidate_successes=candidate_passes,
        baseline_latency_ms=[
            float(turn.latency_ms)
            for trajectory in baseline_trajectories
            for turn in trajectory.turns
        ],
        candidate_latency_ms=[
            float(turn.latency_ms)
            for trajectory in candidate_trajectories
            for turn in trajectory.turns
        ],
        baseline_prompt_tokens=sum(
            int(turn.tokens) for trajectory in baseline_trajectories for turn in trajectory.turns
        ),
        baseline_completion_tokens=0,
        candidate_prompt_tokens=sum(
            int(turn.tokens) for trajectory in candidate_trajectories for turn in trajectory.turns
        ),
        candidate_completion_tokens=0,
        baseline_critical_policy_violations=baseline_critical,
        candidate_critical_policy_violations=candidate_critical,
        new_critical_policy_violations=max(candidate_critical - baseline_critical, 0),
    )


async def _run_suite(
    *,
    baseline_config: ModelConfig,
    candidate_config: ModelConfig,
    suite_path: Path,
    runs: int,
    max_cost: float | None,
    no_cache: bool,
) -> list[ScenarioResult]:
    grouped = load_scenarios(suite_path)
    scenario_results: list[ScenarioResult] = []

    def baseline_factory() -> Any:
        return build_provider(baseline_config)

    def candidate_factory() -> Any:
        return build_provider(candidate_config)

    for category in sorted(grouped):
        for scenario in grouped[category]:
            baseline_runner = ScenarioRunner(
                scenarios=[scenario],
                repetitions=runs,
                max_concurrency=1,
                provider_factory=baseline_factory,
                max_cost=max_cost,
            )
            candidate_runner = ScenarioRunner(
                scenarios=[scenario],
                repetitions=runs,
                max_concurrency=1,
                provider_factory=candidate_factory,
                max_cost=max_cost,
            )
            if no_cache:
                baseline_runner = ScenarioRunner(
                    scenarios=[scenario],
                    repetitions=runs,
                    max_concurrency=1,
                    provider_factory=baseline_factory,
                    max_cost=max_cost,
                    cache=None,
                )
                candidate_runner = ScenarioRunner(
                    scenarios=[scenario],
                    repetitions=runs,
                    max_concurrency=1,
                    provider_factory=candidate_factory,
                    max_cost=max_cost,
                    cache=None,
                )
            baseline_trajectories = await baseline_runner.run_all()
            candidate_trajectories = await candidate_runner.run_all()
            scenario_results.append(
                _evaluate_scenario(
                    scenario,
                    baseline_trajectories=baseline_trajectories,
                    candidate_trajectories=candidate_trajectories,
                )
            )

    return scenario_results


@app.command()
def run(
    baseline: Annotated[
        Path,
        typer.Option(..., "--baseline", exists=False, file_okay=True, dir_okay=False),
    ],
    candidate: Annotated[
        Path,
        typer.Option(..., "--candidate", exists=False, file_okay=True, dir_okay=False),
    ],
    suite: Annotated[
        Path,
        typer.Option(..., "--suite", exists=False, file_okay=True, dir_okay=True),
    ],
    judge: Annotated[
        Path | None, typer.Option("--judge", exists=False, file_okay=True, dir_okay=False)
    ] = None,
    runs: Annotated[int, typer.Option("--runs", min=1)] = 3,
    margin: Annotated[float, typer.Option("--margin", min=0.0, max=1.0)] = 0.05,
    max_cost: Annotated[float | None, typer.Option("--max-cost", min=0.0)] = None,
    output: Annotated[Path | None, typer.Option("--output", file_okay=True, dir_okay=False)] = None,
    json_output: Annotated[
        Path | None, typer.Option("--json-output", file_okay=True, dir_okay=False)
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
) -> None:
    """Run a baseline-vs-candidate evaluation over a scenario suite."""
    baseline_path = _resolve_path(baseline)
    candidate_path = _resolve_path(candidate)
    suite_path = _resolve_path(suite)
    judge_path = _resolve_path(judge) if judge else None

    baseline_config = load_model_config(baseline_path)
    candidate_config = load_model_config(candidate_path)
    if judge_path is not None:
        load_model_config(judge_path)

    scenario_results = asyncio.run(
        _run_suite(
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            suite_path=suite_path,
            runs=runs,
            max_cost=max_cost,
            no_cache=no_cache,
        )
    )
    verdict = VerdictEngine(
        epsilon_overall=margin,
        epsilon_category=0.10,
        min_scenarios=50,
        min_scenarios_per_category=8,
        min_repetitions=1,
        bootstrap_resamples=200,
    ).evaluate(scenario_results)

    report = render_markdown_report(
        verdict,
        baseline_name=baseline_config.name,
        candidate_name=candidate_config.name,
        suite_path=str(suite_path),
    )

    if output is not None:
        output_path = _resolve_path(output)
        output_path.write_text(report, encoding="utf-8")
        console.print(f"[green]Wrote markdown report to {output_path}[/green]")

    if json_output is not None:
        json_path = _resolve_path(json_output)
        json_report = export_json_report(
            verdict,
            baseline_name=baseline_config.name,
            candidate_name=candidate_config.name,
            suite_path=str(suite_path),
        )
        json_path.write_text(json_report, encoding="utf-8")
        console.print(f"[green]Wrote JSON report to {json_path}[/green]")

    table = Table(title=f"Verdict: {verdict.verdict}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Overall delta", f"{verdict.overall_delta:.3f}")
    table.add_row("Overall CI", f"[{verdict.overall_ci[0]:.3f}, {verdict.overall_ci[1]:.3f}]")
    table.add_row("Required sample", str(verdict.required_sample_size))
    table.add_row("Reasons", str(len(verdict.reasons)))
    console.print(table)
    console.print(Panel(report, title="Report", expand=False))


@app.command()
def report(
    path: Annotated[Path, typer.Argument(..., exists=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", file_okay=True, dir_okay=False)] = None,
    format: Annotated[str, typer.Option("--format")] = "markdown",
) -> None:
    """Render a terminal or file report from an existing JSON export."""
    payload_path = _resolve_path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if format.lower() == "json":
        report_text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        report_text = "\n".join(
            [
                "# SwapCheck Report",
                "",
                f"- Verdict: {payload.get('verdict', 'UNKNOWN')}",
                f"- Overall delta: {payload.get('overall_delta', 0.0)}",
                f"- Overall CI: {payload.get('overall_ci', [])}",
            ]
        )
    if output is not None:
        _resolve_path(output).write_text(report_text, encoding="utf-8")
    console.print(Panel(report_text, title=f"JSON report: {payload_path}"))


@app.command()
def label(
    run_id: Annotated[str, typer.Option("--run")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Label recorded trajectories from a run. Placeholder for the future human-label flow."""
    output_path = _resolve_path(output)
    output_path.write_text(
        "trajectory_id,scenario_id,helpfulness,groundedness,tone,off_policy_promises,passed,notes\n",
        encoding="utf-8",
    )
    console.print(
        f"[yellow]Created empty label template for run {run_id} at {output_path}[/yellow]"
    )


@app.command()
def calibrate(
    labels: Annotated[Path, typer.Option("--labels")],
    suite: Annotated[Path, typer.Option("--suite")],
) -> None:
    """Calculate label agreement metrics. Placeholder for the future calibration module."""
    console.print(
        f"[yellow]Calibration is not yet implemented for {labels} with suite {suite}; the label template was prepared.[/yellow]"
    )


if __name__ == "__main__":
    app()
