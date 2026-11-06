"""Typer command-line interface for SwapCheck."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swapcheck.config import ModelConfig, load_model_config
from swapcheck.providers.base import BaseProvider, ChatMessage, LLMResponse, TokenUsage, ToolCall
from swapcheck.providers.mock_provider import MockProvider
from swapcheck.reports.json_export import export_json_report
from swapcheck.reports.markdown import render_markdown_report
from swapcheck.scenarios.loader import load_scenarios
from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.runner import ScenarioRunner
from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.verdict.engine import VerdictEngine
from swapcheck.verdict.models import ScenarioResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Statistically grounded evaluation harness for tool-using LLM agents.",
)
console = Console()


def _parse_booking_id(text: str) -> str:
    match = re.search(r"BKG-\d+", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else "BKG-1000"


def _parse_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else "customer@example.com"


def _tool_response(name: str, *, booking_id: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id=f"tool_{name}", name=name, arguments={"booking_id": booking_id, **arguments}
            )
        ],
        usage=TokenUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
        latency_ms=15.0,
    )


def _make_mock_provider(config: ModelConfig) -> MockProvider:
    provider = MockProvider(
        model_name=config.model,
        temperature=config.temperature,
        cost_per_input_token=config.cost_per_input_token or 0.0,
        cost_per_output_token=config.cost_per_output_token or 0.0,
    )

    def reply(messages: list[ChatMessage]) -> LLMResponse:
        conversation = " ".join((message.content or "") for message in messages if message.content)
        last_message = messages[-1] if messages else ChatMessage(role="user", content="")
        booking_id = _parse_booking_id(conversation)
        email = _parse_email(conversation)
        lower = conversation.lower()

        if last_message.role == "tool":
            if last_message.name == "lookup_booking":
                if "cancel" in lower or "refund" in lower:
                    return _tool_response(
                        "cancel_booking",
                        booking_id=booking_id,
                        reason="Customer requested cancellation.",
                    )
                return LLMResponse(
                    content=f"I found booking {booking_id}. The customer email on file matches the request.",
                    usage=TokenUsage(prompt_tokens=25, completion_tokens=18, total_tokens=43),
                    latency_ms=10.0,
                )
            if last_message.name == "cancel_booking":
                if "refund" in lower or "money" in lower or "fee" in lower:
                    return _tool_response(
                        "issue_refund",
                        booking_id=booking_id,
                        amount=120.0,
                        reason="Policy-eligible refund request.",
                    )
                return LLMResponse(
                    content=f"Booking {booking_id} has been cancelled successfully.",
                    usage=TokenUsage(prompt_tokens=20, completion_tokens=12, total_tokens=32),
                    latency_ms=9.0,
                )
            if last_message.name == "issue_refund":
                return LLMResponse(
                    content=f"I have processed the refund for {booking_id} and confirmed the booking status.",
                    usage=TokenUsage(prompt_tokens=18, completion_tokens=14, total_tokens=32),
                    latency_ms=8.0,
                )
            if last_message.name == "escalate_to_human":
                return LLMResponse(
                    content=f"I’ve escalated the request for {booking_id} to a human specialist.",
                    usage=TokenUsage(prompt_tokens=18, completion_tokens=10, total_tokens=28),
                    latency_ms=8.0,
                )

        if "lookup" in lower or "booking" in lower:
            return _tool_response(
                "lookup_booking",
                booking_id=booking_id,
                customer_email=email,
            )
        if "cancel" in lower:
            return _tool_response(
                "cancel_booking",
                booking_id=booking_id,
                reason="Customer requested cancellation.",
            )
        if "refund" in lower or "money back" in lower:
            return _tool_response(
                "issue_refund",
                booking_id=booking_id,
                amount=120.0,
                reason="Policy-eligible refund request.",
            )
        if "escalate" in lower:
            return _tool_response(
                "escalate_to_human",
                booking_id=booking_id,
                reason="Need a specialist review for this request.",
            )

        return LLMResponse(
            content="I can help with that booking request.",
            usage=TokenUsage(prompt_tokens=18, completion_tokens=9, total_tokens=27),
            latency_ms=8.0,
        )

    provider.add_rule(r".*", reply)
    return provider


def _build_provider_factory(config: ModelConfig) -> Callable[[], BaseProvider]:
    if config.provider.lower() != "mock":
        raise typer.BadParameter(
            f"Provider '{config.provider}' is not supported by the offline CLI. Use a mock provider config.",
            param_hint="--baseline",
        )

    def factory() -> BaseProvider:
        return _make_mock_provider(config)

    return factory


def _evaluate_scenario(
    scenario: Scenario,
    *,
    baseline_trajectories: list[Trajectory],
    candidate_trajectories: list[Trajectory],
) -> ScenarioResult:
    baseline_successes = 0
    candidate_successes = 0

    for trajectory in baseline_trajectories:
        if trajectory.budget_exceeded:
            continue
        if _run_passed(scenario, trajectory):
            baseline_successes += 1

    for trajectory in candidate_trajectories:
        if trajectory.budget_exceeded:
            continue
        if _run_passed(scenario, trajectory):
            candidate_successes += 1

    return ScenarioResult(
        scenario_id=scenario.id,
        category=scenario.category,
        baseline_pass_rate=baseline_successes / max(len(baseline_trajectories), 1),
        candidate_pass_rate=candidate_successes / max(len(candidate_trajectories), 1),
        baseline_repetitions=len(baseline_trajectories),
        candidate_repetitions=len(candidate_trajectories),
        baseline_cost=sum(float(trajectory.total_cost) for trajectory in baseline_trajectories),
        candidate_cost=sum(float(trajectory.total_cost) for trajectory in candidate_trajectories),
        baseline_successes=baseline_successes,
        candidate_successes=candidate_successes,
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
        baseline_critical_policy_violations=0,
        candidate_critical_policy_violations=0,
        new_critical_policy_violations=0,
    )


def _run_passed(scenario: Scenario, trajectory: Trajectory) -> bool:
    expected_final = scenario.expected_final_db_mapping()
    actual_final = trajectory.final_db_snapshot

    for booking_id, expected_entry in expected_final.items():
        actual_entry = actual_final.get(booking_id)
        if actual_entry is None:
            return False
        for key, expected_value in expected_entry.items():
            if actual_entry.get(key) != expected_value:
                return False

    tool_names: list[str] = []
    for turn in trajectory.turns:
        for call in turn.tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("name")
            if isinstance(tool_name, str):
                tool_names.append(tool_name)

    for forbidden in scenario.forbidden_tool_calls:
        if forbidden in tool_names:
            return False

    if scenario.expected_tool_calls:
        missing = [name for name in scenario.expected_tool_calls if name not in tool_names]
        if missing:
            return False

    return True


async def _run_suite(
    *,
    baseline_config: ModelConfig,
    candidate_config: ModelConfig,
    suite_path: Path,
    runs: int,
    max_cost: float | None,
) -> list[ScenarioResult]:
    grouped = load_scenarios(suite_path)
    scenario_results: list[ScenarioResult] = []
    baseline_factory = _build_provider_factory(baseline_config)
    candidate_factory = _build_provider_factory(candidate_config)

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
        typer.Option(..., "--baseline", exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
    candidate: Annotated[
        Path,
        typer.Option(
            ..., "--candidate", exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    ],
    suite: Annotated[
        Path,
        typer.Option(..., "--suite", exists=True, file_okay=True, dir_okay=True, readable=True),
    ],
    runs: Annotated[int, typer.Option("--runs", min=1)] = 3,
    margin: Annotated[float, typer.Option("--margin", min=0.0, max=1.0)] = 0.05,
    max_cost: Annotated[float | None, typer.Option("--max-cost", min=0.0)] = None,
    output: Annotated[Path | None, typer.Option("--output", file_okay=True, dir_okay=False)] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run a baseline-vs-candidate evaluation over a scenario suite."""
    baseline_config = load_model_config(baseline)
    candidate_config = load_model_config(candidate)

    scenario_results = asyncio.run(
        _run_suite(
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            suite_path=suite,
            runs=runs,
            max_cost=max_cost,
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
        suite_path=str(suite),
    )

    if output is not None:
        output.write_text(report, encoding="utf-8")
        console.print(f"[green]Wrote report to {output}[/green]")

    if json_output:
        json_report = export_json_report(
            verdict,
            baseline_name=baseline_config.name,
            candidate_name=candidate_config.name,
            suite_path=str(suite),
        )
        console.print(Panel(json_report, title="JSON Report", expand=False))
        return

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
    path: Annotated[Path, typer.Argument(..., exists=True, readable=True)],
) -> None:
    """Render a report in the terminal from an existing JSON export."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    console.print(
        Panel(json.dumps(payload, indent=2, sort_keys=True), title=f"JSON report: {path}")
    )


@app.command()
def label(
    run_id: Annotated[str, typer.Option("--run")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Label recorded trajectories from a run. Placeholder for the future human-label flow."""
    output.write_text(
        "trajectory_id,scenario_id,helpfulness,groundedness,tone,off_policy_promises,passed,notes\n",
        encoding="utf-8",
    )
    console.print(f"[yellow]Created empty label template for run {run_id} at {output}[/yellow]")


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
