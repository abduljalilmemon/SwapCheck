"""Typer command-line interface for SwapCheck."""

from __future__ import annotations

import asyncio
import csv
import json
import random
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swapcheck.cache import ResponseCache
from swapcheck.config import ModelConfig, load_model_config
from swapcheck.providers import build_provider
from swapcheck.reports.json_export import export_json_report
from swapcheck.reports.markdown import render_markdown_report
from swapcheck.runs import (
    ROLES,
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
from swapcheck.scenarios.loader import load_scenarios
from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.runner import CachedProvider, ScenarioRunner
from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.scoring.calibration import LABEL_TEMPLATE, summarize_calibration
from swapcheck.scoring.deterministic import DeterministicScorer
from swapcheck.scoring.judge import JudgeScorer
from swapcheck.verdict.engine import VerdictEngine

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Statistically grounded evaluation harness for tool-using LLM agents.",
)
console = Console()

DEFAULT_CACHE_PATH = Path(".swapcheck_cache.db")


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


def _fail(message: str, code: int = 1) -> typer.Exit:
    console.print(f"[red]{message}[/red]", markup=True)
    return typer.Exit(code=code)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


async def _run_config(
    *,
    config: ModelConfig,
    scenarios: list[Scenario],
    runs: int,
    concurrency: int,
    max_cost: float | None,
    cache: ResponseCache | None,
) -> list[Trajectory]:
    runner = ScenarioRunner(
        scenarios=scenarios,
        repetitions=runs,
        max_concurrency=concurrency,
        provider_factory=lambda: build_provider(config),
        cache=cache,
        max_cost=max_cost,
        use_cache=cache is not None,
    )
    return await runner.run_all()


async def _judge_records(
    records: list[TrajectoryRecord],
    *,
    scorer: JudgeScorer,
    concurrency: int,
    max_cost: float | None,
) -> int:
    """Attach judge results to records. Returns how many were skipped because of the budget."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    spent = 0.0
    skipped = 0

    async def judge_one(record: TrajectoryRecord) -> None:
        nonlocal spent, skipped
        async with semaphore:
            async with lock:
                if max_cost is not None and spent >= max_cost:
                    skipped += 1
                    return
            result = await scorer.score(record.trajectory, scenario_id=record.scenario_id)
            async with lock:
                spent += float(result.cost)
        record.judge_result = result.model_dump()

    await asyncio.gather(*(judge_one(record) for record in records))
    return skipped


def _refuse_same_model(judge: ModelConfig, others: dict[str, ModelConfig]) -> None:
    for role, config in others.items():
        if judge.model == config.model:
            raise _fail(
                f"The judge model '{judge.model}' is the same as the {role} model. "
                "Use a different model for the judge to avoid self-preference bias.",
                code=2,
            )


@app.command()
def run(
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline model config (YAML).")],
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate model config (YAML).")],
    suite: Annotated[Path, typer.Option("--suite", help="Scenario suite directory.")],
    judge: Annotated[
        Path | None, typer.Option("--judge", help="Judge model config (YAML). Optional.")
    ] = None,
    runs: Annotated[int, typer.Option("--runs", min=1, help="Repetitions per scenario.")] = 3,
    margin: Annotated[
        float, typer.Option("--margin", min=0.0, max=1.0, help="Overall non-inferiority margin.")
    ] = 0.05,
    category_margin: Annotated[float, typer.Option("--category-margin", min=0.0, max=1.0)] = 0.10,
    min_scenarios: Annotated[int, typer.Option("--min-scenarios", min=1)] = 50,
    min_scenarios_per_category: Annotated[
        int, typer.Option("--min-scenarios-per-category", min=1)
    ] = 8,
    min_repetitions: Annotated[int, typer.Option("--min-repetitions", min=1)] = 3,
    bootstrap_resamples: Annotated[int, typer.Option("--bootstrap-resamples", min=100)] = 2000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    max_cost: Annotated[
        float | None,
        typer.Option(
            "--max-cost",
            min=0.0,
            help="USD budget, applied separately to the baseline run, the candidate run and the judge.",
        ),
    ] = None,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 4,
    judge_affects_pass: Annotated[
        bool,
        typer.Option(
            "--judge-affects-pass",
            help="Require the judge to pass a trajectory too. Off by default.",
        ),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Run directory (default: runs/<timestamp>)."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", help="Markdown report path (default: in run dir).")
    ] = None,
    json_output: Annotated[
        Path | None, typer.Option("--json-output", help="JSON report path (default: in run dir).")
    ] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable response caching.")] = False,
    cache_path: Annotated[Path, typer.Option("--cache-path")] = DEFAULT_CACHE_PATH,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on",
            help=(
                "Exit non-zero for CI gating: 'hold' fails on HOLD (exit 10); 'not-ship' also "
                "fails on INCONCLUSIVE (exit 11). Default: never."
            ),
        ),
    ] = "never",
) -> None:
    """Run a baseline-vs-candidate evaluation over a scenario suite."""
    if fail_on not in {"never", "hold", "not-ship"}:
        raise _fail("--fail-on must be one of: never, hold, not-ship.", code=2)
    if judge_affects_pass and judge is None:
        raise _fail("--judge-affects-pass requires --judge.", code=2)

    baseline_config = load_model_config(_resolve_path(baseline))
    candidate_config = load_model_config(_resolve_path(candidate))
    judge_config = load_model_config(_resolve_path(judge)) if judge is not None else None
    if judge_config is not None:
        _refuse_same_model(
            judge_config, {"baseline": baseline_config, "candidate": candidate_config}
        )

    suite_path = _resolve_path(suite)
    grouped = load_scenarios(suite_path)
    scenarios = [scenario for category in sorted(grouped) for scenario in grouped[category]]
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}

    run_dir = output_dir or Path("runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = None if no_cache else ResponseCache(db_path=cache_path)

    configs = {"baseline": baseline_config, "candidate": candidate_config}
    scorer = DeterministicScorer()
    records: list[TrajectoryRecord] = []
    for role in ROLES:
        config = configs[role]
        console.print(f"Running {role} '{config.name}': {len(scenarios)} scenarios x {runs} runs")
        trajectories = asyncio.run(
            _run_config(
                config=config,
                scenarios=scenarios,
                runs=runs,
                concurrency=concurrency,
                max_cost=max_cost,
                cache=cache,
            )
        )
        for trajectory in trajectories:
            scenario = scenario_by_id[trajectory.scenario_id]
            breakdown = scorer.score(trajectory, scenario)
            records.append(
                TrajectoryRecord(
                    trajectory_id=make_trajectory_id(
                        trajectory.scenario_id, role, trajectory.repetition_index
                    ),
                    role=role,
                    config_name=config.name,
                    scenario_id=trajectory.scenario_id,
                    category=trajectory.category,
                    repetition_index=trajectory.repetition_index,
                    trajectory=trajectory,
                    score=breakdown,
                    passed=breakdown.passed,
                )
            )

    judge_skipped = 0
    if judge_config is not None:
        console.print(f"Judging {len(records)} trajectories with '{judge_config.name}'")
        judge_provider = build_provider(judge_config)
        if cache is not None:
            judge_provider = CachedProvider(provider=judge_provider, cache=cache)
        judge_scorer = JudgeScorer(provider=judge_provider)
        to_judge = [record for record in records if not record.trajectory.budget_exceeded]
        judge_skipped = asyncio.run(
            _judge_records(
                to_judge, scorer=judge_scorer, concurrency=concurrency, max_cost=max_cost
            )
        )
        if judge_skipped and judge_affects_pass:
            raise _fail(
                f"The judge budget ran out with {judge_skipped} trajectories unjudged, and "
                "--judge-affects-pass is set. Raise --max-cost or drop --judge-affects-pass.",
                code=3,
            )

    for record in records:
        record.passed = decide_pass(
            record.score, record.judge_result, judge_affects_pass=judge_affects_pass
        )

    results = build_scenario_results(records)
    if not results:
        raise _fail("No scenario completed under both configs; nothing to evaluate.", code=3)

    verdict = VerdictEngine(
        epsilon_overall=margin,
        epsilon_category=category_margin,
        min_scenarios=min_scenarios,
        min_scenarios_per_category=min_scenarios_per_category,
        min_repetitions=min_repetitions,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    ).evaluate(results)

    metrics = judge_metrics(records)
    if metrics is not None:
        verdict.metadata["judge_metrics"] = metrics
    summary = build_summary(records, results)

    settings = {
        "runs": runs,
        "margin": margin,
        "category_margin": category_margin,
        "min_scenarios": min_scenarios,
        "min_scenarios_per_category": min_scenarios_per_category,
        "min_repetitions": min_repetitions,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "max_cost": max_cost,
        "concurrency": concurrency,
        "judge_affects_pass": judge_affects_pass,
        "cache": not no_cache,
        "judge_skipped_for_budget": judge_skipped,
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "suite": str(suite_path),
        "baseline": baseline_config.model_dump(),
        "candidate": candidate_config.model_dump(),
        "judge": judge_config.model_dump() if judge_config is not None else None,
        "settings": settings,
        "verdict": verdict_to_dict(verdict),
        "summary": summary,
    }
    write_run(run_dir, records, manifest)

    report_text = render_markdown_report(
        verdict,
        baseline_name=baseline_config.name,
        candidate_name=candidate_config.name,
        suite_path=str(suite_path),
        summary=summary,
    )
    markdown_path = output or run_dir / "report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(report_text, encoding="utf-8")
    json_path = json_output or run_dir / "report.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        export_json_report(
            verdict,
            baseline_name=baseline_config.name,
            candidate_name=candidate_config.name,
            suite_path=str(suite_path),
            summary=summary,
        ),
        encoding="utf-8",
    )

    _print_verdict(verdict, summary)
    console.print(f"Run saved to {run_dir} (report: {markdown_path})", markup=False)

    if verdict.verdict == "HOLD" and fail_on in {"hold", "not-ship"}:
        raise typer.Exit(code=10)
    if verdict.verdict == "INCONCLUSIVE" and fail_on == "not-ship":
        raise typer.Exit(code=11)


def _print_verdict(verdict: Any, summary: dict[str, Any]) -> None:
    table = Table(title=f"Verdict: {verdict.verdict}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row(
        "Pass rate (baseline / candidate)",
        f"{summary['pass_rates']['baseline']:.1%} / {summary['pass_rates']['candidate']:.1%}",
    )
    table.add_row("Overall delta", f"{verdict.overall_delta:.3f}")
    table.add_row("Overall CI", f"[{verdict.overall_ci[0]:.3f}, {verdict.overall_ci[1]:.3f}]")
    table.add_row("Required sample", str(verdict.required_sample_size))
    table.add_row("Blockers", str(len(verdict.blockers)))
    console.print(table)
    for blocker in verdict.blockers:
        console.print(f"  - {blocker}", markup=False)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    run_dir: Annotated[Path, typer.Argument(help="Run directory created by `swapcheck run`.")],
    output: Annotated[Path | None, typer.Option("--output", help="Write the report here.")] = None,
    format: Annotated[str, typer.Option("--format", help="markdown or json.")] = "markdown",
) -> None:
    """Render the full report from a stored run directory."""
    run_path = _resolve_path(run_dir)
    try:
        manifest = load_manifest(run_path)
        records = load_records(run_path)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    verdict = verdict_from_dict(manifest["verdict"])
    results = build_scenario_results(records)
    summary = build_summary(records, results)
    metrics = judge_metrics(records)
    if metrics is not None:
        verdict.metadata["judge_metrics"] = metrics

    baseline_name = str(manifest["baseline"]["name"])
    candidate_name = str(manifest["candidate"]["name"])
    suite = str(manifest["suite"])
    if format.lower() == "json":
        text = export_json_report(
            verdict,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            suite_path=suite,
            summary=summary,
        )
    elif format.lower() == "markdown":
        text = render_markdown_report(
            verdict,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            suite_path=suite,
            summary=summary,
        )
    else:
        raise _fail(f"Unknown format '{format}'. Use markdown or json.", code=2)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        console.print(f"Wrote report to {output}", markup=False)
    else:
        console.print(text, markup=False)


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def _labeled_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["trajectory_id"] for row in csv.DictReader(handle) if row.get("trajectory_id")}


def _render_trajectory(record: TrajectoryRecord, position: int, total: int) -> str:
    """Plain-text view of a trajectory. Never reveals which config produced it."""
    lines = [f"Trajectory {position}/{total}", ""]
    for turn in record.trajectory.turns:
        lines.append(f"USER: {turn.user_turn}")
        for entry in turn.tool_results:
            lines.append(
                f"  TOOL {entry.get('name')}({json.dumps(entry.get('arguments', {}))}) -> "
                f"{json.dumps(entry.get('result', {}))[:300]}"
            )
        lines.append(f"ASSISTANT: {turn.assistant_text or '(no text reply)'}")
        lines.append("")
    lines.append("FINAL DATABASE STATE:")
    for booking_id, booking in sorted(record.trajectory.final_db_snapshot.items()):
        lines.append(
            f"  {booking_id}: status={booking.get('status')} "
            f"refunded={booking.get('refunded_amount')} escalated={booking.get('escalated')}"
        )
    return "\n".join(lines)


def _ask_score(prompt: str) -> int:
    while True:
        value = typer.prompt(prompt, type=int)
        if 1 <= value <= 5:
            return int(value)
        console.print("Enter a number from 1 to 5.", markup=False)


@app.command()
def label(
    run_dir: Annotated[Path, typer.Option("--run", help="Run directory to label from.")],
    output: Annotated[Path, typer.Option("--output", help="Human labels CSV (appended).")],
    seed: Annotated[int, typer.Option("--seed", help="Seed for the shuffled order.")] = 0,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Stop after this many new labels.")
    ] = None,
) -> None:
    """Label stored trajectories by hand (blind to config and to the judge's scores)."""
    run_path = _resolve_path(run_dir)
    try:
        records = load_records(run_path)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    ordered = sorted(records, key=lambda r: r.trajectory_id)
    random.Random(seed).shuffle(ordered)
    already = _labeled_ids(output)
    if not output.exists() or output.stat().st_size == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(LABEL_TEMPLATE, encoding="utf-8")

    total = len(ordered)
    new_labels = 0
    for position, record in enumerate(ordered, start=1):
        if record.trajectory_id in already:
            continue
        if limit is not None and new_labels >= limit:
            break
        console.print(Panel(_render_trajectory(record, position, total), expand=False))
        action = typer.prompt("Label this one? [y = label, s = skip, q = quit]", default="y")
        action = action.strip().lower()
        if action == "q":
            break
        if action == "s":
            continue

        passed = typer.confirm("Should this trajectory PASS (correct and policy-compliant)?")
        helpfulness = _ask_score("Helpfulness (1-5)")
        groundedness = _ask_score("Groundedness (1-5)")
        tone = _ask_score("Tone (1-5)")
        off_policy = typer.confirm("Did the assistant promise something policy does not allow?")
        notes = typer.prompt("Notes (optional)", default="", show_default=False)

        with output.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    record.trajectory_id,
                    str(passed).lower(),
                    helpfulness,
                    groundedness,
                    tone,
                    str(off_policy).lower(),
                    notes,
                ]
            )
        new_labels += 1
        # The judge's opinion is shown only now, after the human label has been saved.
        if record.judge_result is not None:
            judge_info = record.judge_result
            if judge_info.get("judge_error"):
                console.print(f"Judge error: {judge_info['judge_error']}", markup=False)
            else:
                console.print(
                    f"Judge said: passed={judge_info.get('passed')} "
                    f"helpfulness={judge_info.get('helpfulness')} "
                    f"groundedness={judge_info.get('groundedness')} "
                    f"tone={judge_info.get('tone')} "
                    f"off_policy_promises={judge_info.get('off_policy_promises')}",
                    markup=False,
                )

    console.print(
        f"{new_labels} new labels saved; {len(_labeled_ids(output))} total in {output}",
        markup=False,
    )


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


@app.command()
def calibrate(
    labels: Annotated[Path, typer.Option("--labels", help="Human labels CSV.")],
    run_dir: Annotated[Path, typer.Option("--run", help="Run directory with judge results.")],
    seed: Annotated[int, typer.Option("--seed")] = 0,
    resamples: Annotated[int, typer.Option("--resamples", min=100)] = 2000,
) -> None:
    """Measure agreement between the LLM judge and your human labels."""
    run_path = _resolve_path(run_dir)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = summarize_calibration(
                labels, run_dir=run_path, bootstrap_samples=resamples, seed=seed
            )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise _fail(str(exc).strip("'\"")) from exc

    low, high = summary["kappa_ci_95"]
    table = Table(title="Judge vs human agreement")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Labeled trajectories", str(summary["n_samples"]))
    table.add_row("Raw agreement (pass/fail)", f"{summary['raw_agreement']:.1%}")
    if summary["kappa_defined"]:
        kappa_text = f"{summary['kappa']:.3f} (95% CI {low:.3f} to {high:.3f})"
    else:
        kappa_text = "undefined (no variation in the pass/fail labels)"
    table.add_row("Cohen's kappa (pass/fail)", kappa_text)
    table.add_row(
        "Kappa (off-policy promises)",
        f"{summary['kappa_off_policy']:.3f}"
        if summary["kappa_off_policy_defined"]
        else "undefined (no variation)",
    )
    table.add_row("MAE helpfulness", f"{summary['mae_helpfulness']:.2f}")
    table.add_row("MAE groundedness", f"{summary['mae_groundedness']:.2f}")
    table.add_row("MAE tone", f"{summary['mae_tone']:.2f}")
    console.print(table)
    if not summary["kappa_defined"]:
        console.print(
            "Warning: kappa is undefined because the pass/fail labels never vary. Label more "
            "trajectories, including failures, before trusting the judge.",
            markup=False,
        )
    elif summary["kappa"] < 0.60:
        console.print(
            "Warning: kappa is below 0.60. Do not let the judge affect verdicts.", markup=False
        )
    if summary["n_samples"] < 30:
        console.print(
            f"Warning: only {summary['n_samples']} labels (< 30); the agreement estimate "
            "is unreliable.",
            markup=False,
        )


if __name__ == "__main__":
    app()
