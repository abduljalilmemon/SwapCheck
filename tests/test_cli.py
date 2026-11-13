"""End-to-end tests for the Typer CLI, run fully offline with the scripted mock providers.

These run the real 60-scenario suite through `swapcheck run`, so they exercise the runner,
the scorers, the paired bootstrap, the verdict engine, the stored run directory and the
label / calibrate / report commands together.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from swapcheck.cli.main import app
from swapcheck.runs import TrajectoryRecord, load_records

runner = CliRunner()
SUITE = "suites/booking"
FAST = ["--bootstrap-resamples", "300", "--no-cache"]


def _run(
    out: Path,
    candidate: str,
    *extra: str,
    runs: int = 3,
    suite: str = SUITE,
    judge: bool = True,
) -> tuple[object, Path]:
    args = [
        "run",
        "--baseline",
        "configs/baseline_mock.yaml",
        "--candidate",
        f"configs/{candidate}.yaml",
        "--suite",
        suite,
        "--runs",
        str(runs),
        "--output-dir",
        str(out),
        *FAST,
    ]
    if judge:
        args += ["--judge", "configs/judge_mock.yaml"]
    args += list(extra)
    return runner.invoke(app, args), out


def _manifest(out: Path) -> dict:
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def identical_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    result, out = _run(tmp_path_factory.mktemp("identical"), "baseline_mock")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    return out


@pytest.fixture(scope="module")
def degraded_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    result, out = _run(tmp_path_factory.mktemp("degraded"), "candidate_mock")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    return out


@pytest.fixture(scope="module")
def unsafe_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    result, out = _run(tmp_path_factory.mktemp("unsafe"), "candidate_mock_unsafe")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    return out


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_identical_candidate_ships(identical_run: Path) -> None:
    verdict = _manifest(identical_run)["verdict"]
    assert verdict["verdict"] == "SHIP"
    assert verdict["overall_delta"] == 0.0
    assert verdict["blockers"] == []


def test_degraded_candidate_is_held_for_category_degradation(degraded_run: Path) -> None:
    verdict = _manifest(degraded_run)["verdict"]
    assert verdict["verdict"] == "HOLD"
    assert verdict["overall_delta"] < 0.0
    refund = verdict["category_results"]["refund_policy_edge_cases"]
    assert refund["status"] == "FAIL"
    assert any("statistically significant degradation" in b for b in verdict["blockers"])


def test_unsafe_candidate_is_held_for_new_critical_violations(unsafe_run: Path) -> None:
    manifest = _manifest(unsafe_run)
    verdict = manifest["verdict"]
    assert verdict["verdict"] == "HOLD"
    assert any("New critical policy violations" in b for b in verdict["blockers"])
    records = load_records(unsafe_run)
    critical = [
        v
        for r in records
        if r.role == "candidate"
        for v in r.score.policy_violations
        if v.severity == "critical"
    ]
    assert critical, "the unsafe candidate should produce critical violations"
    assert not any(
        v.severity == "critical"
        for r in records
        if r.role == "baseline"
        for v in r.score.policy_violations
    )


def test_small_suite_is_inconclusive(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    for category in ("simple_lookup", "cancellation"):
        (suite / category).mkdir(parents=True)
        for source in sorted((Path(SUITE) / category).glob("*.yaml"))[:3]:
            shutil.copy(source, suite / category / source.name)
    result, out = _run(tmp_path / "run", "baseline_mock", suite=str(suite))
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    verdict = _manifest(out)["verdict"]
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert any("sample size below minimum" in b for b in verdict["blockers"])
    assert verdict["required_sample_size"] >= 50


def test_too_few_repetitions_prevents_ship(tmp_path: Path) -> None:
    result, out = _run(tmp_path / "run", "baseline_mock", runs=2)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    verdict = _manifest(out)["verdict"]
    assert verdict["verdict"] != "SHIP"
    assert any("Repetitions below minimum" in b for b in verdict["blockers"])


def test_min_repetitions_flag_is_honoured(tmp_path: Path) -> None:
    result, out = _run(tmp_path / "run", "baseline_mock", "--min-repetitions", "2", runs=2)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert _manifest(out)["verdict"]["verdict"] == "SHIP"


def test_cli_flags_are_recorded_in_manifest(tmp_path: Path) -> None:
    result, out = _run(
        tmp_path / "run",
        "baseline_mock",
        "--margin",
        "0.07",
        "--category-margin",
        "0.2",
        "--min-scenarios",
        "40",
        "--min-scenarios-per-category",
        "6",
        "--seed",
        "7",
        "--concurrency",
        "2",
    )
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    settings = _manifest(out)["settings"]
    assert settings["margin"] == 0.07
    assert settings["category_margin"] == 0.2
    assert settings["min_scenarios"] == 40
    assert settings["min_scenarios_per_category"] == 6
    assert settings["seed"] == 7
    assert settings["concurrency"] == 2
    assert settings["bootstrap_resamples"] == 300


# ---------------------------------------------------------------------------
# run directory, tokens, judge storage
# ---------------------------------------------------------------------------


def test_run_directory_layout(identical_run: Path) -> None:
    assert (identical_run / "manifest.json").is_file()
    assert (identical_run / "report.md").is_file()
    assert (identical_run / "report.json").is_file()
    files = sorted((identical_run / "trajectories").glob("*.json"))
    assert len(files) == 60 * 3 * 2
    assert len({f.stem for f in files}) == len(files)
    record = TrajectoryRecord.model_validate_json(files[0].read_text(encoding="utf-8"))
    assert record.trajectory_id == files[0].stem
    assert record.trajectory.turns
    assert record.trajectory.final_db_snapshot


def test_token_accounting_splits_prompt_and_completion(identical_run: Path) -> None:
    totals = _manifest(identical_run)["verdict"]["token_totals"]
    for role in ("baseline", "candidate"):
        assert totals[role]["prompt_tokens"] > 0
        assert totals[role]["completion_tokens"] > 0
        assert totals[role]["total_tokens"] == (
            totals[role]["prompt_tokens"] + totals[role]["completion_tokens"]
        )
        assert totals[role]["prompt_tokens"] > totals[role]["completion_tokens"]


def test_judge_results_are_stored_and_reported(degraded_run: Path) -> None:
    records = load_records(degraded_run)
    assert all(r.judge_result is not None for r in records)
    assert all(not r.judge_result.get("judge_error") for r in records if r.judge_result)
    report = (degraded_run / "report.md").read_text(encoding="utf-8")
    assert "Judge metrics" in report
    assert "judge-mock" in report
    # Without --judge-affects-pass the judge must not change the verdict inputs.
    assert all(r.passed == r.score.passed for r in records)


def test_report_contains_pass_rates_cost_and_worst_failures(degraded_run: Path) -> None:
    report = (degraded_run / "report.md").read_text(encoding="utf-8")
    assert "### Pass rates" in report
    assert "Cost per successful task" in report
    assert "median / p95" in report
    assert "Prompt tokens" in report
    assert "Worst failing candidate trajectories" in report


def test_judge_affects_pass_changes_pass_decisions(tmp_path: Path) -> None:
    result, out = _run(tmp_path / "run", "candidate_mock", "--judge-affects-pass")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    assert _manifest(out)["settings"]["judge_affects_pass"] is True


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


def test_judge_affects_pass_requires_a_judge(tmp_path: Path) -> None:
    result, _ = _run(tmp_path / "run", "baseline_mock", "--judge-affects-pass", judge=False)
    assert result.exit_code == 2  # type: ignore[attr-defined]
    assert "requires --judge" in result.output  # type: ignore[attr-defined]


def test_judge_must_not_share_a_model_with_baseline(tmp_path: Path) -> None:
    baseline = yaml.safe_load(Path("configs/baseline_mock.yaml").read_text(encoding="utf-8"))
    judge_cfg = {**baseline, "name": "judge"}
    judge_path = tmp_path / "judge_same_model.yaml"
    judge_path.write_text(yaml.safe_dump(judge_cfg), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "run",
            "--baseline",
            "configs/baseline_mock.yaml",
            "--candidate",
            "configs/candidate_mock.yaml",
            "--suite",
            SUITE,
            "--judge",
            str(judge_path),
            "--output-dir",
            str(tmp_path / "run"),
            *FAST,
        ],
    )
    assert result.exit_code == 2
    assert "same as the baseline model" in result.output


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------


def test_report_command_rerenders_stored_run(identical_run: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(identical_run)])
    assert result.exit_code == 0, result.output
    assert "Verdict: SHIP" in result.output

    target = tmp_path / "again.json"
    result = runner.invoke(
        app, ["report", str(identical_run), "--format", "json", "--output", str(target)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["verdict"] == "SHIP"
    assert payload["summary"]["pass_rates"]["candidate"] == 1.0


def test_report_command_errors_cleanly(identical_run: Path, tmp_path: Path) -> None:
    missing = runner.invoke(app, ["report", str(tmp_path / "nope")])
    assert missing.exit_code == 1
    bad_format = runner.invoke(app, ["report", str(identical_run), "--format", "pdf"])
    assert bad_format.exit_code == 2


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------

ONE_LABEL = "y\ny\n5\n4\n4\nn\nlooked fine\n"


def test_label_flow_is_blind_resumable_and_shows_judge_after_input(
    degraded_run: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.csv"
    result = runner.invoke(
        app,
        ["label", "--run", str(degraded_run), "--output", str(labels), "--limit", "2"],
        input=ONE_LABEL * 2,
    )
    assert result.exit_code == 0, result.output
    # The human never sees which config produced a trajectory.
    assert "__baseline__" not in result.output
    assert "__candidate__" not in result.output
    # The judge's opinion is revealed only after the human answered.
    assert result.output.index("Notes (optional)") < result.output.index("Judge said")

    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["human_passed"] == "true"
    assert rows[0]["human_helpfulness"] == "5"
    assert rows[0]["notes"] == "looked fine"

    resumed = runner.invoke(
        app,
        ["label", "--run", str(degraded_run), "--output", str(labels), "--limit", "1"],
        input=ONE_LABEL,
    )
    assert resumed.exit_code == 0, resumed.output
    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    assert len(rows) == 3
    assert len({r["trajectory_id"] for r in rows}) == 3, "resume must skip already-labeled ids"


def test_label_skip_and_quit(degraded_run: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    result = runner.invoke(
        app,
        ["label", "--run", str(degraded_run), "--output", str(labels)],
        input="s\nq\n",
    )
    assert result.exit_code == 0, result.output
    assert list(csv.DictReader(labels.open(encoding="utf-8"))) == []


def test_label_rejects_out_of_range_scores(degraded_run: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    result = runner.invoke(
        app,
        ["label", "--run", str(degraded_run), "--output", str(labels), "--limit", "1"],
        input="y\ny\n9\n5\n4\n4\nn\n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Enter a number from 1 to 5" in result.output
    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    assert rows[0]["human_helpfulness"] == "5"


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def _write_labels(path: Path, run: Path, *, count: int, flip_every: int) -> None:
    """Synthetic TEST labels: follow the deterministic result, flipping every Nth one."""
    records = sorted(load_records(run), key=lambda r: r.trajectory_id)
    step = max(1, len(records) // count)
    chosen = records[::step][:count]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trajectory_id",
                "human_passed",
                "human_helpfulness",
                "human_groundedness",
                "human_tone",
                "human_off_policy_promise",
                "notes",
            ]
        )
        for index, record in enumerate(chosen):
            passed = record.passed if (index + 1) % flip_every else not record.passed
            writer.writerow([record.trajectory_id, str(passed).lower(), 4, 5, 4, "false", "test"])


def test_calibrate_reports_disagreement_not_perfect_agreement(
    degraded_run: Path, tmp_path: Path
) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(labels, degraded_run, count=40, flip_every=4)
    result = runner.invoke(app, ["calibrate", "--labels", str(labels), "--run", str(degraded_run)])
    assert result.exit_code == 0, result.output
    assert "Cohen's kappa" in result.output
    assert "1.000" not in result.output.split("Cohen's kappa")[1].splitlines()[0]
    assert "Warning: kappa is below 0.60" in result.output
    assert "only" not in result.output  # 40 labels: no small-sample warning


def test_calibrate_warns_on_few_labels(degraded_run: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(labels, degraded_run, count=10, flip_every=5)
    result = runner.invoke(app, ["calibrate", "--labels", str(labels), "--run", str(degraded_run)])
    assert result.exit_code == 0, result.output
    assert "only 10 labels" in result.output


def test_calibrate_fails_without_judge_results(tmp_path: Path) -> None:
    result, out = _run(tmp_path / "run", "baseline_mock", judge=False)
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    labels = tmp_path / "labels.csv"
    _write_labels(labels, out, count=10, flip_every=5)
    outcome = runner.invoke(app, ["calibrate", "--labels", str(labels), "--run", str(out)])
    assert outcome.exit_code == 1
    assert "judge" in outcome.output.lower()


def test_calibrate_fails_for_unknown_trajectory_ids(degraded_run: Path, tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "trajectory_id,human_passed,human_helpfulness,human_groundedness,human_tone,"
        "human_off_policy_promise,notes\nnot-a-real-id,true,4,4,4,false,\n",
        encoding="utf-8",
    )
    outcome = runner.invoke(app, ["calibrate", "--labels", str(labels), "--run", str(degraded_run)])
    assert outcome.exit_code == 1


# ---------------------------------------------------------------------------
# CI gating
# ---------------------------------------------------------------------------


def test_fail_on_exit_codes(tmp_path: Path) -> None:
    unsafe, _ = _run(tmp_path / "a", "candidate_mock_unsafe", "--fail-on", "hold")
    assert unsafe.exit_code == 10  # type: ignore[attr-defined]

    ship, _ = _run(tmp_path / "b", "baseline_mock", "--fail-on", "not-ship")
    assert ship.exit_code == 0  # type: ignore[attr-defined]

    suite = tmp_path / "suite" / "simple_lookup"
    suite.mkdir(parents=True)
    for source in sorted((Path(SUITE) / "simple_lookup").glob("*.yaml"))[:3]:
        shutil.copy(source, suite / source.name)
    small, _ = _run(
        tmp_path / "c", "baseline_mock", "--fail-on", "not-ship", suite=str(tmp_path / "suite")
    )
    assert small.exit_code == 11  # type: ignore[attr-defined]

    inconclusive_ok, _ = _run(
        tmp_path / "d", "baseline_mock", "--fail-on", "hold", suite=str(tmp_path / "suite")
    )
    assert inconclusive_ok.exit_code == 0  # type: ignore[attr-defined]


def test_fail_on_rejects_unknown_values(tmp_path: Path) -> None:
    result, _ = _run(tmp_path / "run", "baseline_mock", "--fail-on", "always")
    assert result.exit_code == 2  # type: ignore[attr-defined]
