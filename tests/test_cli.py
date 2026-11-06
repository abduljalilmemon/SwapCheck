"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

import json
import os
import tempfile

from typer.testing import CliRunner

from swapcheck.cli.main import app

runner = CliRunner()


def test_cli_run_command_succeeds_for_mock_configs() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--baseline",
            "configs/baseline_mock.yaml",
            "--candidate",
            "configs/candidate_mock.yaml",
            "--suite",
            "suites/booking",
            "--runs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Verdict:" in result.output


def test_cli_run_command_writes_json_report_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            result = runner.invoke(
                app,
                [
                    "run",
                    "--baseline",
                    "configs/baseline_mock.yaml",
                    "--candidate",
                    "configs/candidate_mock.yaml",
                    "--judge",
                    "configs/judge_mock.yaml",
                    "--suite",
                    "suites/booking",
                    "--runs",
                    "1",
                    "--json-output",
                    "report.json",
                ],
            )
            assert result.exit_code == 0, result.output
            with open("report.json", encoding="utf-8") as handle:
                payload = json.load(handle)
            assert payload["verdict"] in {"SHIP", "HOLD", "INCONCLUSIVE"}
        finally:
            os.chdir(cwd)
