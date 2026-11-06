"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

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
