"""Tests for the calibration helpers, Cohen's kappa, and human label joining."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from swapcheck.scoring.calibration import (
    compute_cohens_kappa,
    summarize_calibration,
    write_label_template,
)


def test_hand_computed_cohens_kappa() -> None:
    """Verify hand-computed Cohen's kappa:
    N=4:
    human = [1, 1, 0, 0]
    judge = [1, 1, 1, 0]
    Observed agreement = 3/4 = 0.75.
    Marginals:
    human: p(1)=0.5, p(0)=0.5
    judge: p(1)=0.75, p(0)=0.25
    Expected agreement = (0.5 * 0.75) + (0.5 * 0.25) = 0.375 + 0.125 = 0.50
    Kappa = (0.75 - 0.50) / (1.0 - 0.50) = 0.25 / 0.50 = 0.50
    """
    human = [1, 1, 0, 0]
    judge = [1, 1, 1, 0]
    kappa = compute_cohens_kappa(human, judge)
    assert kappa == pytest.approx(0.50, abs=1e-6)


def test_perfect_agreement_case(tmp_path: Path) -> None:
    """When human and judge agree on all rows, kappa should be 1.0."""
    csv_path = tmp_path / "human_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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
        for i in range(35):
            passed_val = "true" if i % 2 == 0 else "false"
            writer.writerow([f"traj_{i}", passed_val, "4", "5", "4", "false", "ok"])

    judge_results = {
        f"traj_{i}": {
            "passed": (i % 2 == 0),
            "helpfulness": 4,
            "groundedness": 5,
            "tone": 4,
            "off_policy_promises": False,
        }
        for i in range(35)
    }

    summary = summarize_calibration(csv_path, judge_results=judge_results, seed=42)
    assert summary["n_samples"] == 35
    assert summary["raw_agreement"] == pytest.approx(1.0, abs=1e-6)
    assert summary["kappa"] == pytest.approx(1.0, abs=1e-6)
    assert summary["mae_helpfulness"] == pytest.approx(0.0, abs=1e-6)
    assert summary["mae_groundedness"] == pytest.approx(0.0, abs=1e-6)
    assert summary["mae_tone"] == pytest.approx(0.0, abs=1e-6)
    assert summary["kappa_ci_95"][0] == pytest.approx(1.0, abs=1e-6)
    assert summary["kappa_ci_95"][1] == pytest.approx(1.0, abs=1e-6)


def test_disagreement_case_with_kappa_below_one(tmp_path: Path) -> None:
    """When judge and human disagree on some rows, kappa should be clearly below 1.0."""
    csv_path = tmp_path / "human_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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
        # 40 rows: 20 pass, 20 fail
        for i in range(40):
            writer.writerow(
                [
                    f"traj_{i}",
                    "true" if i < 20 else "false",
                    "4",
                    "4",
                    "4",
                    "false",
                    "notes",
                ]
            )

    # Judge agrees on only 25 of 40 rows
    judge_results = {}
    for i in range(40):
        # Disagree on items 10-24
        j_pass = (i >= 20) if (10 <= i < 25) else (i < 20)
        judge_results[f"traj_{i}"] = {
            "passed": j_pass,
            "helpfulness": 3 if i % 2 == 0 else 5,
            "groundedness": 4,
            "tone": 4,
            "off_policy_promises": False,
        }

    summary = summarize_calibration(csv_path, judge_results=judge_results, seed=0)
    assert summary["n_samples"] == 40
    assert 0.0 <= summary["kappa"] < 1.0
    assert summary["raw_agreement"] < 1.0
    assert summary["mae_helpfulness"] > 0.0
    assert summary["kappa_ci_95"][0] <= summary["kappa"] <= summary["kappa_ci_95"][1]


def test_absent_judge_result_raises(tmp_path: Path) -> None:
    """If a human label has a trajectory_id with no corresponding judge result, raise KeyError."""
    csv_path = tmp_path / "human_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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
        writer.writerow(["traj_present", "true", "5", "5", "5", "false", "ok"])
        writer.writerow(["traj_missing", "false", "2", "2", "2", "false", "missing"])

    judge_results = {
        "traj_present": {
            "passed": True,
            "helpfulness": 5,
            "groundedness": 5,
            "tone": 5,
            "off_policy_promises": False,
        }
    }

    with pytest.raises(
        KeyError, match="No judge result found for labeled trajectory ID: 'traj_missing'"
    ):
        summarize_calibration(csv_path, judge_results=judge_results)


def test_no_matching_ids_raises(tmp_path: Path) -> None:
    """If no IDs match or empty, raise ValueError."""
    csv_path = tmp_path / "empty_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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

    with pytest.raises(ValueError, match="No calibration rows found"):
        summarize_calibration(csv_path, judge_results={"a": {}})


def test_label_template_columns(tmp_path: Path) -> None:
    """write_label_template should write expected column headers."""
    dest = tmp_path / "template.csv"
    written = write_label_template(dest)
    header = written.read_text(encoding="utf-8").splitlines()[0]
    expected = "trajectory_id,human_passed,human_helpfulness,human_groundedness,human_tone,human_off_policy_promise,notes"
    assert header == expected


def test_kappa_is_flagged_undefined_when_labels_never_vary(tmp_path: Path) -> None:
    """If human and judge always say 'pass', kappa is undefined, not 'perfect agreement'."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "trajectory_id,human_passed,human_helpfulness,human_groundedness,human_tone,"
        "human_off_policy_promise,notes\n"
        "t1,true,4,5,4,false,\nt2,true,4,5,4,false,\nt3,true,4,5,4,false,\n",
        encoding="utf-8",
    )
    judge_results = {
        tid: {
            "passed": True,
            "helpfulness": 4,
            "groundedness": 5,
            "tone": 4,
            "off_policy_promises": False,
        }
        for tid in ("t1", "t2", "t3")
    }
    summary = summarize_calibration(csv_path, judge_results=judge_results, seed=0)
    assert summary["kappa_defined"] is False
    assert summary["kappa_off_policy_defined"] is False
    assert summary["raw_agreement"] == 1.0
