"""Tests for the calibration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from swapcheck.scoring.calibration import (
    cohen_kappa_confidence_interval,
    compute_cohens_kappa,
    write_label_template,
)


def test_cohens_kappa_matches_expected_value() -> None:
    """Cohen's kappa should match the standard two-label agreement formula."""
    human = [1, 1, 0, 0]
    judge = [1, 0, 1, 0]

    kappa = compute_cohens_kappa(human, judge)
    assert kappa == pytest.approx(0.0, abs=1e-6)


def test_confidence_interval_is_non_negative_for_agreement() -> None:
    """The confidence interval should bracket the observed agreement estimate."""
    lower, upper = cohen_kappa_confidence_interval([1, 1, 1, 1], [1, 1, 1, 1])
    assert lower <= upper
    assert lower <= 1.0
    assert upper >= 0.0


def test_label_template_is_written_with_expected_columns() -> None:
    """The human-label template should include the full calibration schema."""
    path = Path("/tmp/human_labels_template_test.csv")
    result = write_label_template(path)

    assert result.exists()
    header = result.read_text(encoding="utf-8").splitlines()[0]
    assert "trajectory_id,scenario_id,helpfulness,groundedness,tone" in header
    assert "off_policy_promises" in header
    assert "passed" in header
    assert "notes" in header
