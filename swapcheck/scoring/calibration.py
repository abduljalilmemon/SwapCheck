"""Human calibration helpers for judge agreement and labeling workflows."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

LABEL_TEMPLATE = (
    "trajectory_id,scenario_id,helpfulness,groundedness,tone,off_policy_promises,passed,notes\n"
)


def write_label_template(path: str | Path) -> Path:
    """Create a CSV template for human calibration labels. Returns the written path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(LABEL_TEMPLATE, encoding="utf-8")
    return output_path


def _to_binary_sequence(labels: Sequence[int | bool | str]) -> list[int]:
    """Normalize labels to a binary 0/1 sequence when possible."""
    result: list[int] = []
    for value in labels:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in {"true", "yes", "y", "1"}:
                result.append(1)
            elif candidate in {"false", "no", "n", "0"}:
                result.append(0)
            else:
                try:
                    result.append(int(float(candidate)))
                except ValueError:
                    result.append(0)
        elif isinstance(value, bool):
            result.append(int(value))
        else:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                result.append(0)
    return result


def compute_cohens_kappa(
    human_labels: Sequence[int | bool | str],
    judge_labels: Sequence[int | bool | str],
) -> float:
    """Compute Cohen's kappa for two binary or categorical label sets."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must have the same length.")
    if not human_labels:
        return 0.0

    human = _to_binary_sequence(human_labels)
    judge = _to_binary_sequence(judge_labels)
    n = len(human)
    observed = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / n
    classes = sorted(set(human + judge))
    if not classes:
        return 0.0
    if len(classes) == 1:
        return 1.0 if observed == 1.0 else 0.0

    class_counts_human = {cls: human.count(cls) for cls in classes}
    class_counts_judge = {cls: judge.count(cls) for cls in classes}
    expected = sum((class_counts_human[cls] / n) * (class_counts_judge[cls] / n) for cls in classes)
    if math.isclose(expected, 1.0):
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def cohen_kappa_confidence_interval(
    human_labels: Sequence[int | bool | str],
    judge_labels: Sequence[int | bool | str],
    *,
    z_score: float = 1.96,
) -> tuple[float, float]:
    """Return the analytical 95% confidence interval around Cohen's kappa."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must have the same length.")
    if not human_labels:
        return (0.0, 0.0)

    human = _to_binary_sequence(human_labels)
    judge = _to_binary_sequence(judge_labels)
    n = len(human)
    observed = sum(1 for h, j in zip(human, judge, strict=True) if h == j) / n
    classes = sorted(set(human + judge))
    if len(classes) <= 1:
        kappa = 1.0 if observed == 1.0 else 0.0
        return (kappa, kappa)

    class_counts_human = {cls: human.count(cls) for cls in classes}
    class_counts_judge = {cls: judge.count(cls) for cls in classes}
    expected = sum((class_counts_human[cls] / n) * (class_counts_judge[cls] / n) for cls in classes)
    if math.isclose(expected, 1.0):
        kappa = 1.0 if observed == 1.0 else 0.0
        return (kappa, kappa)

    kappa = (observed - expected) / (1.0 - expected)
    standard_error = math.sqrt((observed * (1.0 - observed)) / (n * (1.0 - expected) ** 2))
    margin = z_score * standard_error
    return (float(kappa - margin), float(kappa + margin))


def bootstrap_cohen_kappa(
    human_labels: Sequence[int | bool | str],
    judge_labels: Sequence[int | bool | str],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap the variance of Cohen's kappa by resampling the paired labels."""
    if len(human_labels) != len(judge_labels):
        raise ValueError("human_labels and judge_labels must have the same length.")
    if not human_labels:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(human_labels)
    sampled_kappas: list[float] = []
    for _ in range(max(1, resamples)):
        indices = rng.integers(0, n, size=n)
        sampled_human = [human_labels[int(index)] for index in indices]
        sampled_judge = [judge_labels[int(index)] for index in indices]
        sampled_kappas.append(compute_cohens_kappa(sampled_human, sampled_judge))

    lower, upper = (
        float(np.quantile(sampled_kappas, 0.025)),
        float(np.quantile(sampled_kappas, 0.975)),
    )
    return (lower, upper)


def summarize_calibration(
    labels_path: str | Path,
    *,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """Load a CSV of labels and summarize agreement metrics."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration labels not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No calibration rows found in {path}.")

    # Prefer explicit human/judge columns if present, otherwise use the scalar pass labels.
    human_values = []
    judge_values = []
    for row in rows:
        human_value = row.get("human_passed")
        judge_value = row.get("judge_passed")
        if human_value is not None and judge_value is not None:
            human_values.append(human_value)
            judge_values.append(judge_value)
            continue
        pass_value = row.get("passed")
        if pass_value is not None:
            human_values.append(pass_value)
            judge_values.append(pass_value)

    if not human_values:
        raise ValueError(f"No comparable label columns found in {path}.")

    kappa = compute_cohens_kappa(human_values, judge_values)
    ci_lower, ci_upper = cohen_kappa_confidence_interval(human_values, judge_values)
    boot_lower, boot_upper = bootstrap_cohen_kappa(
        human_values,
        judge_values,
        resamples=bootstrap_samples,
    )
    return {
        "n_samples": len(human_values),
        "kappa": float(kappa),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "bootstrap_ci_95": [float(boot_lower), float(boot_upper)],
    }
