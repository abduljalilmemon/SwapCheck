"""Human calibration helpers for judge agreement and labeling workflows."""

from __future__ import annotations

import csv
import json
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

LABEL_TEMPLATE = "trajectory_id,human_passed,human_helpfulness,human_groundedness,human_tone,human_off_policy_promise,notes\n"


def write_label_template(path: str | Path) -> Path:
    """Create a CSV template for human calibration labels. Returns the written path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(LABEL_TEMPLATE, encoding="utf-8")
    return output_path


def _to_binary(value: int | bool | str | None) -> int:
    """Normalize a value to binary 0 or 1."""
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in {"true", "yes", "y", "1", "pass"}:
            return 1
        return 0
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_binary_sequence(labels: Sequence[int | bool | str]) -> list[int]:
    """Normalize labels to a binary 0/1 sequence."""
    return [_to_binary(val) for val in labels]


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

    p_human_1 = sum(human) / n
    p_human_0 = 1.0 - p_human_1
    p_judge_1 = sum(judge) / n
    p_judge_0 = 1.0 - p_judge_1

    expected = (p_human_1 * p_judge_1) + (p_human_0 * p_judge_0)

    if math.isclose(expected, 1.0):
        return 1.0 if observed == 1.0 else 0.0
    return float((observed - expected) / (1.0 - expected))


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

    lower = float(np.quantile(sampled_kappas, 0.025))
    upper = float(np.quantile(sampled_kappas, 0.975))
    return (lower, upper)


def load_judge_results_from_run(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load judge results from a stored run directory (manifest.json or trajectories/*.json)."""
    run_path = Path(run_dir)
    results: dict[str, dict[str, Any]] = {}

    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            trajectories = manifest.get("trajectories", [])
            for entry in trajectories:
                t_id = entry.get("trajectory_id")
                judge_result = entry.get("judge_result") or entry.get("judge_score")
                if t_id and judge_result:
                    results[t_id] = judge_result
        except Exception:
            pass

    traj_dir = run_path / "trajectories"
    if traj_dir.is_dir():
        for file in traj_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                t_id = data.get("trajectory_id") or file.stem
                judge_result = data.get("judge_result") or data.get("judge_score")
                if t_id and judge_result:
                    results[t_id] = judge_result
            except Exception:
                pass

    return results


def summarize_calibration(
    labels_path: str | Path,
    run_dir: str | Path | None = None,
    *,
    judge_results: dict[str, Any] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Join human labels with stored judge results on trajectory_id and compute calibration metrics."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration labels not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No calibration rows found in {path}.")

    resolved_judge_results: dict[str, dict[str, Any]] = {}
    if run_dir is not None:
        resolved_judge_results.update(load_judge_results_from_run(run_dir))
    if judge_results is not None:
        for k, v in judge_results.items():
            if hasattr(v, "model_dump"):
                resolved_judge_results[k] = v.model_dump()
            elif isinstance(v, dict):
                resolved_judge_results[k] = v

    if not resolved_judge_results:
        raise ValueError("No judge results provided or found in the run directory.")

    human_passed: list[int] = []
    judge_passed: list[int] = []
    human_off_policy: list[int] = []
    judge_off_policy: list[int] = []
    abs_diff_helpfulness: list[float] = []
    abs_diff_groundedness: list[float] = []
    abs_diff_tone: list[float] = []

    matched_ids = 0
    for row in rows:
        t_id = row.get("trajectory_id", "").strip()
        if not t_id:
            continue

        if t_id not in resolved_judge_results:
            raise KeyError(
                f"No judge result found for labeled trajectory ID: '{t_id}'. Ensure the run contains judge scores for all labeled trajectories."
            )

        judge_data = resolved_judge_results[t_id]
        # If judge produced an error, exclude from agreement statistics
        if judge_data.get("judge_error"):
            continue

        matched_ids += 1
        h_pass = _to_binary(row.get("human_passed", row.get("passed")))
        j_pass = _to_binary(judge_data.get("passed"))
        human_passed.append(h_pass)
        judge_passed.append(j_pass)

        h_off = _to_binary(row.get("human_off_policy_promise", row.get("off_policy_promises")))
        j_off = _to_binary(
            judge_data.get("off_policy_promises", judge_data.get("off_policy_promise"))
        )
        human_off_policy.append(h_off)
        judge_off_policy.append(j_off)

        # 1-5 score MAE
        try:
            h_help = float(row.get("human_helpfulness", row.get("helpfulness", 3)))
            j_help = float(judge_data.get("helpfulness", 3))
            abs_diff_helpfulness.append(abs(h_help - j_help))
        except (ValueError, TypeError):
            pass

        try:
            h_ground = float(row.get("human_groundedness", row.get("groundedness", 3)))
            j_ground = float(judge_data.get("groundedness", 3))
            abs_diff_groundedness.append(abs(h_ground - j_ground))
        except (ValueError, TypeError):
            pass

        try:
            h_tone = float(row.get("human_tone", row.get("tone", 3)))
            j_tone = float(judge_data.get("tone", 3))
            abs_diff_tone.append(abs(h_tone - j_tone))
        except (ValueError, TypeError):
            pass

    if matched_ids == 0 or not human_passed:
        raise ValueError(
            f"No matching trajectory IDs found between human labels in '{path}' and judge results."
        )

    n_samples = len(human_passed)
    raw_agreement = (
        sum(1 for h, j in zip(human_passed, judge_passed, strict=True) if h == j) / n_samples
    )
    kappa = compute_cohens_kappa(human_passed, judge_passed)
    boot_lower, boot_upper = bootstrap_cohen_kappa(
        human_passed,
        judge_passed,
        resamples=bootstrap_samples,
        seed=seed,
    )

    kappa_off_policy = compute_cohens_kappa(human_off_policy, judge_off_policy)
    # Kappa is undefined when both raters only ever use the same single label: agreement
    # cannot then be separated from chance, so callers must not read the number as "perfect".
    kappa_defined = len(set(human_passed) | set(judge_passed)) > 1
    kappa_off_policy_defined = len(set(human_off_policy) | set(judge_off_policy)) > 1

    mae_helpfulness = float(np.mean(abs_diff_helpfulness)) if abs_diff_helpfulness else 0.0
    mae_groundedness = float(np.mean(abs_diff_groundedness)) if abs_diff_groundedness else 0.0
    mae_tone = float(np.mean(abs_diff_tone)) if abs_diff_tone else 0.0

    if kappa_defined and kappa < 0.60:
        msg = f"Warning: Low judge agreement! Cohen's kappa is {kappa:.3f} (< 0.60). Judge calibration is insufficient."
        warnings.warn(msg, UserWarning, stacklevel=2)
        print(msg)

    if n_samples < 30:
        msg = f"Warning: Small calibration sample size! Only {n_samples} labels evaluated (< 30). Agreement metrics may be unreliable."
        warnings.warn(msg, UserWarning, stacklevel=2)
        print(msg)

    return {
        "n_samples": n_samples,
        "raw_agreement": float(raw_agreement),
        "kappa": float(kappa),
        "kappa_ci_95": [float(boot_lower), float(boot_upper)],
        "kappa_defined": kappa_defined,
        "kappa_off_policy": float(kappa_off_policy),
        "kappa_off_policy_defined": kappa_off_policy_defined,
        "mae_helpfulness": mae_helpfulness,
        "mae_groundedness": mae_groundedness,
        "mae_tone": mae_tone,
    }
