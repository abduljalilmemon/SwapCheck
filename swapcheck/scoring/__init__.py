"""Deterministic and policy-based scoring for SwapCheck trajectories."""

from swapcheck.scoring.calibration import (
    bootstrap_cohen_kappa,
    cohen_kappa_confidence_interval,
    compute_cohens_kappa,
    summarize_calibration,
    write_label_template,
)
from swapcheck.scoring.deterministic import DeterministicScorer
from swapcheck.scoring.judge import JudgeScore, JudgeScorer
from swapcheck.scoring.models import PolicyViolation, ScoreBreakdown
from swapcheck.scoring.policy_checker import PolicyViolationDetector

__all__ = [
    "DeterministicScorer",
    "JudgeScore",
    "JudgeScorer",
    "PolicyViolation",
    "PolicyViolationDetector",
    "ScoreBreakdown",
    "bootstrap_cohen_kappa",
    "cohen_kappa_confidence_interval",
    "compute_cohens_kappa",
    "summarize_calibration",
    "write_label_template",
]
