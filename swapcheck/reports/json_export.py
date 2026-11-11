"""JSON export helpers for SwapCheck evaluations."""

from __future__ import annotations

import json

from swapcheck.verdict.models import VerdictResult


def export_json_report(
    verdict: VerdictResult,
    *,
    baseline_name: str,
    candidate_name: str,
    suite_path: str,
) -> str:
    """Render a machine-readable summary of the verdict."""
    payload: dict[str, object] = {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "suite": suite_path,
        "verdict": verdict.verdict,
        "overall_delta": verdict.overall_delta,
        "overall_ci": list(verdict.overall_ci),
        "required_sample_size": verdict.required_sample_size,
        "reasons": verdict.reasons,
        "blockers": verdict.blockers,
        "notes": verdict.notes,
        "underpowered_categories": verdict.underpowered_categories,
        "category_results": {
            name: {
                "scenario_count": result.scenario_count,
                "estimate": result.estimate,
                "ci": list(result.ci),
                "powered": result.powered,
                "status": result.status,
                "reasons": result.reasons,
                "blockers": result.blockers,
                "notes": result.notes,
            }
            for name, result in sorted(verdict.category_results.items())
        },
        "cost_summary": verdict.cost_summary,
        "latency_summary": verdict.latency_summary,
        "token_totals": verdict.token_totals,
        "metadata": verdict.metadata,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
