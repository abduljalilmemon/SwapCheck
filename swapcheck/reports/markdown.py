"""Markdown rendering for evaluation results."""

from __future__ import annotations

from typing import Any

from swapcheck.verdict.models import VerdictResult


def _money(value: float) -> str:
    return f"${value:.4f}"


def render_markdown_report(
    verdict: VerdictResult,
    *,
    baseline_name: str,
    candidate_name: str,
    suite_path: str,
    summary: dict[str, Any] | None = None,
) -> str:
    """Render a concise markdown summary of a verdict result."""
    lines: list[str] = [
        "# SwapCheck Evaluation Report",
        "",
        f"- Baseline: {baseline_name}",
        f"- Candidate: {candidate_name}",
        f"- Scenario suite: {suite_path}",
        "",
        f"## Verdict: {verdict.verdict}",
        "",
    ]

    if verdict.blockers:
        lines.append("### Blockers")
        for reason in verdict.blockers:
            lines.append(f"- {reason}")
        lines.append("")
    if verdict.notes:
        lines.append("### Notes")
        for reason in verdict.notes:
            lines.append(f"- {reason}")
        lines.append("")

    lines.extend(
        [
            "### Summary",
            f"- Overall delta: {verdict.overall_delta:.3f}",
            f"- Overall CI: [{verdict.overall_ci[0]:.3f}, {verdict.overall_ci[1]:.3f}]",
            f"- Required sample size: {verdict.required_sample_size if verdict.required_sample_size is not None else 'n/a (verdict is decided)'}",
            "",
            "### Category results",
            "",
            "| Category | Status | Estimate | CI | Powered |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )

    if not verdict.category_results:
        lines.append("| — | — | — | — | — |")
    else:
        for name, result in sorted(verdict.category_results.items()):
            ci_text = f"[{result.ci[0]:.3f}, {result.ci[1]:.3f}]" if result.powered else "—"
            lines.append(
                f"| {name} | {result.status} | {result.estimate:.3f} | {ci_text} | {str(result.powered).lower()} |"
            )

    if "judge_metrics" in verdict.metadata:
        judge_info = verdict.metadata["judge_metrics"]
        lines.append("")
        lines.append("### Judge metrics")
        lines.append("")
        if "model" in judge_info:
            lines.append(f"- Judge model: {judge_info['model']}")
        if "total_cost" in judge_info:
            lines.append(f"- Judge total cost: ${float(judge_info['total_cost']):.4f}")
        if "total_errors" in judge_info:
            lines.append(f"- Judge errors: {judge_info['total_errors']}")
        lines.append("")
        lines.append(
            "| Config | Category | Helpfulness | Groundedness | Tone | Off-policy rate | Judge pass rate | Errors |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in judge_info.get("rows", []):
            lines.append(
                f"| {row['config']} | {row['category']} | {row['helpfulness']:.2f} | {row['groundedness']:.2f} | "
                f"{row['tone']:.2f} | {row['off_policy_rate']:.1%} | {row['judge_pass_rate']:.1%} | {row['errors']} |"
            )

    if summary and summary.get("pass_rates"):
        lines.extend(["", "### Pass rates", ""])
        lines.append("| Scope | Baseline | Candidate |")
        lines.append("| --- | ---: | ---: |")
        lines.append(
            f"| **Overall** | {summary['pass_rates']['baseline']:.1%} | "
            f"{summary['pass_rates']['candidate']:.1%} |"
        )
        for category, rates in summary.get("category_pass_rates", {}).items():
            lines.append(f"| {category} | {rates['baseline']:.1%} | {rates['candidate']:.1%} |")

    lines.extend(["", "### Cost, latency and tokens", ""])
    lines.append("| Metric | Baseline | Candidate |")
    lines.append("| --- | ---: | ---: |")
    cost = verdict.cost_summary
    latency = verdict.latency_summary
    tokens = verdict.token_totals
    lines.append(
        f"| Total cost | {_money(cost['baseline']['total_cost'])} | "
        f"{_money(cost['candidate']['total_cost'])} |"
    )
    lines.append(
        f"| Cost per successful task | {_money(cost['baseline']['cost_per_successful_task'])} | "
        f"{_money(cost['candidate']['cost_per_successful_task'])} |"
    )
    lines.append(
        f"| Latency per trajectory, median / p95 (ms) | "
        f"{latency['baseline']['median']:.1f} / {latency['baseline']['p95']:.1f} | "
        f"{latency['candidate']['median']:.1f} / {latency['candidate']['p95']:.1f} |"
    )
    lines.append(
        f"| Prompt tokens | {tokens['baseline']['prompt_tokens']:,} | "
        f"{tokens['candidate']['prompt_tokens']:,} |"
    )
    lines.append(
        f"| Completion tokens | {tokens['baseline']['completion_tokens']:,} | "
        f"{tokens['candidate']['completion_tokens']:,} |"
    )

    if summary and summary.get("worst_failures"):
        lines.extend(["", "### Worst failing candidate trajectories", ""])
        for item in summary["worst_failures"]:
            lines.append(
                f"- **{item['scenario_id']}** ({item['category']}, "
                f"{item['critical_violations']} critical violation(s)), "
                f"trajectory `{item['trajectory_id']}`"
            )
            for reason in item["reasons"][:4]:
                lines.append(f"  - {reason}")

    return "\n".join(lines) + "\n"
