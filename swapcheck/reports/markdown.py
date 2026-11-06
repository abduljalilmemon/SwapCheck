"""Markdown rendering for evaluation results."""

from __future__ import annotations

from swapcheck.verdict.models import VerdictResult


def render_markdown_report(
    verdict: VerdictResult,
    *,
    baseline_name: str,
    candidate_name: str,
    suite_path: str,
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
            f"- Required sample size: {verdict.required_sample_size}",
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

    lines.extend(
        [
            "",
            "### Cost and latency",
            "",
            f"- Baseline cost: ${verdict.cost_summary['baseline']['total_cost']:.4f}",
            f"- Candidate cost: ${verdict.cost_summary['candidate']['total_cost']:.4f}",
            "",
            f"- Baseline latency mean: {verdict.latency_summary['baseline']['mean']:.1f} ms",
            f"- Candidate latency mean: {verdict.latency_summary['candidate']['mean']:.1f} ms",
        ]
    )

    return "\n".join(lines) + "\n"
