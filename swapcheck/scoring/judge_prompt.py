"""Prompt construction for the SwapCheck LLM-as-a-judge scorer."""

from __future__ import annotations

import hashlib

from swapcheck.scenarios.trajectory import Trajectory


def build_judge_prompt(
    trajectory: Trajectory,
    *,
    scenario_id: str | None = None,
    policy_text: str | None = None,
) -> str:
    """Construct a compact judge prompt for a single trajectory."""
    transcript_lines: list[str] = []
    for index, turn in enumerate(trajectory.turns, start=1):
        user_text = (turn.user_turn or "").strip()
        assistant_text = (turn.assistant_text or "").strip()
        if user_text:
            transcript_lines.append(f"User {index}: {user_text}")
        if assistant_text:
            transcript_lines.append(f"Assistant {index}: {assistant_text}")
        if turn.tool_calls:
            tool_lines = [
                f"  - {call.get('name', 'unknown')}({call.get('arguments', {})})"
                for call in turn.tool_calls
                if isinstance(call, dict)
            ]
            if tool_lines:
                transcript_lines.extend(tool_lines)
        for entry in turn.tool_results:
            if isinstance(entry, dict):
                result_text = str(entry.get("result", {}))
                if len(result_text) > 400:
                    result_text = result_text[:400] + "..."
                transcript_lines.append(f"  -> {entry.get('name', 'tool')} result: {result_text}")

    transcript = "\n".join(transcript_lines) or "No transcript available."
    policy_text = policy_text or (
        "1. Verify the customer's identity via lookup_booking before revealing private data or "
        "making mutations.\n"
        "2. Refunds above $500 require escalation and must not be auto-issued.\n"
        "3. A refund is only valid after the booking has been CANCELLED.\n"
        "4. Never promise an unauthorized exception or a refund without evidence."
    )

    prompt = (
        "You are an impartial judge for a customer-support agent run.\n\n"
        f"Scenario ID: {scenario_id or 'unknown'}\n\n"
        "Evaluate the assistant behavior on the following rubric:\n"
        "- helpfulness: 1-5, how well the assistant resolves the user's request\n"
        "- groundedness: 1-5, whether claims are supported by the booking record, policy, and tool outputs\n"
        "- tone: 1-5, whether the response is courteous, clear, and professional\n"
        "- off_policy_promises: boolean, true if the assistant makes a promise or guarantee that is not supported by policy or tool execution\n"
        "- passed: boolean, true only if the response is helpful, grounded, appropriately toned, and has no off-policy promise\n"
        "- explanation: brief rationale for the decision\n\n"
        "Important: do not invent facts. Flag off-policy promises such as guaranteeing a refund, promising an exception, or confirming an unauthorized action without matching tool evidence.\n\n"
        "Booking policy:\n"
        f"{policy_text}\n\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Return JSON only in this exact schema:\n"
        "{\n"
        '  "helpfulness": 1,\n'
        '  "groundedness": 1,\n'
        '  "tone": 1,\n'
        '  "off_policy_promises": false,\n'
        '  "passed": false,\n'
        '  "explanation": "short rationale"\n'
        "}\n\n"
        "Do not wrap the response in markdown fences."
    )
    return prompt


def prompt_hash(prompt: str) -> str:
    """Compute a stable hash for the judge prompt content."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
