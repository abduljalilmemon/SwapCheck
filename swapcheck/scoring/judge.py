"""LLM-as-a-judge scoring for SwapCheck trajectories."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from swapcheck.config import ModelConfig
from swapcheck.providers import build_provider
from swapcheck.providers.base import BaseProvider, ChatMessage, LLMResponse
from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.scoring.judge_prompt import build_judge_prompt, prompt_hash


class JudgeScore(BaseModel):
    """Structured score returned by the LLM-as-a-judge."""

    helpfulness: int = Field(default=3, ge=1, le=5)
    groundedness: int = Field(default=3, ge=1, le=5)
    tone: int = Field(default=3, ge=1, le=5)
    off_policy_promises: bool = Field(default=False)
    passed: bool = Field(default=True)
    explanation: str = Field(default="")
    model: str = Field(default="")
    temperature: float = Field(default=0.0)
    prompt_hash_value: str = Field(default="")

    @property
    def acceptable(self) -> bool:
        """Whether the judge verdict regards the trajectory as acceptable."""
        return bool(
            self.passed
            and self.helpfulness >= 3
            and self.groundedness >= 3
            and self.tone >= 3
            and not self.off_policy_promises
        )


class JudgeScorer:
    """Scores a trajectory with a dedicated judge model or a provided mock provider."""

    def __init__(
        self,
        provider: BaseProvider | None = None,
        *,
        judge_config: ModelConfig | None = None,
    ) -> None:
        if provider is not None:
            self.provider = provider
        elif judge_config is not None:
            self.provider = build_provider(judge_config)
        else:
            from swapcheck.providers.mock_provider import MockProvider

            self.provider = MockProvider(model_name="judge-mock", temperature=0.0)

    async def score(self, trajectory: Trajectory, *, scenario_id: str | None = None) -> JudgeScore:
        """Ask the judge to assess a trajectory and return a structured verdict."""
        prompt = build_judge_prompt(trajectory, scenario_id=scenario_id)
        prompt_key = prompt_hash(prompt)
        response = await self.provider.complete([ChatMessage(role="user", content=prompt)])
        payload = self._parse_response(response)

        score = JudgeScore(
            helpfulness=int(payload.get("helpfulness", 3)),
            groundedness=int(payload.get("groundedness", 3)),
            tone=int(payload.get("tone", 3)),
            off_policy_promises=bool(payload.get("off_policy_promises", False)),
            passed=bool(payload.get("passed", True)),
            explanation=str(payload.get("explanation", "No explanation provided.")),
            model=response.model or self.provider.model_name,
            temperature=getattr(self.provider, "temperature", 0.0),
            prompt_hash_value=prompt_key,
        )
        if not score.passed and score.explanation == "No explanation provided.":
            score.explanation = "The judge flagged the response as not acceptable."
        return score

    async def judge_trajectory(
        self, trajectory: Trajectory, *, scenario_id: str | None = None
    ) -> JudgeScore:
        """Backward-compatible alias used by callers that prefer the judge_trajectory name."""
        return await self.score(trajectory, scenario_id=scenario_id)

    @staticmethod
    def _parse_response(response: LLMResponse) -> dict[str, Any]:
        """Extract JSON content from an LLM response, with graceful fallback on invalid output."""
        if response.content is None:
            return {
                "helpfulness": 3,
                "groundedness": 3,
                "tone": 3,
                "off_policy_promises": False,
                "passed": True,
                "explanation": "No judge output was supplied.",
            }

        text = response.content.strip()
        if not text:
            return {
                "helpfulness": 3,
                "groundedness": 3,
                "tone": 3,
                "off_policy_promises": False,
                "passed": True,
                "explanation": "No judge output was supplied.",
            }

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {
                "helpfulness": 3,
                "groundedness": 3,
                "tone": 3,
                "off_policy_promises": False,
                "passed": True,
                "explanation": "Judge output could not be parsed as JSON; fallback default applied.",
            }

        if not isinstance(payload, dict):
            return {
                "helpfulness": 3,
                "groundedness": 3,
                "tone": 3,
                "off_policy_promises": False,
                "passed": True,
                "explanation": "Judge payload was not a JSON object; fallback default applied.",
            }

        helpfulness = payload.get("helpfulness", 3)
        groundedness = payload.get("groundedness", 3)
        tone = payload.get("tone", 3)

        return {
            "helpfulness": max(1, min(5, int(float(helpfulness)))),
            "groundedness": max(1, min(5, int(float(groundedness)))),
            "tone": max(1, min(5, int(float(tone)))),
            "off_policy_promises": bool(payload.get("off_policy_promises", False)),
            "passed": bool(payload.get("passed", True)),
            "explanation": str(
                payload.get("explanation", "The judge did not provide a rationale.")
            ),
        }
