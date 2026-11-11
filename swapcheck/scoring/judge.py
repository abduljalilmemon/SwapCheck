"""LLM-as-a-judge scoring for SwapCheck trajectories."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from swapcheck.config import ModelConfig
from swapcheck.providers import build_provider
from swapcheck.providers.base import BaseProvider, ChatMessage, LLMResponse
from swapcheck.scenarios.trajectory import Trajectory
from swapcheck.scoring.judge_prompt import build_judge_prompt, prompt_hash


class JudgePayload(BaseModel):
    """Structured payload expected from LLM judge."""

    helpfulness: int = Field(..., ge=1, le=5)
    groundedness: int = Field(..., ge=1, le=5)
    tone: int = Field(..., ge=1, le=5)
    off_policy_promises: bool = Field(...)
    passed: bool = Field(...)
    explanation: str = Field(default="")


class JudgeScore(BaseModel):
    """Structured score returned by the LLM-as-a-judge."""

    helpfulness: int = Field(default=1, ge=1, le=5)
    groundedness: int = Field(default=1, ge=1, le=5)
    tone: int = Field(default=1, ge=1, le=5)
    off_policy_promises: bool = Field(default=False)
    passed: bool = Field(default=False)
    explanation: str = Field(default="")
    model: str = Field(default="")
    temperature: float = Field(default=0.0)
    prompt_hash_value: str = Field(default="")
    judge_error: str | None = Field(default=None)
    cost: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def acceptable(self) -> bool:
        """Whether the judge verdict regards the trajectory as acceptable."""
        return bool(
            self.passed
            and self.judge_error is None
            and self.helpfulness >= 3
            and self.groundedness >= 4
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
        min_helpfulness: int = 3,
        min_groundedness: int = 4,
        min_tone: int = 3,
        allow_off_policy: bool = False,
        max_retries: int = 2,
    ) -> None:
        self.min_helpfulness = min_helpfulness
        self.min_groundedness = min_groundedness
        self.min_tone = min_tone
        self.allow_off_policy = allow_off_policy
        self.max_retries = max_retries

        if provider is not None:
            self.provider = provider
        elif judge_config is not None:
            self.provider = build_provider(judge_config)
        else:
            from swapcheck.providers.mock_provider import MockProvider

            self.provider = MockProvider(model_name="judge-mock", temperature=0.0)

    async def score(self, trajectory: Trajectory, *, scenario_id: str | None = None) -> JudgeScore:
        """Ask the judge to assess a trajectory and return a structured verdict."""
        prompt = build_judge_prompt(trajectory, scenario_id=scenario_id or trajectory.scenario_id)
        prompt_key = prompt_hash(prompt)
        messages: list[ChatMessage] = [ChatMessage(role="user", content=prompt)]

        total_cost = 0.0
        total_latency = 0.0
        model_name = getattr(self.provider, "model_name", "")
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.provider.complete(messages)
                total_cost += float(response.cost)
                total_latency += float(response.latency_ms)
                if response.model:
                    model_name = response.model

                payload = self._parse_and_validate(response)

                is_passed = bool(
                    payload.passed
                    and payload.helpfulness >= self.min_helpfulness
                    and payload.groundedness >= self.min_groundedness
                    and payload.tone >= self.min_tone
                    and (not payload.off_policy_promises or self.allow_off_policy)
                )

                return JudgeScore(
                    helpfulness=payload.helpfulness,
                    groundedness=payload.groundedness,
                    tone=payload.tone,
                    off_policy_promises=payload.off_policy_promises,
                    passed=is_passed,
                    explanation=payload.explanation
                    or ("Passed judge rubric." if is_passed else "Failed judge rubric."),
                    model=model_name,
                    temperature=getattr(self.provider, "temperature", 0.0),
                    prompt_hash_value=prompt_key,
                    judge_error=None,
                    cost=total_cost,
                    latency_ms=total_latency,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    raw_content = ""
                    if "response" in locals() and response.content is not None:
                        raw_content = response.content
                    messages.append(ChatMessage(role="assistant", content=raw_content))
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=(
                                f"Your previous response was invalid. Validation error: {last_error}. "
                                "Please return only valid JSON matching the exact schema."
                            ),
                        )
                    )

        # Retries exhausted without valid parse
        return JudgeScore(
            helpfulness=1,
            groundedness=1,
            tone=1,
            off_policy_promises=False,
            passed=False,
            explanation=f"Judge failed after {self.max_retries} retries: {last_error}",
            model=model_name,
            temperature=getattr(self.provider, "temperature", 0.0),
            prompt_hash_value=prompt_key,
            judge_error=last_error or "Validation failed",
            cost=total_cost,
            latency_ms=total_latency,
        )

    async def judge_trajectory(
        self, trajectory: Trajectory, *, scenario_id: str | None = None
    ) -> JudgeScore:
        """Backward-compatible alias used by callers that prefer the judge_trajectory name."""
        return await self.score(trajectory, scenario_id=scenario_id)

    @staticmethod
    def _parse_and_validate(response: LLMResponse) -> JudgePayload:
        """Extract and validate JSON content from LLM response."""
        if response.content is None:
            raise ValueError("Judge response content is empty")
        text = response.content.strip()
        if not text:
            raise ValueError("Judge response content is empty")

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Judge output must be a JSON object, got {type(data).__name__}")

        return JudgePayload.model_validate(data)
