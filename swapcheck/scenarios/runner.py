"""Async scenario execution runner for the SwapCheck scenario suite."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swapcheck.agents.booking_agent import BookingSupportAgent
from swapcheck.agents.db import BookingDatabase
from swapcheck.cache import ResponseCache
from swapcheck.providers.base import BaseProvider, ChatMessage, LLMResponse
from swapcheck.providers.mock_provider import MockProvider
from swapcheck.scenarios.models import Scenario
from swapcheck.scenarios.trajectory import Trajectory, TurnRecord


class CachedProvider(BaseProvider):
    """Thin adapter that applies the shared response cache before delegating to a provider."""

    def __init__(self, provider: BaseProvider, cache: ResponseCache) -> None:
        self.provider = provider
        self.cache = cache
        self.model_name = provider.model_name
        self.temperature = getattr(provider, "temperature", 0.0)

    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        cache_key = self.cache.compute_key(
            model=self.provider.model_name,
            messages=messages,
            tools=tools,
            temperature=getattr(self.provider, "temperature", 0.0),
            extra={"kwargs": kwargs},
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        response = await self.provider.complete(messages=messages, tools=tools, **kwargs)
        self.cache.set(cache_key, self.provider.model_name, response)
        return response


class ScenarioRunner:
    """Run scripted scenario workloads across repeated executions with cost guardrails."""

    def __init__(
        self,
        scenarios: list[Scenario] | dict[str, list[Scenario]] | str | Path | None = None,
        *,
        repetitions: int = 1,
        max_concurrency: int = 4,
        provider_factory: Callable[[], BaseProvider] | None = None,
        cache: ResponseCache | None = None,
        max_cost: float | None = None,
    ) -> None:
        self.repetitions = max(1, repetitions)
        self.max_concurrency = max(1, max_concurrency)
        self.provider_factory = provider_factory or (lambda: MockProvider(model_name="mock-runner"))
        if cache is not None:
            self.cache = cache
        else:
            cache_path = Path(tempfile.gettempdir()) / f"swapcheck-runner-{uuid.uuid4().hex}.db"
            self.cache = ResponseCache(db_path=cache_path)
        self.max_cost = max_cost
        self.budget_exceeded = False
        self.total_cost_spent = 0.0
        self._budget_lock = asyncio.Lock()
        self.scenarios = self._normalize_scenarios(scenarios)

    @staticmethod
    def _normalize_scenarios(
        scenarios: list[Scenario] | dict[str, list[Scenario]] | str | Path | None,
    ) -> list[Scenario]:
        if scenarios is None:
            return []
        if isinstance(scenarios, (str, Path)):
            from swapcheck.scenarios.loader import load_scenarios

            grouped = load_scenarios(Path(scenarios))
            flattened: list[Scenario] = []
            for items in grouped.values():
                flattened.extend(items)
            return flattened
        if isinstance(scenarios, dict):
            flattened = []
            for items in scenarios.values():
                flattened.extend(items)
            return flattened
        return list(scenarios)

    async def _run_scenario(self, scenario: Scenario, repetition_index: int = 0) -> Trajectory:
        provider = self.provider_factory()
        cached_provider = CachedProvider(provider=provider, cache=self.cache)
        db = BookingDatabase(initial_data=scenario.initial_db_state)
        agent = BookingSupportAgent(provider=cached_provider, db=db)

        trajectory = Trajectory(
            scenario_id=scenario.id,
            category=scenario.category,
            repetition_index=repetition_index,
            final_db_snapshot=db.snapshot(),
        )

        if self.max_cost is not None:
            async with self._budget_lock:
                if self.budget_exceeded:
                    trajectory.budget_exceeded = True
                    return trajectory

        for turn_index, user_turn in enumerate(scenario.user_turns):
            result = await agent.step(user_turn)
            turn_record = TurnRecord(
                turn_index=turn_index,
                user_turn=user_turn,
                assistant_text=result.response_text,
                tool_calls=[tc.model_dump() for tc in result.tool_calls],
                tool_results=result.tool_outputs,
                latency_ms=result.total_latency_ms,
                tokens=result.total_tokens,
                cost=result.total_cost,
            )
            trajectory.turns.append(turn_record)
            trajectory.total_latency_ms += result.total_latency_ms
            trajectory.total_tokens += result.total_tokens
            trajectory.total_cost += result.total_cost

            if self.max_cost is not None:
                async with self._budget_lock:
                    self.total_cost_spent += result.total_cost
                    if self.total_cost_spent >= self.max_cost:
                        self.budget_exceeded = True
                        trajectory.budget_exceeded = True
                        break

        trajectory.final_db_snapshot = db.snapshot()
        return trajectory

    async def run_scenario(self, scenario: Scenario, repetition_index: int = 0) -> Trajectory:
        """Run a single scenario for one repetition."""
        return await self._run_scenario(scenario=scenario, repetition_index=repetition_index)

    async def run_all(self) -> list[Trajectory]:
        """Run all scenarios for the configured number of repetitions with bounded concurrency."""
        self.budget_exceeded = False
        self.total_cost_spent = 0.0

        work_items: list[tuple[Scenario, int]] = []
        for scenario in self.scenarios:
            for repetition_index in range(self.repetitions):
                work_items.append((scenario, repetition_index))

        if not work_items:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_limited(scenario: Scenario, repetition_index: int) -> Trajectory:
            async with semaphore:
                return await self._run_scenario(
                    scenario=scenario,
                    repetition_index=repetition_index,
                )

        results: list[Trajectory] = []
        batch: list[asyncio.Task[Trajectory]] = []

        for scenario, repetition_index in work_items:
            if self.max_cost is not None and self.budget_exceeded:
                break
            batch.append(asyncio.create_task(run_limited(scenario, repetition_index)))
            if len(batch) >= self.max_concurrency:
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)
                batch.clear()
                if self.max_cost is not None and self.budget_exceeded:
                    break

        if batch:
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

        return results
