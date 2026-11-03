"""Trajectory and turn-record models for scenario execution capture."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TurnRecord(BaseModel):
    """Record of a single scripted user turn and the agent response that followed."""

    turn_index: int
    user_turn: str
    assistant_text: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens: int = 0
    cost: float = 0.0


class Trajectory(BaseModel):
    """Full execution trace for a single scenario repetition."""

    scenario_id: str
    category: str
    repetition_index: int = 0
    turns: list[TurnRecord] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    final_db_snapshot: dict[str, Any] = Field(default_factory=dict)
    budget_exceeded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
