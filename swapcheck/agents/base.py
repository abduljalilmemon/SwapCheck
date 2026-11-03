"""Abstract base agent for multi-turn conversational agents with tool execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from swapcheck.providers.base import BaseProvider, ChatMessage, LLMResponse, ToolCall


class AgentStepResult(BaseModel):
    """Result of an agent turn, capturing assistant response, tool calls, and metadata."""

    response_text: str | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    llm_responses: list[LLMResponse] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0


class BaseAgent(ABC):
    """Abstract interface for tool-using agents under test."""

    def __init__(self, provider: BaseProvider, system_prompt: str = "") -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.messages: list[ChatMessage] = []
        if self.system_prompt:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))

    @abstractmethod
    async def step(self, user_message: str) -> AgentStepResult:
        """Process user message, executing tool calls until a final message is emitted."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset conversation state, memory, and any session authentication."""
        pass
