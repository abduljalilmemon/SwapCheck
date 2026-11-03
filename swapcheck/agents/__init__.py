"""Agents, tools, mock databases, and policies for SwapCheck."""

from swapcheck.agents.base import AgentStepResult, BaseAgent
from swapcheck.agents.booking_agent import (
    DEFAULT_SYSTEM_PROMPT,
    BookingSupportAgent,
)
from swapcheck.agents.db import Booking, BookingDatabase
from swapcheck.agents.policy import (
    DEFAULT_POLICY,
    PolicyRules,
    RefundCalculation,
    calculate_eligible_refund,
    load_policy_document,
)
from swapcheck.agents.tools import (
    BOOKING_TOOLS_SCHEMA,
    ToolResult,
    cancel_booking,
    escalate_to_human,
    execute_booking_tool,
    get_booking_tools_schema,
    issue_refund,
    lookup_booking,
)

__all__ = [
    "AgentStepResult",
    "BaseAgent",
    "BOOKING_TOOLS_SCHEMA",
    "Booking",
    "BookingDatabase",
    "BookingSupportAgent",
    "DEFAULT_POLICY",
    "DEFAULT_SYSTEM_PROMPT",
    "PolicyRules",
    "RefundCalculation",
    "ToolResult",
    "calculate_eligible_refund",
    "cancel_booking",
    "escalate_to_human",
    "execute_booking_tool",
    "get_booking_tools_schema",
    "issue_refund",
    "load_policy_document",
    "lookup_booking",
]
