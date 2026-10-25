"""Agent layer: a local LLM or scripted policy plays one side through the same API.

The engine remains pure; everything here is an optional client.  The single
agent interface (``Agent.plan``) is also the council seam: :class:`CouncilAgent`
is just an agent that combines a commander and several generals' plans.
"""

from tbs.agents.council import CommanderAgent, CouncilAgent
from tbs.agents.generals import DEFAULT_GENERALS, GeneralSpec
from tbs.agents.protocol import Agent, AgentPlan, PlanFormatError, TurnRequest

__all__ = [
    "Agent",
    "AgentPlan",
    "CommanderAgent",
    "CouncilAgent",
    "DEFAULT_GENERALS",
    "GeneralSpec",
    "PlanFormatError",
    "TurnRequest",
]
