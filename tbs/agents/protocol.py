"""The agent seam: a prompt-sized request in, a validated plan out.

Agents never receive a :class:`~tbs.engine.state.GameState`.  They see a
:class:`TurnRequest` carrying exactly what the engine's view serves and return
an :class:`AgentPlan` of JSON-style orders.  The coordinator validates every
order against the engine, which keeps a scripted agent, a local LLM and a future
council of agents interchangeable and unable to mutate the game behind the
rules' back.

The same interface is the M10 council seam: a council is an ``Agent`` whose
``plan`` asks four members and merges their orders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tbs.engine.actions import ACTION_TYPES

__all__ = [
    "Agent",
    "AgentPlan",
    "PlanFormatError",
    "TurnRequest",
    "parse_plan",
]


class PlanFormatError(ValueError):
    """The model's reply was not a usable order plan.  Retryable."""


@dataclass
class TurnRequest:
    """Everything one agent gets to decide its half-turn.

    A single agent leaves ``role``, ``allowed_units`` and ``brief`` empty.  A
    council member is handed a doctrine in ``role``, its disjoint slice of the
    army in ``allowed_units``, and the supreme commander's intent in ``brief``.
    """

    player: str
    view: dict[str, Any]
    recent_events: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    role: str | None = None
    allowed_units: list[str] | None = None
    brief: str = ""


@dataclass
class AgentPlan:
    """One agent's proposed orders for the current player's turn."""

    orders: list[dict[str, Any]] = field(default_factory=list)
    commentary: str = ""
    end_turn: bool = False


@runtime_checkable
class Agent(Protocol):
    """Anything that can plan one player-turn from a :class:`TurnRequest`."""

    name: str

    def plan(self, request: TurnRequest) -> AgentPlan:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# parsing a model reply
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = stripped[3:]
    if stripped.lower().startswith("json"):
        stripped = stripped[4:]
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _json_candidates(text: str):
    """Yield every object/array that parses anywhere in ``text``, in order."""
    stripped = _strip_fences(text)
    if not stripped:
        return
    try:
        yield json.loads(stripped)
        return
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            candidate, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        yield candidate


def _plan_from_candidate(candidate: Any) -> AgentPlan | None:
    if isinstance(candidate, list):
        raw_orders = candidate
        commentary = ""
        end_turn = False
    elif isinstance(candidate, dict):
        if "orders" in candidate or "commentary" in candidate or "end_turn" in candidate:
            raw_orders = candidate.get("orders", [])
            commentary = candidate.get("commentary", "")
            end_turn = bool(candidate.get("end_turn", False))
        elif "action" in candidate:
            raw_orders = [candidate]
            commentary = ""
            end_turn = False
        else:
            return None
    else:
        return None

    if isinstance(raw_orders, dict):  # a single order object
        raw_orders = [raw_orders]
    if not isinstance(raw_orders, list):
        raise PlanFormatError('"orders" must be a list of order objects')

    orders: list[dict[str, Any]] = []
    for index, item in enumerate(raw_orders):
        if not isinstance(item, dict):
            raise PlanFormatError(f"orders[{index}] must be an object, got {type(item).__name__}")
        action = item.get("action")
        if not isinstance(action, str) or action not in ACTION_TYPES:
            raise PlanFormatError(
                f"orders[{index}] has action {action!r}; use one of: {', '.join(ACTION_TYPES)}"
            )
        if action == "end_turn":
            end_turn = True
            continue
        orders.append(dict(item))

    if not isinstance(commentary, str):
        commentary = str(commentary)
    return AgentPlan(orders=orders, commentary=commentary.strip(), end_turn=end_turn)


def parse_plan(text: str) -> AgentPlan:
    """Parse a model reply into an :class:`AgentPlan`.

    Accepts clean JSON, fenced JSON, a bare list of orders, a single order
    object, or JSON embedded in prose.  Raises :class:`PlanFormatError` with a
    short excerpt when no usable plan is found.
    """
    if not isinstance(text, str):
        raise PlanFormatError(f"the reply must be text, got {type(text).__name__}")
    for candidate in _json_candidates(text):
        plan = _plan_from_candidate(candidate)
        if plan is not None:
            return plan
    excerpt = " ".join(text.split())[:160]
    raise PlanFormatError(
        f"no JSON order plan found in the reply; expected "
        f'{{"orders":[...],"commentary":"..."}}. Reply began: {excerpt!r}'
    )
