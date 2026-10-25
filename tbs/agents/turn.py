"""Turn coordination: validate an agent's whole-turn plan, retry, and end the turn.

The coordinator is deliberately mechanical.  It never asks the engine to trust
an agent: every order goes through :func:`tbs.engine.rules.perform`, so a stale
target, a made-up coordinate or a unit acting twice is rejected with the same
correction-shaped message a human would get.  Accepted orders stay applied; a
rejection triggers one repaired request with the error text and a fresh view.
After the retry budget (or no usable output at all) the turn ends anyway, so a
weak local model can never stall the game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tbs.agents.llm import JsonlLogger, LlmResponseError
from tbs.agents.protocol import Agent, AgentPlan, PlanFormatError, TurnRequest
from tbs.engine import rules, serialize, view
from tbs.engine.errors import ActionError
from tbs.engine.state import GameState

__all__ = [
    "AppliedOrder",
    "ApplyResult",
    "StepwiseSession",
    "TurnResult",
    "TurnSession",
    "apply_plan",
    "build_request",
    "normalize_order",
    "recent_events",
    "run_turn",
    "run_turn_stepwise",
]

_NOTABLE_EVENTS = {
    "attack",
    "counterattack",
    "destroyed",
    "captured",
    "objective_captured",
    "build",
    "out_of_fuel",
    "eliminated",
}


def recent_events(state: GameState, limit: int = 8) -> list[str]:
    """A short, prompt-friendly tail of the action log."""
    lines: list[str] = []
    for entry in state.log[-limit:]:
        lines.append(f"turn {entry['turn']} {entry['player']}: {entry['describe']}")
        for event in entry.get("events", []):
            if event.get("type") not in _NOTABLE_EVENTS:
                continue
            kind = event["type"]
            if kind == "attack":
                lines.append(
                    f"  {event['attacker']} hit {event['defender']} for {event['damage']} "
                    f"(defender hp {event['defender_hp']})"
                )
            elif kind == "counterattack":
                lines.append(
                    f"  {event['attacker']} counterattacked {event['defender']} for {event['damage']}"
                )
            elif kind == "destroyed":
                lines.append(f"  {event['unit']} destroyed ({event.get('cause', 'combat')})")
            elif kind == "captured":
                lines.append(f"  {event.get('site')} captured by {event.get('owner')}")
            elif kind == "objective_captured":
                lines.append(f"  OBJECTIVE {event.get('site')} taken by {event.get('winner')}")
            elif kind == "build":
                lines.append(f"  {event.get('unit')} built at {event.get('site')}")
            elif kind == "out_of_fuel":
                lines.append(f"  {event.get('unit')} out of fuel (-{event.get('damage')} hp)")
            elif kind == "eliminated":
                lines.append(f"  player {event.get('player')} eliminated")
    return lines[-(limit * 2) :]


def build_request(
    state: GameState,
    player: str,
    feedback: list[str] | None = None,
    *,
    role: str | None = None,
    allowed_units: list[str] | None = None,
    history: int = 8,
) -> TurnRequest:
    """The view + recent events + optional rejection feedback for one attempt."""
    return TurnRequest(
        player=player,
        view=view.state_view(state, player),
        recent_events=recent_events(state, history),
        feedback=list(feedback or []),
        role=role,
        allowed_units=allowed_units,
    )


# ---------------------------------------------------------------------------
# normalising and applying orders
# ---------------------------------------------------------------------------


def normalize_order(state: GameState, order: dict[str, Any]) -> dict[str, Any]:
    """Validate one order and expand the agent-friendly ``move`` shorthand.

    The prompt view lists move *destinations*, not paths, so an LLM should not
    have to invent a path.  ``{"action":"move","unit":"A2","to":[q,r]}`` is
    expanded here from the engine's current legal moves.  Orders that already
    carry a ``path`` are passed through unchanged.

    Attacks, captures, loads, unloads and builds are also checked against the
    engine's own option lists first.  The resulting error names the legal choices,
    which is far more useful to a retrying model than a bare range message.
    """
    if not isinstance(order, dict):
        raise ActionError(f"an order must be a JSON object, got {type(order).__name__}")
    action = order.get("action")
    if not isinstance(action, str):
        raise ActionError('an order needs an "action" string')
    unit_id = str(order.get("unit", ""))

    if action == "move":
        if "path" in order:
            return order
        destination = order.get("to", order.get("target"))
        if not isinstance(destination, (list, tuple)) or len(destination) != 2:
            raise ActionError(
                'a move needs "to":[q,r] with a destination listed in '
                "unit_actions[<unit>].move_to (or an engine path)"
            )
        try:
            wanted = (int(destination[0]), int(destination[1]))
        except (TypeError, ValueError):
            raise ActionError(f"move destination {destination!r} is not a [q,r] coordinate") from None
        try:
            moves = rules.legal_moves(state, unit_id)
        except KeyError as exc:
            raise ActionError(str(exc)) from None
        match = next((move for move in moves if tuple(move["pos"]) == wanted), None)
        if match is None:
            choices = ", ".join(str(move["pos"]) for move in moves[:12]) or "none"
            raise ActionError(
                f"no legal move for {unit_id} to {list(wanted)}; choose a destination from "
                f"unit_actions[{unit_id}].move_to (current choices: {choices})"
            )
        return {"action": "move", "unit": unit_id, "path": match["path"]}

    if action == "attack":
        target = order.get("target")
        if not isinstance(target, str) or not target:
            raise ActionError('an attack needs a "target" unit id')
        weapon = order.get("weapon")
        try:
            options = rules.attack_options(state, unit_id)
        except KeyError as exc:
            raise ActionError(str(exc)) from None
        if not any(
            option["target"] == target and (weapon is None or option["weapon"] == weapon)
            for option in options
        ):
            legal = ", ".join(
                f"{option['target']} with {option['weapon']}" for option in options[:12]
            ) or "none"
            wanted = f"{target} with {weapon}" if weapon else target
            raise ActionError(
                f"{unit_id} has no legal attack on {wanted}; legal attacks: {legal}"
            )
        return order

    if action == "capture":
        try:
            options = rules.capture_options(state, unit_id)
        except KeyError as exc:
            raise ActionError(str(exc)) from None
        if not options:
            raise ActionError(
                f"{unit_id} has no capturable site under it; move onto a city, factory, port, "
                "airport or HQ first"
            )
        return order

    if action == "build":
        site = order.get("site")
        unit_type = order.get("unit")
        if not isinstance(site, (list, tuple)) or len(site) != 2:
            raise ActionError('a build needs a "site":[q,r] coordinate')
        try:
            wanted_site = [int(site[0]), int(site[1])]
        except (TypeError, ValueError):
            raise ActionError(f"build site {site!r} is not a [q,r] coordinate") from None
        options = rules.build_options(state, state.current_player)
        if not any(option["site"] == wanted_site and option["unit"] == unit_type for option in options):
            legal = ", ".join(
                f"{option['unit']} at {option['site']} ({'affordable' if option['affordable'] else 'too expensive'})"
                for option in options[:12]
            ) or "none: every build site is occupied, not owned, or has no options"
            raise ActionError(
                f"cannot build {unit_type!r} at {wanted_site}; legal builds: {legal}"
            )
        return order

    if action in ("load", "unload"):
        cargo = order.get("cargo")
        try:
            if action == "load":
                options = [
                    option for option in rules.load_options(state, unit_id) if option["cargo"] == cargo
                ]
            else:
                target = order.get("target")
                wanted = [int(target[0]), int(target[1])] if isinstance(target, (list, tuple)) and len(target) == 2 else None
                options = [
                    option
                    for option in rules.unload_options(state, unit_id)
                    if option["cargo"] == cargo and option["at"] == wanted
                ]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ActionError(str(exc)) from None
        if not options:
            raise ActionError(
                f"{unit_id} has no legal {action} for cargo {cargo!r} right now; "
                f"check unit_actions[{unit_id}] for the adjacent cargo and drop hexes"
            )
        return order

    return order


@dataclass
class AppliedOrder:
    """One accepted order and the events it produced."""

    order: dict[str, Any]
    events: list[dict]


@dataclass
class ApplyResult:
    applied: list[AppliedOrder] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return bool(self.errors)


def apply_plan(state: GameState, plan: AgentPlan) -> ApplyResult:
    """Apply every order that validates, collecting the ones that do not.

    Orders are independent (a partial-order protocol), so one bad order does not
    discard the others.  A rejected move leaves its unit in place, and a later
    follow-up for that unit is validated against the position that really exists.
    That is more forgiving to a model than aborting the whole plan on the first
    hallucinated target, while the engine remains the authority on what happened.
    """
    result = ApplyResult()
    resolved_units: set[str] = set()
    for raw_order in plan.orders:
        if state.status != "ongoing":
            break
        action = raw_order.get("action") if isinstance(raw_order, dict) else None
        unit_key = raw_order.get("unit") if isinstance(raw_order, dict) and action not in ("build", "end_turn") else None
        if unit_key is not None and unit_key in resolved_units:
            result.ignored.append(f"duplicate order for {unit_key} ignored: it already acted this turn")
            continue
        try:
            order = normalize_order(state, raw_order)
            if order.get("action") == "end_turn":
                continue
            events = rules.perform(state, order)
        except ActionError as exc:
            result.errors.append(str(exc))
            continue
        except (KeyError, ValueError) as exc:
            result.errors.append(str(exc))
            continue
        result.applied.append(AppliedOrder(order=order, events=events))
        if unit_key is not None:
            resolved_units.add(str(unit_key))
    return result


def _units_can_act(state: GameState, player: str) -> bool:
    for unit in state.units_of(player):
        if unit.acted or unit.loaded or unit.deployed_this_turn:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# one turn, possibly across several plan attempts
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """What one agent turn actually did."""

    player: str
    applied: list[AppliedOrder]
    errors: list[str]
    attempts: int
    commentary: list[str]
    ended_turn: bool
    end_events: list[dict]
    state_hash: str
    ignored: list[str] = field(default_factory=list)

    @property
    def orders(self) -> list[dict[str, Any]]:
        """The accepted orders plus the coordinator's end_turn, ready for replay."""
        orders = [item.order for item in self.applied]
        if self.ended_turn:
            orders.append({"action": "end_turn"})
        return orders

    @property
    def events(self) -> list[dict]:
        events = [event for item in self.applied for event in item.events]
        events.extend(self.end_events)
        return events


class TurnSession:
    """A single agent turn driven step-by-step, for UI threads or a sync loop.

    Call :meth:`next_request` to get the next plan request, feed the resulting
    plan to :meth:`record_plan` (or a parse error to :meth:`record_failure`),
    then ask :meth:`ready` whether another attempt is warranted.  Finish with
    :meth:`finish`, which always ends the turn while the game is ongoing.
    """

    def __init__(
        self,
        state: GameState,
        agent: Agent,
        *,
        attempts: int = 2,
        history: int = 8,
        logger: JsonlLogger | None = None,
    ):
        self.state = state
        self.agent = agent
        self.attempts = max(1, int(attempts))
        self.history = history
        self.logger = logger
        self.player = state.current_player
        self.attempt = 0
        self.feedback: list[str] = []
        self.applied: list[AppliedOrder] = []
        self.errors: list[str] = []
        self.ignored: list[str] = []
        self.commentary: list[str] = []
        self.complete = False
        self.aborted = False
        self.finished = False
        self._result: TurnResult | None = None

    def ready(self) -> bool:
        if self.finished or self.aborted or self.complete:
            return False
        if self.attempt >= self.attempts:
            return False
        if self.attempt == 0:
            return True
        return _units_can_act(self.state, self.player)

    def next_request(self) -> TurnRequest | None:
        if not self.ready():
            return None
        self.attempt += 1
        return build_request(
            self.state,
            self.player,
            self.feedback,
            history=self.history,
        )

    def record_plan(self, plan: AgentPlan) -> None:
        if plan.commentary:
            self.commentary.append(plan.commentary)
        result = apply_plan(self.state, plan)
        self.applied.extend(result.applied)
        self.errors.extend(result.errors)
        self.ignored.extend(result.ignored)
        self.feedback = result.errors
        if not result.rejected:
            self.complete = True
        self._log(
            kind="turn_attempt",
            accepted=[item.order for item in result.applied],
            rejected=result.errors,
            ignored=result.ignored,
            commentary=plan.commentary,
        )

    def record_failure(self, message: str) -> None:
        self.feedback = [message]
        self.errors.append(message)
        self._log(kind="turn_attempt", accepted=[], rejected=[message], commentary="")

    def abort(self) -> None:
        """Stop without ending the turn (used when the LLM server is unreachable)."""
        self.aborted = True
        self.finished = True

    def finish(self) -> TurnResult:
        if self._result is not None:
            return self._result
        ended = False
        end_events: list[dict] = []
        if not self.aborted and self.state.status == "ongoing" and self.state.current_player == self.player:
            end_events = rules.perform(self.state, {"action": "end_turn"})
            ended = True
        self._result = TurnResult(
            player=self.player,
            applied=list(self.applied),
            errors=list(self.errors),
            attempts=self.attempt,
            commentary=[text for text in self.commentary if text],
            ended_turn=ended,
            end_events=end_events,
            state_hash=serialize.state_hash(self.state),
            ignored=list(self.ignored),
        )
        self.finished = True
        self._log(
            kind="turn_complete",
            accepted=[item.order for item in self.applied],
            rejected=self.errors,
            ignored=self.ignored,
            ended_turn=ended,
        )
        return self._result

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is None:
            return
        self.logger.write(
            {
                "kind": kind,
                "player": self.player,
                "turn": self.state.turn_number,
                "attempt": self.attempt,
                "state_hash": serialize.state_hash(self.state),
                **fields,
            }
        )


def run_turn(
    state: GameState,
    agent: Agent,
    *,
    attempts: int = 2,
    history: int = 8,
    logger: JsonlLogger | None = None,
) -> TurnResult:
    """Play one full agent turn synchronously.

    Model/format failures are retried up to ``attempts``; a transport failure
    propagates so the caller can offer the human a retry instead of silently
    forfeiting the game.
    """
    session = TurnSession(state, agent, attempts=attempts, history=history, logger=logger)
    while True:
        request = session.next_request()
        if request is None:
            break
        try:
            plan = agent.plan(request)
        except (PlanFormatError, LlmResponseError) as exc:
            session.record_failure(str(exc))
            continue
        session.record_plan(plan)
        if session.complete:
            break
    return session.finish()


class StepwiseSession:
    """A stepwise turn driven by an external loop, one agent batch per call.

    The UI uses this class to keep the window responsive: ask
    :meth:`next_request`, run the agent in a worker thread, hand the plan to
    :meth:`record_plan`, and repeat while :meth:`ready` is true.  The semantics
    are exactly :func:`run_turn_stepwise`: a fresh view each step, feedback after
    rejections, and an unconditional end to the turn in :meth:`finish`.

    ``attempts`` on the resulting :class:`TurnResult` counts agent calls, not
    retries within a call.
    """

    def __init__(
        self,
        state: GameState,
        agent: Agent,
        *,
        max_steps: int = 32,
        history: int = 8,
        logger: JsonlLogger | None = None,
    ):
        self.state = state
        self.agent = agent
        self.max_steps = max(1, int(max_steps))
        self.history = history
        self.logger = logger
        self.player = state.current_player
        self.steps = 0
        self.feedback: list[str] = []
        self.applied: list[AppliedOrder] = []
        self.errors: list[str] = []
        self.ignored: list[str] = []
        self.commentary: list[str] = []
        self.done = False
        self.aborted = False
        self.finished = False
        self._result: TurnResult | None = None

    def ready(self) -> bool:
        if self.done or self.aborted or self.finished:
            return False
        if self.steps >= self.max_steps:
            return False
        if self.state.status != "ongoing" or self.state.current_player != self.player:
            return False
        return _units_can_act(self.state, self.player)

    def next_request(self) -> TurnRequest | None:
        if not self.ready():
            return None
        self.steps += 1
        return build_request(
            self.state,
            self.player,
            self.feedback,
            history=self.history,
        )

    def record_plan(self, plan: AgentPlan) -> None:
        if plan.commentary:
            self.commentary.append(plan.commentary)
        result = apply_plan(self.state, plan)
        self.applied.extend(result.applied)
        self.errors.extend(result.errors)
        self.ignored.extend(result.ignored)
        self.feedback = result.errors
        self._log(
            kind="turn_attempt",
            accepted=[item.order for item in result.applied],
            rejected=result.errors,
            ignored=result.ignored,
            commentary=plan.commentary,
        )
        if plan.end_turn:
            self.done = True
        elif not result.applied and not result.errors:
            # Ignored-only or empty plans mean the agent has nothing more to
            # say; asking again would only burn the step budget.
            self.done = True

    def record_failure(self, message: str) -> None:
        self.feedback = [message]
        self.errors.append(message)
        self._log(
            kind="turn_attempt",
            accepted=[],
            rejected=[message],
            ignored=[],
            commentary="",
        )

    def abort(self) -> None:
        """Stop without ending the turn (used when the agent server is unreachable)."""
        self.aborted = True
        self.finished = True

    def finish(self) -> TurnResult:
        if self._result is not None:
            return self._result
        ended = False
        end_events: list[dict] = []
        if (
            not self.aborted
            and self.state.status == "ongoing"
            and self.state.current_player == self.player
        ):
            end_events = rules.perform(self.state, {"action": "end_turn"})
            ended = True
        self._result = TurnResult(
            player=self.player,
            applied=list(self.applied),
            errors=list(self.errors),
            attempts=self.steps,
            commentary=[text for text in self.commentary if text],
            ended_turn=ended,
            end_events=end_events,
            state_hash=serialize.state_hash(self.state),
            ignored=list(self.ignored),
        )
        self.finished = True
        self._log(
            kind="turn_complete",
            accepted=[item.order for item in self.applied],
            rejected=self.errors,
            ignored=self.ignored,
            ended_turn=ended,
        )
        return self._result

    def _log(self, kind: str, **fields: Any) -> None:
        if self.logger is None:
            return
        self.logger.write(
            {
                "kind": kind,
                "player": self.player,
                "turn": self.state.turn_number,
                "attempt": self.steps,
                "step": self.steps,
                "state_hash": serialize.state_hash(self.state),
                **fields,
            }
        )


def run_turn_stepwise(
    state: GameState,
    agent: Agent,
    *,
    max_steps: int = 32,
    history: int = 8,
    logger: JsonlLogger | None = None,
) -> TurnResult:
    """Play one turn as a sequence of agent calls, one batch per step.

    :func:`run_turn` asks once for a whole-turn plan and ends the turn.  A
    stepwise agent -- NCM, or any model that should choose one order at a time --
    instead gets a fresh view after every batch, so each decision sees the board
    the previous order produced.  Agents that already return whole-turn plans
    work unchanged: their orders are applied, and the next call only happens if
    a unit can still act.

    The loop stops when no unit can act, the agent sends ``end_turn``, the game
    ends, a call applies and reports nothing (the agent has nothing more to say),
    or ``max_steps`` calls have been made.  Format failures are retried with
    feedback while the budget lasts; transport failures propagate so the caller
    can offer the human a retry.  The turn is always ended afterwards while the
    game is ongoing, and ``TurnResult.attempts`` counts every agent call made.
    """
    session = StepwiseSession(
        state, agent, max_steps=max_steps, history=history, logger=logger
    )
    while True:
        request = session.next_request()
        if request is None:
            break
        try:
            plan = agent.plan(request)
        except (PlanFormatError, LlmResponseError) as exc:
            session.record_failure(str(exc))
            continue
        session.record_plan(plan)
        if session.done:
            break
    return session.finish()
