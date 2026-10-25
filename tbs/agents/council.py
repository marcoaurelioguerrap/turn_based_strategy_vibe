"""The M10 council: a commander, several generals, and a deterministic merger.

The council is one :class:`~tbs.agents.protocol.Agent`.  A single call to
:meth:`CouncilAgent.plan` briefs the generals, asks each one for orders over its
own slice of the army, and returns a single :class:`AgentPlan` for the existing
turn coordinator.  No engine state is touched here: generals only receive views,
and the coordinator remains the sole authority on legality.

The reference deployment uses one local Nemotron endpoint for every role; the
differences between generals are doctrine and unit slice, not weights.  Members
and the commander are injectable so tests run without a network and a future
mixed-model council needs no changes here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Protocol

from tbs.agents.generals import (
    DEFAULT_GENERALS,
    GeneralSpec,
    assign_units,
    filter_view,
    general_role,
    select_generals,
)
from tbs.agents.llm import (
    JsonlLogger,
    LlmClient,
    LlmResponseError,
    LlmTransportError,
    SingleLlmAgent,
    _compact_view,
)
from tbs.agents.protocol import Agent, AgentPlan, PlanFormatError, TurnRequest

__all__ = [
    "BriefingAgent",
    "CommanderAgent",
    "CouncilAgent",
    "build_commander_messages",
    "merge_plans",
]


class BriefingAgent(Protocol):
    """Anything that can turn a turn request into a short commander's intent."""

    name: str

    def brief(self, request: TurnRequest) -> str:  # pragma: no cover - protocol
        ...


def build_commander_messages(request: TurnRequest) -> list[dict[str, str]]:
    """The two chat messages for the supreme commander's short pre-brief."""
    system = (
        f"You are the supreme commander of player {request.player} in a deterministic hex wargame.\n"
        "You do not issue unit orders; your generals execute your intent.\n"
        "Read the game view and reply with at most 60 words of plain text: the main effort,\n"
        "1-3 priority targets or sites, and whether to build now. No JSON, no markdown.\n"
    )
    payload: dict[str, Any] = {
        "you_are": request.player,
        "turn": request.view.get("turn"),
        "status": request.view.get("status"),
        "recent_events": request.recent_events[-6:],
        "view": _compact_view(request.view),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        },
    ]


class CommanderAgent:
    """One small LLM call that sets the council's intent for the turn."""

    name = "commander"

    def __init__(self, client: LlmClient, *, logger: JsonlLogger | None = None):
        self.client = client
        self.logger = logger

    def brief(self, request: TurnRequest) -> str:
        messages = build_commander_messages(request)
        if self.logger is not None:
            self.logger.write(
                {
                    "kind": "commander_request",
                    "agent": self.name,
                    "model": self.client.config.model,
                    "player": request.player,
                    "turn": request.view.get("turn"),
                    "messages": messages,
                }
            )
        try:
            raw = self.client.complete(messages)
        except (LlmResponseError, LlmTransportError) as exc:
            if self.logger is not None:
                self.logger.write(
                    {
                        "kind": "commander_error",
                        "agent": self.name,
                        "player": request.player,
                        "turn": request.view.get("turn"),
                        "error": str(exc),
                    }
                )
            raise
        text = " ".join(raw.split()).strip()
        if not text:
            raise LlmResponseError("the commander returned an empty brief")
        text = text[:600]
        if self.logger is not None:
            self.logger.write(
                {
                    "kind": "commander_response",
                    "agent": self.name,
                    "player": request.player,
                    "turn": request.view.get("turn"),
                    "raw": raw,
                    "brief": text,
                }
            )
        return text


def _legal_build(view: dict[str, Any], site: Any, unit_type: Any) -> bool:
    if not isinstance(site, (list, tuple)) or len(site) != 2:
        return False
    wanted = [site[0], site[1]]
    for option in view.get("build_options", []):
        if not isinstance(option, dict):
            continue
        if (
            option.get("site") == wanted
            and option.get("unit") == unit_type
            and option.get("affordable", True)
        ):
            return True
    return False


def _order_legal_in_view(
    view: dict[str, Any], order: dict[str, Any]
) -> tuple[bool, str]:
    """Conservatively reject orders the snapshot already says are impossible.

    The coordinator is still the authority; this only removes obvious
    hallucinations (a destination missing from ``move_to``, an attack missing
    from ``attack``) before they cost a repair round.  Orders that are legal now
    but stale after another general acts are left for the coordinator.
    """
    unit_id = str(order.get("unit", ""))
    summary = (view.get("unit_actions") or {}).get(unit_id)
    if not isinstance(summary, dict):
        return False, f"{unit_id or 'an order'} has no legal actions"
    if summary.get("acted") or summary.get("loaded_in") or summary.get("new_unit"):
        return False, f"{unit_id} cannot act"
    action = order.get("action")
    if action == "move":
        if "path" in order:
            return True, ""
        to = order.get("to", order.get("target"))
        if not isinstance(to, (list, tuple)) or len(to) != 2:
            return False, f"{unit_id} has a malformed move destination"
        wanted = [to[0], to[1]]
        if not any(
            isinstance(entry, list)
            and entry
            and isinstance(entry[0], (list, tuple))
            and list(entry[0]) == wanted
            for entry in summary.get("move_to", [])
        ):
            return False, f"{unit_id} cannot reach {wanted}"
        return True, ""
    if action == "attack":
        target = order.get("target")
        weapon = order.get("weapon")
        if not any(
            isinstance(entry, list)
            and len(entry) >= 2
            and entry[0] == target
            and (weapon is None or entry[1] == weapon)
            for entry in summary.get("attack", [])
        ):
            suffix = f" with {weapon}" if weapon else ""
            return False, f"{unit_id} has no legal attack on {target}{suffix}"
        return True, ""
    if action == "capture":
        if "capture_remaining_hp" not in summary:
            return False, f"{unit_id} has nothing to capture"
        return True, ""
    if action == "load":
        cargo = order.get("cargo")
        if cargo not in summary.get("load", []):
            return False, f"{unit_id} cannot load {cargo}"
        return True, ""
    if action == "unload":
        cargo = order.get("cargo")
        at = order.get("target")
        wanted = list(at) if isinstance(at, (list, tuple)) else None
        if wanted is None or not any(
            isinstance(entry, list)
            and len(entry) == 2
            and entry[0] == cargo
            and isinstance(entry[1], (list, tuple))
            and list(entry[1]) == wanted
            for entry in summary.get("unload", [])
        ):
            return False, f"{unit_id} cannot unload {cargo} there"
        return True, ""
    if action == "hold" and not summary.get("hold"):
        return False, f"{unit_id} cannot hold"
    return True, ""


def merge_plans(
    results: Iterable[tuple[GeneralSpec, AgentPlan]],
    assignments: dict[str, list[str]],
    view: dict[str, Any],
) -> tuple[AgentPlan, list[str]]:
    """Combine member plans into one deterministic, priority-ordered plan.

    Returns the plan and human-readable reasons for every dropped order, which
    the council logs.  Duplicate attacks on the same enemy are intentionally
    kept: focus fire is a feature, and the coordinator rejects a stale second
    attack if the target is already gone.
    """
    ordered = sorted(results, key=lambda item: (item[0].priority, item[0].id))
    seen_units: set[str] = set()
    build_sites: set[tuple[Any, ...]] = set()
    orders: list[dict[str, Any]] = []
    comments: list[str] = []
    dropped: list[str] = []
    for general, plan in ordered:
        if plan.commentary:
            comments.append(f"{general.name}: {plan.commentary.strip()}")
        emitted = 0
        for raw in plan.orders:
            if not isinstance(raw, dict):
                dropped.append(f"{general.name}: ignored a non-object order")
                continue
            action = raw.get("action")
            if action == "end_turn":
                continue
            if action == "build":
                if not general.can_build:
                    dropped.append(f"{general.name}: builds are the Quartermaster's job")
                    continue
                site = raw.get("site")
                key = tuple(site) if isinstance(site, (list, tuple)) and len(site) == 2 else None
                if key is None or not _legal_build(view, site, raw.get("unit")):
                    dropped.append(f"{general.name}: dropped an illegal build {raw.get('unit')!r} at {site!r}")
                    continue
                if key in build_sites:
                    dropped.append(f"{general.name}: dropped a second build at {site!r}")
                    continue
                if general.max_orders is not None and emitted >= general.max_orders:
                    dropped.append(f"{general.name}: reached its order cap")
                    continue
                build_sites.add(key)
                orders.append(dict(raw))
                emitted += 1
                continue
            unit_id = str(raw.get("unit", ""))
            if unit_id not in assignments.get(general.id, []):
                dropped.append(f"{general.name}: {unit_id or 'an order'} is outside its command")
                continue
            if unit_id in seen_units:
                dropped.append(f"{general.name}: {unit_id} was already ordered by a senior general")
                continue
            legal, reason = _order_legal_in_view(view, raw)
            if not legal:
                dropped.append(f"{general.name}: dropped {reason}")
                continue
            if general.max_orders is not None and emitted >= general.max_orders:
                dropped.append(f"{general.name}: reached its order cap")
                continue
            seen_units.add(unit_id)
            orders.append(dict(raw))
            emitted += 1
    return AgentPlan(orders=orders, commentary="; ".join(comments)), dropped


class CouncilAgent:
    """One agent that runs the commander and every general, then merges them."""

    def __init__(
        self,
        client: LlmClient | None = None,
        *,
        generals: Iterable[GeneralSpec] = DEFAULT_GENERALS,
        members: dict[str, Agent] | None = None,
        commander: BriefingAgent | None = None,
        logger: JsonlLogger | None = None,
        history: int = 8,
        use_commander: bool = True,
        parallel: bool = False,
        name: str = "council",
    ):
        self.name = name
        self.logger = logger
        self.history = history
        self.parallel = bool(parallel)
        self.generals = select_generals(generals)
        if members is None and client is None:
            raise ValueError("a council needs an LlmClient or injected members")
        self.client = client
        if members is None:
            assert client is not None
            members = {
                general.id: SingleLlmAgent(client, logger=logger, history=history, name=general.name)
                for general in self.generals
            }
        self.members = dict(members)
        if use_commander:
            if commander is None and client is not None:
                commander = CommanderAgent(client, logger=logger)
            self.commander: BriefingAgent | None = commander
        else:
            self.commander = None
        self.last_stats: dict[str, Any] = {}

    # -- one turn -----------------------------------------------------------

    def plan(self, request: TurnRequest) -> AgentPlan:
        assignments = assign_units(request.view, self.generals)
        active = [
            general
            for general in self.generals
            if self._is_active(general, assignments.get(general.id, []), request.view)
        ]
        if not active:
            self.last_stats = {"generals": [], "calls": 0, "dropped": [], "failures": []}
            return AgentPlan(orders=[], commentary="council: no unit can act")

        brief = ""
        if self.commander is not None:
            try:
                brief = self.commander.brief(request)
            except (PlanFormatError, LlmResponseError, LlmTransportError) as exc:
                self._log(request, "council_brief_error", error=str(exc))

        results, failures, transport_errors = self._consult(
            active, assignments, request, brief
        )
        if not results:
            if transport_errors:
                raise transport_errors[0][1]
            if failures:
                raise failures[0][1]
            raise LlmResponseError("the council produced no usable plan")

        merged, dropped = merge_plans(results, assignments, request.view)
        if brief:
            commander_note = f"Commander: {brief}"
            merged.commentary = (
                f"{commander_note}; {merged.commentary}" if merged.commentary else commander_note
            )
        if failures:
            failed = ", ".join(general.name for general, _error in failures)
            note = f"unavailable: {failed}"
            merged.commentary = f"{merged.commentary}; {note}" if merged.commentary else note
        self.last_stats = {
            "generals": [general.id for general in active],
            "calls": len(active) + (1 if brief else 0),
            "dropped": dropped,
            "failures": [general.id for general, _error in failures],
            "transport_failures": [general.id for general, _error in transport_errors],
        }
        self._log(
            request,
            "council_merge",
            brief=brief,
            generals=[general.id for general in active],
            accepted=[order for order in merged.orders],
            dropped=dropped,
            failures=[general.id for general, _error in failures],
            transport_failures=[general.id for general, _error in transport_errors],
        )
        return merged

    @staticmethod
    def _is_active(
        general: GeneralSpec, units: list[str], view: dict[str, Any]
    ) -> bool:
        if units:
            return True
        if not general.can_build:
            return False
        return any(
            isinstance(option, dict) and option.get("affordable")
            for option in view.get("build_options", [])
        )

    def _consult(
        self,
        active: list[GeneralSpec],
        assignments: dict[str, list[str]],
        request: TurnRequest,
        brief: str,
    ) -> tuple[
        list[tuple[GeneralSpec, AgentPlan]],
        list[tuple[GeneralSpec, Exception]],
        list[tuple[GeneralSpec, Exception]],
    ]:
        results: list[tuple[GeneralSpec, AgentPlan]] = []
        failures: list[tuple[GeneralSpec, Exception]] = []
        transport_errors: list[tuple[GeneralSpec, Exception]] = []

        def ask(general: GeneralSpec) -> tuple[GeneralSpec, AgentPlan]:
            member = self.members[general.id]
            sub_request = TurnRequest(
                player=request.player,
                view=filter_view(
                    request.view,
                    assignments.get(general.id, []),
                    can_build=general.can_build,
                ),
                recent_events=list(request.recent_events),
                feedback=list(request.feedback),
                role=general_role(general),
                allowed_units=list(assignments.get(general.id, [])),
                brief=brief,
            )
            return general, member.plan(sub_request)

        def record(general: GeneralSpec, exc: Exception) -> None:
            if isinstance(exc, LlmTransportError):
                transport_errors.append((general, exc))
            else:
                failures.append((general, exc))
            self._log(request, "council_general_error", general=general.id, error=str(exc))

        if self.parallel and len(active) > 1:
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                futures = {pool.submit(ask, general): general for general in active}
                for future in as_completed(futures):
                    general = futures[future]
                    try:
                        results.append(future.result())
                    except (PlanFormatError, LlmResponseError, LlmTransportError) as exc:
                        record(general, exc)
        else:
            for general in active:
                try:
                    results.append(ask(general))
                except (PlanFormatError, LlmResponseError, LlmTransportError) as exc:
                    record(general, exc)

        priority = {general.id: index for index, general in enumerate(self.generals)}
        results.sort(key=lambda item: priority[item[0].id])
        return results, failures, transport_errors

    def _log(self, request: TurnRequest, kind: str, **fields: Any) -> None:
        if self.logger is None:
            return
        self.logger.write(
            {
                "kind": kind,
                "agent": self.name,
                "player": request.player,
                "turn": request.view.get("turn"),
                **fields,
            }
        )
