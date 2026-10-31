"""Turn the engine's advisory ferry hints into legal agent orders.

The engine's view names a transport, a cargo unit, a pickup hex and a landing
beach when an amphibious operation is worth attempting.  This module is the
small scripted executor for that advice: it issues the cargo move, transport
move, load, sail and unload orders in the one order that makes them valid.

It deliberately uses only ``unit_actions`` from the view, so a scripted agent
stays inside the same protocol as an LLM or NCM player.  The engine remains the
authority on every order; a stale hint simply produces a rejected order and the
coordinator retries (or ends the turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FerryPlan", "plan_ferry"]


@dataclass
class FerryPlan:
    """Orders to try first, plus the units the normal policy should leave alone."""

    orders: list[dict[str, Any]] = field(default_factory=list)
    claimed: set[str] = field(default_factory=set)
    destinations: set[tuple[int, int]] = field(default_factory=set)


def _unit(view: dict[str, Any], unit_id: str) -> dict[str, Any] | None:
    for unit in view.get("units", ()):
        if isinstance(unit, dict) and str(unit.get("id")) == unit_id:
            return unit
    return None


def _unit_pos(view: dict[str, Any], unit_id: str) -> tuple[int, int] | None:
    unit = _unit(view, unit_id)
    pos = unit.get("pos") if unit else None
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None
    return int(pos[0]), int(pos[1])


def _coord(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _summary(view: dict[str, Any], unit_id: str) -> dict[str, Any] | None:
    actions = view.get("unit_actions")
    if not isinstance(actions, dict):
        return None
    summary = actions.get(unit_id)
    return summary if isinstance(summary, dict) else None


def _actionable(view: dict[str, Any], unit_id: str) -> bool:
    summary = _summary(view, unit_id)
    if summary is None:
        return False
    return not (
        summary.get("acted")
        or summary.get("loaded_in")
        or summary.get("new_unit")
    )


def _move_order(
    view: dict[str, Any],
    unit_id: str,
    destination: tuple[int, int],
    used: set[tuple[int, int]],
) -> dict[str, Any] | None:
    if destination in used:
        return None
    summary = _summary(view, unit_id)
    if summary is None or summary.get("acted") or summary.get("loaded_in") or summary.get("new_unit"):
        return None
    moves = summary.get("move_to") or []
    if not any(_coord(row[0]) == destination for row in moves if isinstance(row, list) and row):
        return None
    used.add(destination)
    return {"action": "move", "unit": unit_id, "to": list(destination)}


def _load_order(view: dict[str, Any], transport: str, cargo: str) -> dict[str, Any] | None:
    summary = _summary(view, transport)
    if summary is None or summary.get("acted") or summary.get("loaded_in") or summary.get("new_unit"):
        return None
    if cargo not in (summary.get("load") or []):
        return None
    return {"action": "load", "unit": transport, "cargo": cargo}


def _unload_order(
    view: dict[str, Any], transport: str, cargo: str, target: tuple[int, int]
) -> dict[str, Any] | None:
    summary = _summary(view, transport)
    if summary is None or summary.get("acted") or summary.get("loaded_in") or summary.get("new_unit"):
        return None
    options = summary.get("unload") or []
    if not any(
        isinstance(row, list)
        and len(row) >= 2
        and row[0] == cargo
        and _coord(row[1]) == target
        for row in options
    ):
        return None
    return {"action": "unload", "unit": transport, "cargo": cargo, "target": list(target)}


def plan_ferry(view: dict[str, Any]) -> FerryPlan:
    """Execute the best ferry recommendation for each transport.

    A transport and its cargo are claimed even when the hint cannot be issued
    this turn.  That keeps the ordinary greedy policy from walking a stranded
    unit into the sea (or sailing an empty transport at the enemy) while the
    boat works through a multi-turn crossing.
    """
    ferry = view.get("ferry")
    if not isinstance(ferry, dict):
        return FerryPlan()
    claimed = {
        str(unit_id)
        for unit_id in (ferry.get("transports") or ())
        if isinstance(unit_id, str)
    } | {
        str(unit_id)
        for unit_id in (ferry.get("stranded") or ())
        if isinstance(unit_id, str)
    }
    plan = FerryPlan(claimed=claimed)
    used = plan.destinations
    processed: set[str] = set()
    assigned_cargo: set[str] = set()

    for raw in ferry.get("operations") or ():
        if not isinstance(raw, dict):
            continue
        transport = str(raw.get("transport") or "")
        cargo = str(raw.get("cargo") or "")
        action = str(raw.get("action") or "")
        if not transport or not cargo or transport in processed or cargo in assigned_cargo:
            continue
        processed.add(transport)
        assigned_cargo.add(cargo)
        if not _actionable(view, transport):
            continue
        if action in ("load", "pickup") and not _actionable(view, cargo):
            continue

        if action == "load":
            order = _load_order(view, transport, cargo)
            if order is not None:
                plan.orders.append(order)
            continue

        if action == "unload":
            target = _coord(raw.get("at"))
            if target is None:
                continue
            order = _unload_order(view, transport, cargo, target)
            if order is not None:
                plan.orders.append(order)
            continue

        if action == "pickup":
            cargo_to = _coord(raw.get("cargo_to"))
            transport_to = _coord(raw.get("transport_to"))
            cargo_pos = _unit_pos(view, cargo)
            transport_pos = _unit_pos(view, transport)
            staged: list[dict[str, Any]] = []
            if cargo_to is not None and cargo_to != cargo_pos:
                order = _move_order(view, cargo, cargo_to, used)
                if order is None:
                    continue
                staged.append(order)
            if transport_to is not None and transport_to != transport_pos:
                order = _move_order(view, transport, transport_to, used)
                if order is None:
                    continue
                staged.append(order)
            if not staged:
                load = _load_order(view, transport, cargo)
                if load is None:
                    continue
                staged.append(load)
            else:
                # The load is legal from the refreshed post-move position; the
                # current view's load list is intentionally stale here.
                staged.append({"action": "load", "unit": transport, "cargo": cargo})
            plan.orders.extend(staged)
            continue

        if action == "reposition":
            transport_to = _coord(raw.get("transport_to"))
            transport_pos = _unit_pos(view, transport)
            if transport_to is not None and transport_to != transport_pos:
                order = _move_order(view, transport, transport_to, used)
                if order is not None:
                    plan.orders.append(order)
            continue

        if action == "sail":
            transport_to = _coord(raw.get("transport_to"))
            at = _coord(raw.get("at"))
            transport_pos = _unit_pos(view, transport)
            if transport_to is not None and transport_to != transport_pos:
                order = _move_order(view, transport, transport_to, used)
                if order is None:
                    continue
                plan.orders.append(order)
            if at is not None:
                if transport_to is not None and transport_to != transport_pos:
                    # The unload list is from the current position.  The hint
                    # guarantees the beach after the move; the coordinator
                    # validates the order against the refreshed state.
                    plan.orders.append(
                        {
                            "action": "unload",
                            "unit": transport,
                            "cargo": cargo,
                            "target": list(at),
                        }
                    )
                else:
                    order = _unload_order(view, transport, cargo, at)
                    if order is not None:
                        plan.orders.append(order)
            continue

    return plan
