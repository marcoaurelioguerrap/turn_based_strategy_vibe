"""The council roster: doctrines, unit slices and prompt-sized views.

A general is deliberately data, not a subclass: its name, doctrine and units are
a :class:`GeneralSpec`, and the same ``SingleLlmAgent`` asks the model to play
any of them.  That keeps the council a composition of ordinary agents and lets a
JSON file replace the roster without touching the engine or the coordinator.

The engine still owns legality.  A general only ever sees its own
``unit_actions``; the council's merger drops anything outside its slice and the
turn coordinator validates what remains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "DEFAULT_GENERALS",
    "GeneralSpec",
    "assign_units",
    "filter_view",
    "general_role",
    "load_generals",
    "select_generals",
]


@dataclass(frozen=True)
class GeneralSpec:
    """One council member: a doctrine and the units it commands."""

    id: str
    name: str
    title: str
    doctrine: str
    unit_types: tuple[str, ...] = ()
    priority: int = 50
    can_build: bool = False
    reserve: bool = False
    max_orders: int | None = None

    def commands(self, unit_type: str | None) -> bool:
        return unit_type is not None and unit_type in self.unit_types


#: The default council, in merge-priority order.  Fleet and Talon are skipped
#: automatically on maps with no air or naval units; Quartermaster is the
#: reserve that catches every unclaimed type and owns production.
DEFAULT_GENERALS: tuple[GeneralSpec, ...] = (
    GeneralSpec(
        id="vanguard",
        name="Vanguard",
        title="the armoured fist",
        doctrine=(
            "Speed and mass win. Concentrate armour on one axis, destroy the most "
            "dangerous unit in reach, and only trade shots you can survive. Do not "
            "split the spearhead to chase distant captures; take ground the infantry "
            "can hold."
        ),
        unit_types=("tank", "artillery", "aa_missile"),
        priority=10,
    ),
    GeneralSpec(
        id="anvil",
        name="Anvil",
        title="the line",
        doctrine=(
            "The line holds. Keep the HQ screened, occupy cover, and take cities "
            "adjacent to your infantry. Never attack a full-health tank frontally; "
            "make the enemy come to you, then punish the exhausted."
        ),
        unit_types=("infantry", "hq"),
        priority=20,
    ),
    GeneralSpec(
        id="talon",
        name="Talon",
        title="the sky marshal",
        doctrine=(
            "Air power is scarce and decisive. Strike artillery, transports, and "
            "wounded high-value units; stay outside anti-air range; do not fly into a "
            "kill zone you cannot leave, and watch fuel."
        ),
        unit_types=("helicopter", "attack_plane", "bomber"),
        priority=30,
    ),
    GeneralSpec(
        id="fleet",
        name="Fleet",
        title="the admiral",
        doctrine=(
            "Control the water. Keep submarines hidden, escort transports, and avoid "
            "shore artillery. Bombard only targets that cannot answer, and do not "
            "trade a capital ship for a city."
        ),
        unit_types=("destroyer", "cruiser", "submarine"),
        priority=35,
    ),
    GeneralSpec(
        id="quartermaster",
        name="Quartermaster",
        title="the ledger",
        doctrine=(
            "Logistics wins long wars. Build only units that fill a real gap and that "
            "you can afford; keep supply vehicles alive; capture with infantry. Do not "
            "spend the last funds on a tank when the line is thin."
        ),
        unit_types=("apc", "truck", "transport_ship"),
        priority=40,
        can_build=True,
        reserve=True,
    ),
)


def general_role(general: GeneralSpec) -> str:
    """The system-prompt role text for one general."""
    return f"{general.name}, {general.title}. {general.doctrine}"


def assign_units(
    view: dict[str, Any],
    generals: Iterable[GeneralSpec] = DEFAULT_GENERALS,
) -> dict[str, list[str]]:
    """Assign every unit that may still act to exactly one general.

    The view only knows unit ids, types and the current player's ``unit_actions``;
    that is enough.  Units outside every slice fall to the reserve general, so a
    new unit type never disappears from the council's plan.
    """
    roster = tuple(sorted(generals, key=lambda general: (general.priority, general.id)))
    if not roster:
        return {}
    reserve = next((general for general in roster if general.reserve), roster[-1])
    unit_types = {
        str(unit.get("id")): unit.get("type")
        for unit in view.get("units", [])
        if isinstance(unit, dict) and unit.get("id") is not None
    }
    assignments: dict[str, list[str]] = {general.id: [] for general in roster}
    actions = view.get("unit_actions")
    if not isinstance(actions, dict):
        return assignments
    for unit_id in sorted(actions):
        summary = actions[unit_id]
        if not isinstance(summary, dict):
            continue
        if summary.get("acted") or summary.get("loaded_in") or summary.get("new_unit"):
            continue
        unit_type = unit_types.get(unit_id)
        owner = next((general for general in roster if general.commands(unit_type)), None)
        if owner is None:
            owner = reserve
        assignments[owner.id].append(unit_id)
    return assignments


def filter_view(
    view: dict[str, Any],
    allowed_units: Iterable[str] | None,
    *,
    can_build: bool = False,
) -> dict[str, Any]:
    """A copy of ``view`` with only this general's actions (and optional builds).

    Units, sites and both map layers stay visible so a general can reason about
    the whole battlefield without being able to order anyone else's troops.
    """
    filtered = dict(view)
    actions = view.get("unit_actions")
    if isinstance(actions, dict):
        allowed = set(allowed_units or [])
        filtered["unit_actions"] = {
            unit_id: summary for unit_id, summary in actions.items() if unit_id in allowed
        }
    if not can_build:
        filtered.pop("build_options", None)
    return filtered


def select_generals(
    generals: Iterable[GeneralSpec],
    ids: Iterable[str] | None = None,
) -> tuple[GeneralSpec, ...]:
    """Return the roster in priority order, optionally narrowed to ``ids``."""
    roster = tuple(sorted(generals, key=lambda general: (general.priority, general.id)))
    wanted = {str(item).strip() for item in (ids or []) if str(item).strip()}
    if not wanted:
        return roster
    known = {general.id for general in roster}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(
            f"unknown general(s) {', '.join(unknown)}; available: {', '.join(sorted(known))}"
        )
    return tuple(general for general in roster if general.id in wanted)


def load_generals(path: str | Path) -> tuple[GeneralSpec, ...]:
    """Load a roster from JSON, e.g. ``{"generals":[{...}, ...]}``.

    Unknown keys are ignored so a config can carry notes.  Bad input raises a
    one-sentence ``ValueError`` naming the file and the problem.
    """
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read general roster {source}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"general roster {source} is not valid JSON: {exc}") from None
    entries = data.get("generals") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'general roster {source} needs a non-empty "generals" list')
    loaded: list[GeneralSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"general roster {source}: generals[{index}] must be an object")
        general_id = str(entry.get("id", "")).strip()
        if not general_id:
            raise ValueError(f'general roster {source}: generals[{index}] needs an "id"')
        if general_id in seen:
            raise ValueError(f"general roster {source}: duplicate id {general_id!r}")
        seen.add(general_id)
        name = str(entry.get("name") or general_id)
        title = str(entry.get("title") or "")
        doctrine = str(entry.get("doctrine") or "")
        if not doctrine.strip():
            raise ValueError(
                f"general roster {source}: general {general_id!r} needs a non-empty doctrine"
            )
        raw_types = entry.get("unit_types", [])
        if not isinstance(raw_types, list):
            raise ValueError(
                f"general roster {source}: general {general_id!r} unit_types must be a list"
            )
        max_orders = entry.get("max_orders")
        if max_orders is not None:
            try:
                max_orders = int(max_orders)
            except (TypeError, ValueError):
                raise ValueError(
                    f"general roster {source}: general {general_id!r} max_orders must be an integer"
                ) from None
        try:
            priority = int(entry.get("priority", 50))
        except (TypeError, ValueError):
            raise ValueError(
                f"general roster {source}: general {general_id!r} priority must be an integer"
            ) from None
        loaded.append(
            GeneralSpec(
                id=general_id,
                name=name,
                title=title,
                doctrine=doctrine.strip(),
                unit_types=tuple(str(item) for item in raw_types),
                priority=priority,
                can_build=bool(entry.get("can_build", False)),
                reserve=bool(entry.get("reserve", False)),
                max_orders=max_orders,
            )
        )
    if not any(general.reserve for general in loaded):
        # The last general is already the fallback; flagging it keeps the
        # assignment explicit in logs and configs.
        loaded[-1] = replace(loaded[-1], reserve=True)
    return tuple(sorted(loaded, key=lambda general: (general.priority, general.id)))
