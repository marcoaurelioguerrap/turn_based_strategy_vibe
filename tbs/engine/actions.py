"""Action types and their JSON representation.

One action = one unit activation (or one production order, or ending a turn).
Agents submit them as JSON objects; :func:`action_from_dict` turns them into the
validated dataclasses used by :mod:`tbs.engine.rules`.

    {"action": "move",   "unit": "A3", "path": [[0,0],[1,0],[1,1]]}
    {"action": "attack", "unit": "A3", "target": "B7", "weapon": "cannon"}
    {"action": "capture","unit": "A3"}
    {"action": "load",   "unit": "A5", "cargo": "A3"}
    {"action": "unload", "unit": "A5", "cargo": "A3", "target": [4,2]}
    {"action": "build",  "site": [6,1], "unit": "tank"}
    {"action": "hold",   "unit": "A3"}
    {"action": "end_turn"}

All coordinates are axial ``[q, r]``.  Actions are immutable so that a swarm
orchestrator can safely hold, sort and replay them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hexgrid import Hex

ACTION_TYPES = ("move", "attack", "capture", "load", "unload", "build", "hold", "end_turn")


def _hex(value: Any) -> Hex:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected a coordinate as [q,r], got {value!r}")
    return (int(value[0]), int(value[1]))


@dataclass(frozen=True)
class MoveAction:
    unit: str
    path: tuple[Hex, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"action": "move", "unit": self.unit, "path": [list(pos) for pos in self.path]}


@dataclass(frozen=True)
class AttackAction:
    unit: str
    target: str
    weapon: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"action": "attack", "unit": self.unit, "target": self.target}
        if self.weapon is not None:
            data["weapon"] = self.weapon
        return data


@dataclass(frozen=True)
class CaptureAction:
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": "capture", "unit": self.unit}


@dataclass(frozen=True)
class LoadAction:
    """Load an adjacent friendly unit into a transport."""

    unit: str
    cargo: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": "load", "unit": self.unit, "cargo": self.cargo}


@dataclass(frozen=True)
class UnloadAction:
    """Drop loaded cargo onto an empty hex adjacent to the transport."""

    unit: str
    cargo: str
    target: Hex

    def to_dict(self) -> dict[str, Any]:
        return {"action": "unload", "unit": self.unit, "cargo": self.cargo, "target": list(self.target)}


@dataclass(frozen=True)
class BuildAction:
    """Produce a new unit on an owned, empty build site."""

    site: Hex
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": "build", "site": list(self.site), "unit": self.unit}


@dataclass(frozen=True)
class HoldAction:
    """Do nothing with this unit; its turn is spent."""

    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": "hold", "unit": self.unit}


@dataclass(frozen=True)
class EndTurnAction:
    """Finish the current player's turn."""

    def to_dict(self) -> dict[str, Any]:
        return {"action": "end_turn"}


Action = (
    MoveAction
    | AttackAction
    | CaptureAction
    | LoadAction
    | UnloadAction
    | BuildAction
    | HoldAction
    | EndTurnAction
)


def action_from_dict(data: dict[str, Any]) -> Action:
    """Parse an action dictionary, raising ``ValueError`` with a helpful message."""
    if not isinstance(data, dict):
        raise ValueError(f"an action must be an object, got {data!r}")
    kind = data.get("action")
    if kind not in ACTION_TYPES:
        raise ValueError(f"unknown action {kind!r}; valid actions: {', '.join(ACTION_TYPES)}")
    try:
        if kind == "move":
            path = tuple(_hex(pos) for pos in data["path"])
            return MoveAction(unit=str(data["unit"]), path=path)
        if kind == "attack":
            weapon = data.get("weapon")
            return AttackAction(unit=str(data["unit"]), target=str(data["target"]), weapon=None if weapon is None else str(weapon))
        if kind == "capture":
            return CaptureAction(unit=str(data["unit"]))
        if kind == "load":
            return LoadAction(unit=str(data["unit"]), cargo=str(data["cargo"]))
        if kind == "unload":
            return UnloadAction(unit=str(data["unit"]), cargo=str(data["cargo"]), target=_hex(data["target"]))
        if kind == "build":
            return BuildAction(site=_hex(data["site"]), unit=str(data["unit"]))
        if kind == "hold":
            return HoldAction(unit=str(data["unit"]))
        return EndTurnAction()
    except KeyError as exc:
        raise ValueError(f"{kind!r} action is missing the {exc.args[0]!r} field") from None


def action_to_dict(action: Action | dict[str, Any]) -> dict[str, Any]:
    if isinstance(action, dict):
        return action
    return action.to_dict()


def describe_action(action: Action | dict[str, Any]) -> str:
    """A one-line human readable description, used in logs and views."""
    data = action_to_dict(action)
    kind = data.get("action")
    if kind == "move":
        path = data["path"]
        return f"{data['unit']} moves {list(path[0]) if path else ''} -> {list(path[-1]) if len(path) > 1 else ''} ({len(path) - 1 if path else 0} hexes)"
    if kind == "attack":
        weapon = f" with {data['weapon']}" if data.get("weapon") else ""
        return f"{data['unit']} attacks {data['target']}{weapon}"
    if kind == "capture":
        return f"{data['unit']} captures the site it stands on"
    if kind == "load":
        return f"{data['unit']} loads {data['cargo']}"
    if kind == "unload":
        return f"{data['unit']} unloads {data['cargo']} at {list(data['target'])}"
    if kind == "build":
        return f"produce {data['unit']} at {list(data['site'])}"
    if kind == "hold":
        return f"{data['unit']} holds position"
    if kind == "end_turn":
        return "end of turn"
    return f"unknown action {kind!r}"
