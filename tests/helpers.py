"""Test helpers: build small scenarios without touching the bundled maps.

Test maps are authored the way real maps are (text rows) and units are placed
with ``offset`` coordinates, because counting characters in a row is far easier
to read than axial coordinates.  In the odd-r layout the engine loads, straight
horizontal and vertical neighbours are always adjacent, so these tests stick to
straight lines and never depend on parity-dependent diagonals.  (``test_hexgrid``
is the one place where diagonals are checked directly.)
"""

from __future__ import annotations

from typing import Any

from tbs.engine import rules
from tbs.engine.data import DATA_ROOT, load_scenario, load_specs
from tbs.engine.hexgrid import offset_to_axial
from tbs.engine.state import GameState, Unit, new_game

SPECS = load_specs()

#: 10x4 map of plains, with one city on column 2 of row 1 (no owner).
PLAIN_ROWS = [
    "..........",
    "..C.......",
    "..........",
    "..........",
]

#: Where each side parks a unit that is never involved, so that a test killing
#: one unit does not end the game by elimination.  Both corners are plains.
ANCHORS = {"A": (0, 0), "B": (9, 3)}


def scenario(
    rows: list[str] | None = None,
    units: list[dict[str, Any]] | None = None,
    regions: list[dict[str, Any]] | None = None,
    tile_overrides: list[dict[str, Any]] | None = None,
    players: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A complete scenario dictionary from the pieces a test cares about."""
    data: dict[str, Any] = {
        "schema": 1,
        "name": "test-scenario",
        "seed": 7,
        "first_player": "A",
        "initial_income": 0,
        "max_turns": 50,
        "damage_luck": False,
        "players": players or [
            {"id": "A", "name": "Alpha", "funds": 20000},
            {"id": "B", "name": "Bravo", "funds": 20000},
        ],
        "rows": rows or PLAIN_ROWS,
        "regions": regions or [],
        "tile_overrides": tile_overrides or [],
        "units": units or [],
    }
    data.update(overrides)
    return data


def make_state(**kwargs: Any) -> GameState:
    """Create and start a game from keyword arguments of :func:`scenario`."""
    state = new_game(scenario(**kwargs), SPECS)
    rules.start_game(state)
    return state


def unit(
    type_id: str,
    owner: str,
    col: int,
    row: int,
    unit_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A unit placement expressed in authoring (column, row) coordinates."""
    placement: dict[str, Any] = {"type": type_id, "owner": owner, "offset": [col, row]}
    if unit_id:
        placement["id"] = unit_id
    placement.update(extra)
    return placement


def anchor(owner: str = "B", type_id: str = "infantry") -> dict[str, Any]:
    """A placement far away from the action, keeping ``owner`` in the game."""
    col, row = ANCHORS[owner]
    return unit(type_id, owner, col, row, unit_id=f"{owner}9")


def owned_cells(player_id: str, cells: list[tuple[int, int]]) -> dict[str, Any]:
    """A region claiming specific authoring cells: ``[(col, row), ...]``."""
    return {"owner": player_id, "cells": [[*offset_to_axial(col, row)] for col, row in cells]}


def cell(col: int, row: int) -> list[int]:
    """An authored cell as an axial coordinate list, for use in actions."""
    return [*offset_to_axial(col, row)]


def line(col: int, row: int, steps: int, step: int = 1) -> list[list[int]]:
    """A straight move path of ``steps`` hexes starting at (col, row).

    ``step`` of ``+1`` walks east along a row, ``-1`` walks west.
    """
    return [cell(col + index * step, row) for index in range(steps + 1)]


def column_path(col: int, row: int, steps: int, step: int = 1) -> list[list[int]]:
    """A straight move path walking south (``step=+1``) or north along a column."""
    return [cell(col, row + index * step) for index in range(steps + 1)]


def move(unit_id: str, path: list[list[int]]) -> dict[str, Any]:
    return {"action": "move", "unit": unit_id, "path": path}


def attack(unit_id: str, target: str, weapon: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"action": "attack", "unit": unit_id, "target": target}
    if weapon:
        data["weapon"] = weapon
    return data


def capture(unit_id: str) -> dict[str, Any]:
    return {"action": "capture", "unit": unit_id}


def load(unit_id: str, cargo: str) -> dict[str, Any]:
    return {"action": "load", "unit": unit_id, "cargo": cargo}


def unload(unit_id: str, cargo: str, target: list[int]) -> dict[str, Any]:
    return {"action": "unload", "unit": unit_id, "cargo": cargo, "target": target}


def build(col: int, row: int, unit_type: str) -> dict[str, Any]:
    return {"action": "build", "site": cell(col, row), "unit": unit_type}


def hold(unit_id: str) -> dict[str, Any]:
    return {"action": "hold", "unit": unit_id}


def end_turn() -> dict[str, Any]:
    return {"action": "end_turn"}


def do(state: GameState, action: dict[str, Any]) -> list[dict]:
    """Shortcut for :func:`tbs.engine.rules.perform`."""
    return rules.perform(state, action)


def at(state: GameState, col: int, row: int) -> Unit:
    """The unit standing on an authored cell (raises if there is none)."""
    target = state.unit_at(offset_to_axial(col, row))
    if target is None:
        raise AssertionError(f"no unit at column {col}, row {row}")
    return target


def tile_at(state: GameState, col: int, row: int):
    return state.map.tile(offset_to_axial(col, row))


def types_of(state: GameState, owner: str) -> list[str]:
    """The unit types a player owns, sorted, for asserting what is alive."""
    return sorted(unit.type_id for unit in state.units_of(owner))


def events_of(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event["type"] == kind]


def bundled_path():
    return DATA_ROOT / "maps" / "border_skirmish.json"


def bundled_scenario() -> dict[str, Any]:
    return load_scenario(bundled_path())
