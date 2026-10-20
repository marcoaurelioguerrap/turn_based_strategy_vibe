"""Hover text for a map cell.  Pure: no pygame import.

The window shows exactly this text when the mouse rests on a hex: terrain,
owner, cover, capture/income and — if a unit stands there — its condition,
ammo, fuel, supply and this-turn status.  All values come from the engine.
"""

from __future__ import annotations

from tbs.engine import rules
from tbs.engine.hexgrid import Hex, axial_to_offset
from tbs.engine.state import GameState


def tile_lines(state: GameState, pos: Hex) -> list[str]:
    tile = state.map.tiles.get(tuple(pos))
    if tile is None:
        return []
    terrain = state.terrain_of(tile)
    col, row = axial_to_offset(tile.pos)
    owner = state.players[tile.owner].name if tile.owner else "neutral"
    lines = [
        f"{terrain.name}  ({col}, {row})",
        f"owner: {owner}   cover {terrain.defense_pct}%",
    ]
    if terrain.build:
        lines.append(f"builds: {terrain.build}")
    if terrain.capture_points:
        lines.append(f"capture {tile.capture_progress}/{terrain.capture_points}")
        if terrain.income:
            lines.append(f"income {terrain.income}/turn")
    if terrain.supply_domains:
        lines.append(f"supplies: {'/'.join(terrain.supply_domains)}")
    if tile.objective:
        lines.append("OBJECTIVE: capture to win")
    return lines


def unit_lines(state: GameState, pos: Hex) -> list[str]:
    units = state.units_at(pos)
    if not units:
        return []
    lines: list[str] = []
    for index, unit in enumerate(units):
        if index:
            lines.append("")
        spec = state.spec_of(unit)
        supplied = "yes" if rules.is_supplied(state, unit) else "NO"
        lines.extend(
            [
                f"{unit.id}  {spec.name}",
                f"hp {unit.hp}/{spec.max_hp}   rearm {supplied}",
            ]
        )
        ammo = ", ".join(
            f"{weapon.id} {unit.ammo.get(weapon.id, 0)}/{weapon.ammo}" for weapon in spec.weapons
        ) or "none"
        lines.append(f"ammo {ammo}")
        fuel = "n/a" if unit.fuel is None else str(unit.fuel)
        lines.append(f"fuel {fuel}")
        status: list[str] = []
        if unit.acted:
            status.append("acted")
        if unit.moved and not unit.acted:
            status.append("moved")
        if unit.deployed_this_turn:
            status.append("new")
        if unit.loaded:
            status.append(f"loaded in {unit.carried_by}")
        if unit.cargo:
            status.append(f"carrying {', '.join(unit.cargo)}")
        lines.append("status: " + (", ".join(status) if status else "ready"))
    return lines


def hover_lines(state: GameState, pos: Hex) -> list[str]:
    """Everything the tooltip shows for one hex: tile, then the unit if any."""
    lines = tile_lines(state, pos)
    unit = unit_lines(state, pos)
    if lines and unit:
        lines.append("")
    return lines + unit
