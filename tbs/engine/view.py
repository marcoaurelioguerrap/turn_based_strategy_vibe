"""Text views of a game state.

These are the same views an LLM agent is given (milestone M7): the board as
aligned character layers, a per-unit table, and a JSON-able structured view.
The pygame UI renders the very same state from the same fields, so what an
agent "sees" and what a human sees never diverge.

Maps are authored in ``odd-r`` offset layout: row ``r`` of a printed layer is
the authored row, and column ``c`` is the authored column, so a printed cell is
addressed as ``(col=c, row=r)`` and its axial coordinates come from
:func:`cell_coordinates`.  Because odd rows are shifted half a hex right, the
rectangular text layout is *not* the adjacency graph: trust the coordinates and
``rules.legal_moves``, never the drawing.  Unit tables and action paths always
name exact axial coordinates.

Fog of war (M9) will filter these views per player; until then everything is
visible to everyone.
"""

from __future__ import annotations

import json
from typing import Any

from .hexgrid import axial_to_offset, offset_to_axial
from .rules import (
    attack_options,
    build_options,
    capture_options,
    explain_blockers,
    is_supplied,
    legal_moves,
    load_options,
    turn_income,
    unload_options,
)
from .state import GameState


# --------------------------------------------------------------------------
# coordinate layers
# --------------------------------------------------------------------------


def player_glyphs(state: GameState) -> dict[str, str]:
    """One display character per player, preferring the player's own id."""
    glyphs: dict[str, str] = {}
    used: set[str] = set()
    for index, player_id in enumerate(state.player_ids()):
        candidate = player_id[0] if player_id else "?"
        if candidate in used or not candidate.isalnum():
            candidate = str(index + 1)
        while candidate in used:
            candidate = f"{candidate}{index + 1}"
        glyphs[player_id] = candidate
        used.add(candidate)
    return glyphs


def map_rows(state: GameState) -> list[str]:
    """The terrain layer: one character per hex, in authored order."""
    grid: dict[int, dict[int, str]] = {}
    for pos, tile in state.map.tiles.items():
        col, row = axial_to_offset(pos)
        grid.setdefault(row, {})[col] = tile.terrain
    return ["".join(grid[row][col] for col in sorted(grid[row])) for row in sorted(grid)]


def ownership_rows(state: GameState) -> list[str]:
    """The ownership layer: a player glyph, or ``.`` for neutral ground."""
    glyphs = player_glyphs(state)
    grid: dict[int, dict[int, str]] = {}
    for pos, tile in state.map.tiles.items():
        col, row = axial_to_offset(pos)
        grid.setdefault(row, {})[col] = glyphs.get(tile.owner, ".") if tile.owner else "."
    return ["".join(grid[row][col] for col in sorted(grid[row])) for row in sorted(grid)]


def unit_rows(state: GameState) -> list[str]:
    """The unit layer: a unit glyph, or ``.`` where a hex is empty.

    Units loaded inside a transport are not on the board; they appear in the
    unit table instead.
    """
    grid: dict[int, dict[int, str]] = {}
    for pos in state.map.tiles:
        col, row = axial_to_offset(pos)
        grid.setdefault(row, {})[col] = "."
    for unit in state.units.values():
        if unit.pos is None:
            continue
        col, row = axial_to_offset(unit.pos)
        grid[row][col] = state.spec_of(unit).glyph
    return ["".join(grid[row][col] for col in sorted(grid[row])) for row in sorted(grid)]


def cell_coordinates(state: GameState, row: int, col: int) -> dict[str, Any]:
    """Translate a printed cell position into game coordinates."""
    pos = offset_to_axial(col, row)
    tile = state.map.tiles.get(pos)
    unit = state.unit_at(pos) if tile else None
    return {
        "pos": list(pos),
        "offset": [col, row],
        "on_map": tile is not None,
        "terrain": tile.terrain if tile else None,
        "owner": tile.owner if tile else None,
        "unit": unit.id if unit else None,
    }


def render_map(state: GameState, *, show_units: bool = True) -> str:
    """The board as aligned layers, each row ``map.width`` characters."""
    glyphs = player_glyphs(state)
    owners = ", ".join(f"{glyph}={player.name} ({pid})" for pid, player in state.players.items() for glyph in [glyphs[pid]])
    blocks = [
        f"TERRAIN (row r, column index; {state.map.width}x{state.map.height})",
        *map_rows(state),
        "",
        f"OWNERSHIP (. = neutral)  {owners}",
        *ownership_rows(state),
    ]
    if show_units:
        blocks += ["", "UNITS (. = empty hex; see UNIT LETTERS for the glyphs)", *unit_rows(state)]

    captures = [
        f"  {list(tile.pos)} {state.terrain_of(tile).name} {tile.capture_progress}/{tile.capture_points} by {tile.capture_unit}"
        for tile in sorted(state.map.tiles.values(), key=lambda t: (t.pos[1], t.pos[0]))
        if tile.capture_unit is not None and tile.capture_progress
    ]
    if captures:
        blocks += ["", "CAPTURES IN PROGRESS", *captures]
    return "\n".join(blocks)


def unit_legend(state: GameState) -> str:
    """Glyphs of the unit types actually present in this game."""
    present = {state.spec_of(unit).glyph: state.spec_of(unit).name for unit in state.units.values()}
    if not present:
        return "UNIT LETTERS: (no units in play)"
    return "UNIT LETTERS: " + ", ".join(f"{glyph}={name}" for glyph, name in sorted(present.items()))


def terrain_legend(state: GameState) -> str:
    """Every terrain on this map with what it does, for prompt context."""
    used = sorted({tile.terrain for tile in state.map.tiles.values()})
    lines = ["TERRAIN"]
    for code in used:
        spec = state.terrain_of(code)
        costs = ", ".join(f"{movement}:{cost}" for movement, cost in sorted(spec.move_cost.items()))
        extra: list[str] = []
        if spec.build:
            extra.append(f"builds {spec.build}")
        if spec.supply_domains:
            extra.append(f"supplies {'/'.join(spec.supply_domains)}")
        if spec.capture_points:
            extra.append(f"capture {spec.capture_points}")
        if spec.income:
            extra.append(f"income {spec.income}")
        if spec.objective:
            extra.append("OBJECTIVE: capturing it wins")
        lines.append(
            f"  {code} {spec.name:14} cover {spec.defense_pct:>3}%  move: {costs}"
            + ("  " + "; ".join(extra) if extra else "")
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------


def describe_unit(state: GameState, unit_id: str) -> str:
    """One line describing a unit's situation, with exact coordinates."""
    unit = state.unit(unit_id)
    spec = state.spec_of(unit)
    if unit.pos is None:
        where = f"loaded in {unit.carried_by}"
    else:
        col, row = axial_to_offset(unit.pos)
        where = f"at {list(unit.pos)} (col {col} row {row})"
    ammo = ", ".join(f"{weapon}:{amount}" for weapon, amount in sorted(unit.ammo.items())) or "none"
    fuel = "n/a" if unit.fuel is None else str(unit.fuel)
    flags: list[str] = []
    if unit.moved:
        flags.append("moved")
    if unit.acted:
        flags.append("acted")
    if unit.deployed_this_turn:
        flags.append("new")
    if unit.loaded:
        flags.append("loaded")
    if unit.cargo:
        flags.append(f"carrying {'+'.join(unit.cargo)}")
    supplied = "yes" if is_supplied(state, unit) else "NO"
    label = f" [{unit.label}]" if unit.label else ""
    return (
        f"{unit.id:<4} {spec.glyph} {spec.name:<15}{label:<22} {where:<26} "
        f"hp {unit.hp:>3}/{spec.max_hp:<3} ammo {ammo:<24} fuel {fuel:>4} supply {supplied:>3} "
        f"status [{', '.join(flags) or 'ready'}]"
    )


def render_units(state: GameState, player_id: str | None = None) -> str:
    if player_id is None:
        units = sorted(state.units.values(), key=lambda u: (u.owner, u.id))
    else:
        units = state.units_of(player_id)
    if not units:
        return "(no units)"
    return "\n".join(describe_unit(state, unit.id) for unit in units)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def render_status(state: GameState) -> str:
    lines = [
        f"scenario {state.scenario_name!r} seed {state.seed} turn {state.turn_number} "
        f"status {state.status}" + (f" winner {state.winner}" if state.winner else ""),
    ]
    for player_id, player in state.players.items():
        lines.append(
            f"  {player.id} {player.name:<16} funds {player.funds:>6} income/turn {turn_income(state, player_id):>5} "
            f"units {len(state.units_of(player_id)):>2} sites {len(state.owned_tiles(player_id)):>2}"
            + ("  ELIMINATED" if player.eliminated else "")
        )
    objectives = [
        f"  {list(tile.pos)} {tile.owner or 'neutral'} HQ (objective)"
        for tile in sorted(state.map.tiles.values(), key=lambda t: (t.pos[1], t.pos[0]))
        if tile.objective
    ]
    if objectives:
        lines.append("OBJECTIVES (capturing one ends the game):")
        lines.extend(objectives)
    return "\n".join(lines)


def unit_options_text(state: GameState, unit_id: str) -> str:
    """What this unit can be ordered to do, as readable text."""
    unit = state.unit(unit_id)
    lines = [f"orders available to {unit.id}:"]
    moves = legal_moves(state, unit_id)
    if moves:
        lines.append(f"  move: {len(moves)} legal destinations, cheapest first")
        for move in moves[:12]:
            lines.append(f"    -> {move['pos']} (cost {move['move_cost']}, path {move['path']})")
        if len(moves) > 12:
            lines.append(f"    ... {len(moves) - 12} more (see rules.legal_moves)")
    else:
        lines.append("  move: none")
    for option in attack_options(state, unit_id):
        counter = option.get("counterattack")
        note = f", defender counters with {counter['weapon']} for about {counter['preview_damage']}" if counter else ", no counterattack"
        lines.append(
            f"  attack {option['target']} ({option['target_type']}) at range {option['distance']} "
            f"with {option['weapon']}: about {option['preview_damage']} damage{note}"
        )
    for option in capture_options(state, unit_id):
        done = " - captures it now" if option["completes_now"] else ""
        lines.append(
            f"  capture {option['terrain']} at {option['pos']}: progress {option['progress']}/{option['needed']}{done}"
        )
    for option in load_options(state, unit_id):
        lines.append(f"  load {option['cargo']} ({option['cargo_type']}) from {option['pos']}")
    for option in unload_options(state, unit_id):
        lines.append(f"  unload {option['cargo']} onto {option['at']}")
    for blocker in explain_blockers(state, unit_id):
        lines.append(f"  note: {blocker}")
    if not unit.acted:
        lines.append("  or hold (spend its turn doing nothing)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# structured view: the payload an agent is prompted with
# --------------------------------------------------------------------------

RULE_SUMMARY = (
    "One action per unit per turn: move once then attack/capture/load/unload/hold, or act "
    "without moving; no unit acts twice and none moves after acting. Indirect weapons (min "
    "range > 1) cannot fire after a unit has moved; direct weapons can. Ranged weapons (min "
    "range > 1) neither counterattack nor are countered; range-1 attacks are countered at 50% "
    "power by a surviving defender that can shoot back. Attacking requires supply: stand on or "
    "next to your own city/factory/port/airport/HQ or next to a supply vehicle. Movement and "
    "capture do not need supply. Capturing adds your current HP to the site each time you act "
    "there; capturing an enemy HQ ends the game. Income is paid at the start of your turn and "
    "produced units act next turn. Coordinates are axial [q,r]."
)


def unit_options_summary(state: GameState, unit_id: str) -> dict[str, Any]:
    """A trimmed version of :func:`tbs.engine.rules.unit_options` for prompts."""
    unit = state.unit(unit_id)
    if unit.acted:
        return {"acted": True}
    if unit.loaded:
        return {"loaded_in": unit.carried_by}
    if unit.deployed_this_turn:
        return {"new_unit": True}
    summary: dict[str, Any] = {}
    moves = legal_moves(state, unit_id)
    if moves:
        summary["move_to"] = [[move["pos"], move["move_cost"]] for move in moves]
    attacks = attack_options(state, unit_id)
    if attacks:
        summary["attack"] = [
            [option["target"], option["weapon"], option["distance"], option["preview_damage"]]
            for option in attacks
        ]
    captures = capture_options(state, unit_id)
    if captures:
        summary["capture_remaining_hp"] = captures[0]["needed"] - captures[0]["progress"]
    loads = load_options(state, unit_id)
    if loads:
        summary["load"] = [option["cargo"] for option in loads]
    unloads = unload_options(state, unit_id)
    if unloads:
        summary["unload"] = [[option["cargo"], option["at"]] for option in unloads]
    blockers = explain_blockers(state, unit_id)
    if blockers:
        summary["blockers"] = blockers
    summary["hold"] = True
    return summary


def state_view(state: GameState, player_id: str | None = None) -> dict[str, Any]:
    """A compact JSON-able view of the whole game.

    Map layers give the terrain and ownership; ``sites`` lists every capture or
    build target with its owner; ``units`` gives everything needed to order a
    unit; for the acting player ``unit_actions`` lists what each unit may do.
    """
    asking = player_id or state.current_player
    sites = []
    for tile in sorted(state.map.tiles.values(), key=lambda t: (t.pos[1], t.pos[0])):
        terrain = state.terrain_of(tile.terrain)
        if terrain.capture_points is None and not terrain.build:
            continue
        sites.append(
            {
                "pos": list(tile.pos),
                "terrain": tile.terrain,
                "name": terrain.name,
                "owner": tile.owner,
                "build": terrain.build,
                "capture_progress": tile.capture_progress,
                "capture_needed": tile.capture_points,
                "objective": tile.objective,
            }
        )
    units = []
    for unit in sorted(state.units.values(), key=lambda u: (u.owner, u.id)):
        spec = state.spec_of(unit)
        units.append(
            {
                "id": unit.id,
                "owner": unit.owner,
                "type": unit.type_id,
                "name": spec.name,
                "pos": list(unit.pos) if unit.pos is not None else None,
                "hp": unit.hp,
                "max_hp": spec.max_hp,
                "move": spec.move,
                "ammo": dict(unit.ammo),
                "fuel": unit.fuel,
                "supplied": is_supplied(state, unit),
                "cargo": list(unit.cargo),
                "loaded": unit.loaded,
                "moved": unit.moved,
                "acted": unit.acted,
                "new": unit.deployed_this_turn,
            }
        )

    view: dict[str, Any] = {
        "scenario": state.scenario_name,
        "seed": state.seed,
        "turn": state.turn_number,
        "status": state.status,
        "winner": state.winner,
        "current_player": state.current_player,
        "asking_about": asking,
        "map_size": [state.map.width, state.map.height],
        "map_terrain_rows": map_rows(state),
        "map_ownership_rows": ownership_rows(state),
        "map_unit_rows": unit_rows(state),
        "coordinates": "axial [q,r]; map rows are authored rows, column index is the authoring column; rules.legal_moves gives exact paths",
        "rules": RULE_SUMMARY,
        "players": [
            {
                "id": player.id,
                "name": player.name,
                "funds": player.funds,
                "income_per_turn": turn_income(state, player.id),
                "eliminated": player.eliminated,
                "unit_count": len(state.units_of(player.id)),
            }
            for player in state.players.values()
        ],
        "sites": sites,
        "units": units,
    }
    if state.status == "ongoing":
        view["build_options"] = build_options(state, state.current_player)
        view["unit_actions"] = {
            unit["id"]: unit_options_summary(state, unit["id"])
            for unit in view["units"]
            if unit["owner"] == state.current_player
        }
    return view


def view_json(state: GameState, player_id: str | None = None) -> str:
    """The structured view as a compact JSON string, ready for a prompt."""
    return json.dumps(state_view(state, player_id), sort_keys=True, separators=(",", ":"))


def render_log(state: GameState, limit: int = 20) -> str:
    """The tail of the action log, for reviewing what happened."""
    entries = state.log[-limit:]
    if not entries:
        return "(no actions yet)"
    lines = [
        f"seq {entry['seq']} turn {entry['turn']} {entry['player']}: {entry['describe']}"
        for entry in entries
    ]
    for entry in entries:
        for event in entry["events"]:
            if event.get("type") in {"attack", "counterattack", "destroyed", "captured", "build", "eliminated", "objective_captured"}:
                lines.append(f"    {event}")
    return "\n".join(lines)


def render_game(state: GameState) -> str:
    """Everything a human would need to play the current position."""
    return "\n\n".join(
        [
            render_status(state),
            render_map(state),
            unit_legend(state),
            terrain_legend(state),
            render_units(state),
        ]
    )
