"""Landmass and ferry hints for agents.

A one-step action list cannot express the three-turn shape of an amphibious
operation.  This module adds an advisory layer on top of the rules: it groups
the board into connected landmasses, marks land units with no capturable site
on their own landmass, and names the transport, cargo, pickup hex and landing
beach that would start or finish a crossing.

Nothing here changes legality.  Every coordinate in a hint is taken from the
engine's own ``legal_moves`` / ``load_options`` / ``unload_options`` results,
or is checked with the same movement and stacking rules before it is returned.
The hints are deliberately compact and deterministic so they can be embedded in
the JSON view and the NCM state codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hexgrid import Hex, hex_distance, hex_neighbors
from .rules import legal_moves, load_options, unload_options
from .state import GameState, Unit, can_share_hex

__all__ = ["LandmassIndex", "ferry_hints", "landmass_index"]

#: Movement types that belong on the land layer.  Air and sea units are not
#: assigned a landmass id.
LAND_MOVEMENTS = frozenset({"foot", "wheel", "tread"})


@dataclass(frozen=True)
class LandmassIndex:
    """Connected components of foot-passable land, in stable id order."""

    component: dict[Hex, int]
    tiles: dict[int, tuple[Hex, ...]]

    def of(self, pos: Hex | None) -> int | None:
        return self.component.get(tuple(pos)) if pos is not None else None


def landmass_index(state: GameState) -> LandmassIndex:
    """Group the map's land tiles into connected landmasses.

    ``foot`` passability is the common denominator: every ground unit can enter
    those tiles, even when a wheeled or tracked unit cannot cross a particular
    rough or mountain hex.  The hints use this only to answer "same island?";
    exact movement still comes from :func:`tbs.engine.rules.legal_moves`.
    """
    land = {
        pos
        for pos, tile in state.map.tiles.items()
        if state.terrain_of(tile).cost_for("foot") is not None
    }
    component: dict[Hex, int] = {}
    tiles: dict[int, list[Hex]] = {}
    next_id = 0
    for start in sorted(land, key=lambda pos: (pos[1], pos[0])):
        if start in component:
            continue
        component[start] = next_id
        group = [start]
        stack = [start]
        while stack:
            here = stack.pop()
            for nxt in hex_neighbors(here):
                if nxt in land and nxt not in component:
                    component[nxt] = next_id
                    group.append(nxt)
                    stack.append(nxt)
        tiles[next_id] = tuple(sorted(group, key=lambda pos: (pos[1], pos[0])))
        next_id += 1
    return LandmassIndex(component=component, tiles={key: tuple(value) for key, value in tiles.items()})


# ---------------------------------------------------------------------------
# small rule-shaped helpers
# ---------------------------------------------------------------------------


def _is_land(state: GameState, pos: Hex) -> bool:
    tile = state.map.tiles.get(pos)
    return tile is not None and state.terrain_of(tile).cost_for("foot") is not None


def _reaches_component(
    state: GameState,
    start: Hex | None,
    component: int,
    index: LandmassIndex,
) -> bool:
    """Whether a ship can reach a sea hex touching ``component``.

    This is map connectivity, not a one-turn movement check.  A transport
    parked in an inland lake or a blocked harbour should never be given a
    pickup order for a beach it can never leave for.
    """
    if start is None:
        return False
    seen = {start}
    stack = [start]
    while stack:
        here = stack.pop()
        for nxt in hex_neighbors(here):
            tile = state.map.tiles.get(nxt)
            if tile is None or nxt in seen:
                continue
            if state.terrain_of(tile).cost_for("sea") is None:
                continue
            seen.add(nxt)
            stack.append(nxt)
    for sea in seen:
        for land in hex_neighbors(sea):
            if index.of(land) == component and _is_land(state, land):
                return True
    return False


def _can_stand(state: GameState, unit: Unit, pos: Hex) -> bool:
    """Can ``unit`` stand on ``pos`` without a stacking conflict?"""
    tile = state.map.tiles.get(pos)
    if tile is None:
        return False
    if state.terrain_of(tile).cost_for(state.spec_of(unit).movement) is None:
        return False
    for other in state.units_at(pos):
        if other.id == unit.id:
            continue
        if not can_share_hex(state, unit, other):
            return False
    return True


def _positions(state: GameState, unit: Unit) -> list[tuple[Hex, int, list[Hex]]]:
    """Every position ``unit`` could occupy this turn, including its current one.

    The current position is included even after a move because the engine lets a
    moved transport still load or unload, and a moved cargo can still be picked
    up.  Acted units are filtered by the caller.
    """
    out: list[tuple[Hex, int, list[Hex]]] = []
    if unit.pos is not None:
        out.append((unit.pos, 0, [unit.pos]))
    for move in legal_moves(state, unit.id):
        pos = tuple(move["pos"])
        if pos == unit.pos:
            continue
        out.append((pos, int(move["move_cost"]), list(move["path"])))
    return out


def _target_sites(
    state: GameState,
    player: str,
    index: LandmassIndex,
) -> list[tuple[Hex, int, bool]]:
    """Enemy or neutral capturable sites, with their landmass and objective flag."""
    sites: list[tuple[Hex, int, bool]] = []
    for pos, tile in state.map.tiles.items():
        terrain = state.terrain_of(tile)
        if terrain.capture_points is None or tile.owner == player:
            continue
        component = index.of(pos)
        if component is None:
            continue
        sites.append((pos, component, bool(terrain.objective)))
    return sorted(sites, key=lambda item: (item[0][1], item[0][0]))


def _target_for_transport(
    state: GameState,
    transport: Unit,
    sites: list[tuple[Hex, int, bool]],
    index: LandmassIndex,
) -> tuple[Hex, int, bool] | None:
    """The best target component this particular ship can actually reach."""
    for site in sorted(
        sites,
        key=lambda item: (
            0 if item[2] else 1,
            hex_distance(transport.pos, item[0]) if transport.pos is not None else 10**6,
            item[0][1],
            item[0][0],
        ),
    ):
        if _reaches_component(state, transport.pos, site[1], index):
            return site
    return None


def _operation(
    transport: Unit,
    cargo: Unit,
    action: str,
    target: tuple[Hex, int, bool] | None,
    *,
    transport_to: Hex | None = None,
    cargo_to: Hex | None = None,
    at: Hex | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "transport": transport.id,
        "cargo": cargo.id,
        "action": action,
    }
    if transport_to is not None:
        payload["transport_to"] = list(transport_to)
    if cargo_to is not None:
        payload["cargo_to"] = list(cargo_to)
    if at is not None:
        payload["at"] = list(at)
    if target is not None:
        payload["target"] = list(target[0])
    return payload


# ---------------------------------------------------------------------------
# the hint builder
# ---------------------------------------------------------------------------


def _pickup_operation(
    state: GameState,
    transport: Unit,
    target: tuple[Hex, int, bool] | None,
    stranded_ids: set[str],
    index: LandmassIndex,
) -> dict[str, Any] | None:
    """Best empty-transport pickup this turn, if one exists."""
    transport_spec = state.spec_of(transport)
    if transport.acted or transport.loaded or transport.deployed_this_turn:
        return None
    if transport.pos is None or len(transport.cargo) >= transport_spec.capacity:
        return None
    if target is None or not _reaches_component(state, transport.pos, target[1], index):
        return None

    loads = [
        option for option in load_options(state, transport.id)
        if option["cargo"] in stranded_ids
    ]
    if loads:
        # The transport is already next to a cargo; prefer a capture unit.
        choices = []
        for option in loads:
            cargo = state.units[option["cargo"]]
            cargo_spec = state.spec_of(cargo)
            choices.append(
                (
                    0 if cargo_spec.can_capture else 1,
                    cargo.id,
                )
            )
        cargo = state.units[min(choices)[1]]
        return _operation(transport, cargo, "load", target)

    if target is None:
        return None
    target_pos = target[0]

    stranded = []
    for unit in state.units_of(transport.owner):
        if unit.id not in stranded_ids:
            continue
        if unit.loaded or unit.pos is None or unit.acted or unit.deployed_this_turn:
            continue
        spec = state.spec_of(unit)
        if spec.movement not in transport_spec.carries or spec.move <= 0:
            continue
        stranded.append(unit)

    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    transport_positions = _positions(state, transport)
    for cargo in stranded:
        cargo_positions = _positions(state, cargo)
        for land_pos, cargo_cost, _cargo_path in cargo_positions:
            if not _can_stand(state, cargo, land_pos):
                continue
            for sea_pos, sea_cost, _sea_path in transport_positions:
                if hex_distance(sea_pos, land_pos) != 1:
                    continue
                # The cargo must be on ground it can enter, and the transport
                # must be able to carry its movement type.
                if not _is_land(state, land_pos):
                    continue
                operation = _operation(
                    transport,
                    cargo,
                    "pickup",
                    target,
                    transport_to=None if sea_pos == transport.pos else sea_pos,
                    cargo_to=None if land_pos == cargo.pos else land_pos,
                )
                score = (
                    0 if state.spec_of(cargo).can_capture else 1,
                    cargo_cost + sea_cost,
                    hex_distance(land_pos, target_pos),
                    cargo.id,
                    sea_pos[1],
                    sea_pos[0],
                    land_pos[1],
                    land_pos[0],
                )
                if best is None or score < best[0]:
                    best = (score, operation)
    if best is not None:
        return best[1]

    # No pickup can be completed this turn.  If the transport is empty and
    # there are units waiting, sail toward the nearest one instead of idling
    # at the beach; the next turn can complete the pickup.
    if not stranded:
        return None
    cargo = min(
        stranded,
        key=lambda unit: (
            0 if state.spec_of(unit).can_capture else 1,
            hex_distance(transport.pos, unit.pos) if unit.pos is not None else 10**6,
            unit.id,
        ),
    )
    if cargo.pos is None:
        return None
    moves = [item for item in _positions(state, transport) if item[0] != transport.pos]
    if not moves:
        return None
    sea_pos, _cost, _path = min(
        moves,
        key=lambda item: (
            hex_distance(item[0], cargo.pos),
            item[1],
            item[0][1],
            item[0][0],
        ),
    )
    return _operation(
        transport,
        cargo,
        "reposition",
        target,
        transport_to=sea_pos,
    )


def _sail_operation(
    state: GameState,
    transport: Unit,
    target: tuple[Hex, int, bool] | None,
    index: LandmassIndex,
) -> dict[str, Any] | None:
    """Best move/unload for a loaded transport this turn."""
    if transport.acted or transport.loaded or transport.deployed_this_turn:
        return None
    if transport.pos is None or not transport.cargo:
        return None
    cargo = state.units[sorted(transport.cargo)[0]]
    cargo_spec = state.spec_of(cargo)

    if target is None:
        return None
    target_pos, target_component, _objective = target
    if not _reaches_component(state, transport.pos, target_component, index):
        return None

    unloads = [
        option
        for option in unload_options(state, transport.id)
        if index.of(tuple(option["at"])) == target_component
    ]
    if unloads:
        option = min(
            unloads,
            key=lambda item: (
                hex_distance(tuple(item["at"]), target_pos),
                item["at"][1],
                item["at"][0],
            ),
        )
        return _operation(
            transport,
            cargo,
            "unload",
            target,
            at=tuple(option["at"]),
        )

    # Move to a sea hex from which the cargo can be dropped onto the target
    # landmass, or if none exists this turn, close the distance to the target.
    best_beach: tuple[tuple[Any, ...], Hex, Hex] | None = None
    best_approach: tuple[tuple[Any, ...], Hex] | None = None
    for sea_pos, sea_cost, _path in _positions(state, transport):
        for land_pos in hex_neighbors(sea_pos):
            if not _is_land(state, land_pos):
                continue
            if index.of(land_pos) != target_component:
                continue
            if state.terrain_of(state.map.tile(land_pos)).cost_for(cargo_spec.movement) is None:
                continue
            if not _can_stand(state, cargo, land_pos):
                continue
            score = (
                hex_distance(land_pos, target_pos),
                sea_cost,
                0 if sea_pos == transport.pos else 1,
                sea_pos[1],
                sea_pos[0],
                land_pos[1],
                land_pos[0],
            )
            if best_beach is None or score < best_beach[0]:
                best_beach = (score, sea_pos, land_pos)
        score = (
            hex_distance(sea_pos, target_pos),
            0 if sea_pos == transport.pos else 1,
            sea_cost,
            sea_pos[1],
            sea_pos[0],
        )
        if best_approach is None or score < best_approach[0]:
            best_approach = (score, sea_pos)

    if best_beach is not None:
        _score, sea_pos, land_pos = best_beach
        return _operation(
            transport,
            cargo,
            "sail",
            target,
            transport_to=None if sea_pos == transport.pos else sea_pos,
            at=land_pos,
        )
    if best_approach is not None and best_approach[1] != transport.pos:
        return _operation(
            transport,
            cargo,
            "sail",
            target,
            transport_to=best_approach[1],
        )
    return None


def ferry_hints(state: GameState, player: str | None = None) -> dict[str, Any] | None:
    """Advisory ferry hints for ``player`` (default: the current player).

    Returns ``None`` when the player has no sea transports and no stranded land
    units.  Otherwise the dictionary has:

    ``stranded``
        Land unit ids with no capturable enemy/neutral site and no enemy unit
        on their landmass.
    ``transports``
        The player's sea transports, including ones with no current operation.
    ``operations``
        One recommended operation per transport, in transport-id order.
    """
    player = player or state.current_player
    index = landmass_index(state)
    targets = _target_sites(state, player, index)
    target_components = {component for _pos, component, _objective in targets}

    enemies_on: set[int] = set()
    for enemy in state.enemies_of(player):
        component = index.of(enemy.pos)
        if component is not None:
            enemies_on.add(component)

    stranded: list[str] = []
    for unit in state.units_of(player):
        if unit.loaded or unit.pos is None:
            continue
        spec = state.spec_of(unit)
        if spec.movement not in LAND_MOVEMENTS or spec.move <= 0:
            continue
        component = index.of(unit.pos)
        if component is None or component in enemies_on:
            continue
        if component not in target_components:
            stranded.append(unit.id)
    stranded.sort()

    transports = sorted(
        unit.id
        for unit in state.units_of(player)
        if unit.pos is not None
        and not unit.loaded
        and state.spec_of(unit).movement == "sea"
        and state.spec_of(unit).capacity > 0
    )
    if not transports and not stranded:
        return None

    operations: list[dict[str, Any]] = []
    for unit_id in transports:
        transport = state.units[unit_id]
        target = _target_for_transport(state, transport, targets, index)
        if transport.cargo:
            operation = _sail_operation(state, transport, target, index)
        else:
            operation = _pickup_operation(state, transport, target, set(stranded), index)
        if operation is not None:
            operations.append(operation)

    return {
        "stranded": stranded,
        "transports": transports,
        "operations": operations,
    }
