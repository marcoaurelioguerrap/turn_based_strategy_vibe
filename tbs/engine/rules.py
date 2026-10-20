"""The rules: turn flow, action validation and application, legal-action queries.

The engine is a pure state machine.  :func:`apply_action` mutates the state in
place, appends to ``state.log`` and returns the events caused by that action.
Illegal actions raise :class:`ActionError` with a message that names the
constraint and what to do instead, so a driver (human UI, LLM agent or swarm
orchestrator) can retry immediately.

Turn structure, in short:

* a unit may move once, then act once (attack / capture / load / unload / hold),
  or act without moving; it may not act twice and may not move after acting;
* indirect weapons (``min_range > 1``) cannot fire after moving; direct weapons
  can, so a moved unit may still attack;
* units produced this turn cannot act until the owner's next turn;
* transports may load or unload exactly one cargo per turn;
* a turn ends with the ``end_turn`` action, which pays income, resets flags and
  hands over to the next player.
"""

from __future__ import annotations

from typing import Any, Callable

from . import combat
from .actions import (
    Action,
    AttackAction,
    BuildAction,
    CaptureAction,
    EndTurnAction,
    HoldAction,
    LoadAction,
    MoveAction,
    UnloadAction,
    action_from_dict,
    action_to_dict,
    describe_action,
)
from .data import SUPPLY_DOMAINS
from .errors import ActionError
from .hexgrid import Hex, hex_distance, hex_neighbors, reachable_hexes
from .state import OUT_OF_FUEL_DAMAGE, GameState, Unit, create_unit

__all__ = [
    "apply_action",
    "perform",
    "apply_actions",
    "start_game",
    "legal_moves",
    "attack_options",
    "capture_options",
    "load_options",
    "unload_options",
    "build_options",
    "unit_options",
    "explain_blockers",
    "is_supplied",
    "turn_income",
    "check_victory",
]


# --------------------------------------------------------------------------
# applying actions
# --------------------------------------------------------------------------


def apply_action(state: GameState, action: Action) -> list[dict]:
    """Validate and apply one action, returning the events it produced."""
    if state.status != "ongoing":
        suffix = f" (winner: {state.winner})" if state.winner else ""
        raise ActionError(f"the game is already {state.status}{suffix}")

    if isinstance(action, EndTurnAction):
        player_id = state.current_player
        events = _end_turn(state)
    else:
        player_id, events = _apply_unit_action(state, action)

    state.log.append(
        {
            "seq": state.action_seq,
            "turn": state.turn_number,
            "player": player_id,
            "action": action_to_dict(action),
            "describe": describe_action(action),
            "events": events,
        }
    )
    state.action_seq += 1
    check_victory(state)
    return events


def perform(state: GameState, action: dict[str, Any] | Action) -> list[dict]:
    """Apply an action given as a JSON-style dictionary (the agent-facing entry point)."""
    if isinstance(action, dict):
        try:
            action = action_from_dict(action)
        except ValueError as exc:
            # A malformed action is the agent's mistake, not the engine's: raise the
            # same retryable error type the rules use so a driver can correct it.
            raise ActionError(f"malformed action: {exc}") from None
    return apply_action(state, action)


def apply_actions(state: GameState, actions: list[dict[str, Any] | Action]) -> list[dict]:
    """Apply a batch of actions in order; the first illegal one aborts the batch."""
    events: list[dict] = []
    for item in actions:
        events.extend(perform(state, item))
    return events


def start_game(state: GameState) -> list[dict]:
    """Begin the first player's turn (income, flag resets).  Call right after ``new_game``.

    The units keep exactly the equipment the scenario gave them; resupply only
    starts from the following turn.
    """
    return _begin_turn(state, replenish=False)


def _apply_unit_action(state: GameState, action: Action) -> tuple[str, list[dict]]:
    if isinstance(action, BuildAction):
        return state.current_player, _apply_build(state, action)

    try:
        unit = state.unit(_unit_id(action))
    except KeyError as exc:
        raise ActionError(str(exc)) from None
    if unit.owner != state.current_player:
        raise ActionError(
            f"it is {state.current_player}'s turn; {unit.id} belongs to {unit.owner} and cannot be ordered now"
        )

    handlers: dict[type, Callable[[GameState, Any], list[dict]]] = {
        MoveAction: _apply_move,
        AttackAction: _apply_attack,
        CaptureAction: _apply_capture,
        LoadAction: _apply_load,
        UnloadAction: _apply_unload,
        HoldAction: _apply_hold,
    }
    handler = handlers.get(type(action))
    if handler is None:  # pragma: no cover - action_from_dict cannot produce this
        raise ActionError(f"unsupported action {type(action).__name__}")
    return unit.owner, handler(state, unit, action)


def _unit_id(action: Action) -> str:
    unit_id = getattr(action, "unit", None)
    if unit_id is None:  # pragma: no cover
        raise ActionError("this action needs a 'unit' field")
    return unit_id


# --------------------------------------------------------------------------
# movement
# --------------------------------------------------------------------------


def _occupied_hexes(state: GameState, ignore: Unit | None = None) -> set[Hex]:
    occupied = {unit.pos for unit in state.units.values() if unit.pos is not None}
    if ignore is not None and ignore.pos is not None:
        occupied.discard(ignore.pos)
    return occupied


def _terrain_cost(state: GameState, unit: Unit) -> Callable[[Hex], int | None]:
    movement = state.spec_of(unit).movement

    def cost(pos: Hex) -> int | None:
        tile = state.map.tiles.get(pos)
        if tile is None:
            return None
        return state.terrain_of(tile).cost_for(movement)

    return cost


def _require_can_act(state: GameState, unit: Unit) -> None:
    if unit.acted:
        raise ActionError(
            f"{unit.id} has already acted this turn; order its actions before ending the turn, "
            f"or spend its turn now with 'hold'"
        )
    if unit.deployed_this_turn:
        raise ActionError(f"{unit.id} was produced this turn and cannot act until your next turn")
    if unit.loaded:
        raise ActionError(f"{unit.id} is loaded inside {unit.carried_by}; unload it before it can act")


def _apply_move(state: GameState, unit: Unit, action: MoveAction) -> list[dict]:
    _require_can_act(state, unit)
    spec = state.spec_of(unit)
    if not spec.can_move:
        raise ActionError(f"{spec.name} {unit.id} cannot move (move {spec.move})")
    if unit.moved:
        raise ActionError(f"{unit.id} has already moved this turn; attack, capture, load, unload or hold instead")

    path = list(action.path)
    if len(path) < 2:
        raise ActionError(
            "a move needs at least two hexes: the unit's current position, then one step; "
            "to act without moving use attack, capture, load, unload or hold"
        )
    if path[0] != unit.pos:
        raise ActionError(f"the move path must start at {unit.id}'s position {list(unit.pos)}, not {list(path[0])}")

    cost_of = _terrain_cost(state, unit)
    occupied = _occupied_hexes(state, ignore=unit)
    total = 0
    for from_hex, to_hex in zip(path, path[1:]):
        if from_hex not in state.map.tiles:
            raise ActionError(f"{list(from_hex)} is not part of the map")
        if to_hex not in state.map.tiles:
            raise ActionError(f"{list(to_hex)} is not part of the map; a move has to stay inside the map")
        if hex_distance(from_hex, to_hex) != 1:
            raise ActionError(f"{list(from_hex)} and {list(to_hex)} are not adjacent; move one hex at a time")
        step_cost = cost_of(to_hex)
        if step_cost is None:
            tile = state.map.tiles[to_hex]
            raise ActionError(
                f"{spec.name} {unit.id} ({spec.movement}) cannot enter {state.terrain_of(tile).name} at {list(to_hex)}"
            )
        occupant = state.unit_at(to_hex)
        if occupant is not None:
            raise ActionError(f"{list(to_hex)} is occupied by {occupant.id}; move around it or attack it instead")
        total += step_cost

    if total > spec.move:
        raise ActionError(
            f"this move costs {total} movement points but {spec.name} {unit.id} only has {spec.move}; "
            f"legal_moves() lists every affordable path"
        )
    if unit.fuel is not None and unit.fuel < total:
        raise ActionError(
            f"this move needs {total} fuel but {unit.id} has {unit.fuel} left; shorten the move or resupply it"
        )

    origin = unit.pos
    unit.pos = path[-1]
    unit.moved = True
    if unit.fuel is not None:
        unit.fuel -= total
    return [
        {
            "type": "move",
            "unit": unit.id,
            "from": list(origin),
            "to": list(unit.pos),
            "move_cost": total,
            "fuel_used": total if unit.fuel is not None else None,
            "fuel_left": unit.fuel,
        }
    ]


# --------------------------------------------------------------------------
# combat
# --------------------------------------------------------------------------


def _apply_attack(state: GameState, unit: Unit, action: AttackAction) -> list[dict]:
    _require_can_act(state, unit)
    try:
        target = state.unit(action.target)
    except KeyError as exc:
        raise ActionError(str(exc)) from None
    if target.owner == unit.owner:
        raise ActionError(f"{unit.id} cannot attack its own side's unit {target.id}")
    if target.pos is None:
        raise ActionError(f"{target.id} is inside a transport and cannot be attacked")

    dist = hex_distance(unit.pos, target.pos)
    weapon = _select_weapon(state, unit, target, action.weapon, dist)
    if not is_supplied(state, unit):
        raise ActionError(
            f"{unit.id} is out of supply and cannot attack; move it next to one of your "
            f"cities, factories, ports, airports or HQs, or next to a supply vehicle (Truck)"
        )

    events = combat.resolve_attack(state, unit, target, weapon)
    unit.acted = True
    return events


def _select_weapon(state: GameState, unit: Unit, target: Unit, requested: str | None, dist: int):
    spec = state.spec_of(unit)
    target_spec = state.spec_of(target)
    if not spec.weapons:
        raise ActionError(f"{spec.name} {unit.id} has no weapons")

    if requested is not None:
        weapon = next((w for w in spec.weapons if w.id == requested), None)
        if weapon is None:
            known = ", ".join(w.id for w in spec.weapons)
            raise ActionError(f"{unit.id} has no weapon {requested!r}; it has: {known}")
        if unit.moved and not weapon.is_direct:
            raise ActionError(
                f"{unit.id} moved this turn: {weapon.id} is an indirect weapon and cannot fire after "
                f"moving; attack with a direct weapon or fire without moving"
            )
        if not weapon.covers(dist):
            raise ActionError(
                f"{weapon.id} reaches {weapon.min_range}-{weapon.max_range} hexes but {target.id} is {dist} away"
            )
        if not weapon.can_hit(target_spec.damage_class):
            raise ActionError(f"{weapon.id} cannot hit a {target_spec.damage_class} ({target.id})")
        if unit.ammo.get(weapon.id, 0) <= 0:
            raise ActionError(f"{weapon.id} is out of ammo; resupply {unit.id} (adjacent Truck or friendly base)")
        return weapon

    options = combat.weapon_options(state, unit, target)
    if options:
        return options[0]
    raise _attack_error(state, unit, target, dist)


def _attack_error(state: GameState, unit: Unit, target: Unit, dist: int) -> ActionError:
    spec = state.spec_of(unit)
    target_spec = state.spec_of(target)
    reasons: list[str] = []
    for weapon in spec.weapons:
        if unit.moved and not weapon.is_direct:
            reasons.append(f"{weapon.id} is indirect and {unit.id} moved this turn")
        elif not weapon.can_hit(target_spec.damage_class):
            reasons.append(f"{weapon.id} cannot hit {target_spec.damage_class}")
        elif not weapon.covers(dist):
            reasons.append(f"{weapon.id} reaches {weapon.min_range}-{weapon.max_range}, {target.id} is {dist} away")
        elif unit.ammo.get(weapon.id, 0) <= 0:
            reasons.append(f"{weapon.id} is out of ammo")
    detail = "; ".join(reasons) if reasons else "it has no weapons"
    return ActionError(f"{unit.id} cannot attack {target.id} at range {dist}: {detail}")


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _apply_capture(state: GameState, unit: Unit, action: CaptureAction) -> list[dict]:
    _require_can_act(state, unit)
    spec = state.spec_of(unit)
    if not spec.can_capture:
        raise ActionError(f"{spec.name} {unit.id} cannot capture sites; only {', '.join(_capture_unit_names(state))} can")
    if unit.pos is None:
        raise ActionError(f"{unit.id} is loaded and cannot capture")
    tile = state.map.tiles[unit.pos]
    terrain = state.terrain_of(tile)
    if tile.capture_points is None:
        raise ActionError(f"{terrain.name} at {list(tile.pos)} cannot be captured")
    if tile.owner == unit.owner:
        raise ActionError(f"{terrain.name} at {list(tile.pos)} is already yours")

    if tile.capture_unit != unit.id:
        tile.capture_unit = unit.id
        tile.capture_progress = 0
    tile.capture_progress += unit.hp
    unit.acted = True

    events = [
        {
            "type": "capture_progress",
            "unit": unit.id,
            "site": list(tile.pos),
            "progress": tile.capture_progress,
            "needed": tile.capture_points,
        }
    ]
    if tile.capture_progress >= tile.capture_points:
        previous_owner = tile.owner
        tile.owner = unit.owner
        tile.capture_progress = 0
        tile.capture_unit = None
        events.append(
            {
                "type": "captured",
                "unit": unit.id,
                "site": list(tile.pos),
                "terrain": terrain.name,
                "previous_owner": previous_owner,
                "new_owner": unit.owner,
            }
        )
        if tile.objective:
            state.status = "victory"
            state.winner = unit.owner
            events.append({"type": "objective_captured", "site": list(tile.pos), "winner": unit.owner})
    return events


def _capture_unit_names(state: GameState) -> list[str]:
    return sorted({spec.name for spec in state.specs.units.values() if spec.can_capture})


# --------------------------------------------------------------------------
# transports
# --------------------------------------------------------------------------


def _apply_load(state: GameState, unit: Unit, action: LoadAction) -> list[dict]:
    _require_can_act(state, unit)
    transport_spec = state.spec_of(unit)
    if not transport_spec.is_transport:
        raise ActionError(f"{transport_spec.name} {unit.id} cannot carry units (capacity {transport_spec.capacity})")
    try:
        cargo = state.unit(action.cargo)
    except KeyError as exc:
        raise ActionError(str(exc)) from None
    if cargo.owner != unit.owner:
        raise ActionError(f"{unit.id} can only load its own side's units, not {cargo.id}")
    if cargo.loaded:
        raise ActionError(f"{cargo.id} is already loaded inside {cargo.carried_by}")
    if cargo.id == unit.id:
        raise ActionError("a transport cannot load itself")
    if len(unit.cargo) >= transport_spec.capacity:
        raise ActionError(f"{unit.id} is full ({transport_spec.name} carries {transport_spec.capacity})")
    cargo_spec = state.spec_of(cargo)
    if cargo_spec.movement not in transport_spec.carries:
        raise ActionError(
            f"{transport_spec.name} {unit.id} carries {', '.join(transport_spec.carries) or 'nothing'}; "
            f"it cannot carry a {cargo_spec.name} ({cargo_spec.movement})"
        )
    dist = hex_distance(unit.pos, cargo.pos)
    if dist != 1:
        raise ActionError(f"{cargo.id} is {dist} hexes from {unit.id}; loading needs the transport to be adjacent")

    unit.cargo.append(cargo.id)
    cargo.loaded = True
    cargo.carried_by = unit.id
    cargo.pos = None
    unit.acted = True
    return [{"type": "load", "transport": unit.id, "cargo": cargo.id, "capacity_used": len(unit.cargo)}]


def _apply_unload(state: GameState, unit: Unit, action: UnloadAction) -> list[dict]:
    _require_can_act(state, unit)
    if not unit.cargo:
        raise ActionError(f"{unit.id} is not carrying anything")
    try:
        cargo = state.unit(action.cargo)
    except KeyError as exc:
        raise ActionError(str(exc)) from None
    if cargo.id not in unit.cargo:
        raise ActionError(f"{unit.id} is not carrying {cargo.id}; it carries: {', '.join(unit.cargo) or 'nothing'}")
    target = tuple(action.target)
    if hex_distance(unit.pos, target) != 1:
        raise ActionError(f"unloading needs an empty hex adjacent to {unit.id}; {list(target)} is {hex_distance(unit.pos, target)} away")
    tile = state.map.tiles.get(target)
    if tile is None:
        raise ActionError(f"{list(target)} is not part of the map")
    cargo_spec = state.spec_of(cargo)
    if state.terrain_of(tile).cost_for(cargo_spec.movement) is None:
        raise ActionError(
            f"{cargo_spec.name} cannot be dropped on {state.terrain_of(tile).name} at {list(target)}"
        )
    occupant = state.unit_at(target)
    if occupant is not None:
        raise ActionError(f"{list(target)} is occupied by {occupant.id}; choose an empty hex")

    unit.cargo.remove(cargo.id)
    cargo.loaded = False
    cargo.carried_by = None
    cargo.pos = target
    unit.acted = True
    return [{"type": "unload", "transport": unit.id, "cargo": cargo.id, "at": list(target)}]


# --------------------------------------------------------------------------
# production and holding
# --------------------------------------------------------------------------


def _apply_build(state: GameState, action: BuildAction) -> list[dict]:
    player = state.players[state.current_player]
    site = state.map.tiles.get(action.site)
    if site is None:
        raise ActionError(f"{list(action.site)} is not part of the map")
    terrain = state.terrain_of(site)
    if not terrain.build:
        raise ActionError(f"{terrain.name} at {list(site.pos)} is not a build site (factory, port or airport)")
    if site.owner != player.id:
        owner = site.owner or "nobody"
        raise ActionError(f"you do not own the {terrain.name} at {list(site.pos)} (owner: {owner})")
    if state.unit_at(site.pos) is not None:
        raise ActionError(
            f"the {terrain.name} at {list(site.pos)} is occupied by {state.unit_at(site.pos).id}; "
            "production needs an empty build site"
        )
    try:
        spec = state.spec_of(action.unit)
    except KeyError as exc:
        raise ActionError(str(exc)) from None
    if terrain.build not in spec.build_at:
        where = ", ".join(spec.build_at) or "nowhere"
        raise ActionError(f"{spec.name} can only be produced at: {where} (this site is {terrain.build})")
    if player.funds < spec.cost:
        raise ActionError(f"{spec.name} costs {spec.cost} but you only have {player.funds}")

    unit = create_unit(state, spec.id, player.id, site.pos, deployed_this_turn=True)
    player.funds -= spec.cost
    return [
        {
            "type": "build",
            "unit": unit.id,
            "site": list(site.pos),
            "cost": spec.cost,
            "funds_left": player.funds,
        }
    ]


def _apply_hold(state: GameState, unit: Unit, action: HoldAction) -> list[dict]:
    _require_can_act(state, unit)
    unit.acted = True
    return [{"type": "hold", "unit": unit.id}]


# --------------------------------------------------------------------------
# turn flow
# --------------------------------------------------------------------------


def turn_income(state: GameState, player_id: str) -> int:
    """Income paid at the start of a player's turn: base income plus every owned site."""
    income = state.initial_income
    for tile in state.owned_tiles(player_id):
        income += state.terrain_of(tile).income
    return income


def _replenish(state: GameState, unit: Unit) -> None:
    """A unit that is connected to a supply source tops up its fuel and every
    weapon's ammo at the start of your turn (Advance Wars style).

    Units cut off from their cities, ports, airfields, HQs or Trucks keep what
    they have left, which is what makes cutting a supply line worth doing.
    """
    if not _near_supply_source(state, unit):
        return
    spec = state.spec_of(unit)
    if spec.fuel is not None and unit.fuel is not None:
        unit.fuel = spec.fuel
    for weapon in spec.weapons:
        unit.ammo[weapon.id] = weapon.ammo


def _begin_turn(state: GameState, replenish: bool = True) -> list[dict]:
    """Start ``state.current_player``'s turn: income, flags, replenishment.

    ``replenish`` is off for the very first turn (:func:`start_game`), so that a
    scenario describing a unit that has already been fighting - low fuel, part
    spent ammo, a wounded garrison - is respected exactly as authored.  From the
    second turn on, a unit in supply is brought back up to full.
    """
    player_id = state.current_player
    player = state.players[player_id]
    income = turn_income(state, player_id)
    player.funds += income

    for unit in state.units_of(player_id):
        unit.moved = False
        unit.acted = False
        unit.deployed_this_turn = False
        unit.supplied = is_supplied(state, unit)
        if replenish:
            _replenish(state, unit)

    abandoned: list[list[int]] = []
    for tile in state.map.tiles.values():
        if tile.capture_unit is None:
            continue
        capturer = state.units.get(tile.capture_unit)
        if capturer is None or capturer.pos != tile.pos:
            # A capture is abandoned only when the unit doing it dies or steps
            # off the site.  It is not cancelled merely because the site's own
            # owner begins a turn: taking an enemy base is a job that spans
            # the enemy's turns, and capturing the enemy HQ - the headline
            # victory condition - depends on the progress surviving them.
            tile.capture_unit = None
            tile.capture_progress = 0
            abandoned.append(list(tile.pos))

    return [
        {
            "type": "turn_start",
            "player": player_id,
            "turn": state.turn_number,
            "income": income,
            "funds": player.funds,
            "capture_progress_reset": abandoned,
        }
    ]


def _end_turn(state: GameState) -> list[dict]:
    player_id = state.current_player
    events: list[dict] = []

    for unit in state.units_of(player_id):
        spec = state.spec_of(unit)
        if spec.fuel is None or unit.fuel is None or unit.fuel > 0:
            continue
        if spec.domain not in {"air", "naval"} or unit.pos is None:
            continue
        if _near_supply_source(state, unit):
            # Its crews can top the tank up from a base or a supply vehicle, so
            # running dry here is not yet a crash (Advance Wars makes the same
            # distinction: a dry aircraft next to its airfield is safe).
            continue
        unit.hp -= OUT_OF_FUEL_DAMAGE
        events.append({"type": "out_of_fuel", "unit": unit.id, "damage": OUT_OF_FUEL_DAMAGE, "hp": max(0, unit.hp)})
        if unit.hp <= 0:
            events.extend(combat.destroy_unit(state, unit, cause="ran out of fuel"))

    order = state.player_ids()
    index = order.index(player_id)
    next_player: str | None = None
    wrapped = False
    for step in range(1, len(order) + 1):
        candidate = order[(index + step) % len(order)]
        if state.players[candidate].eliminated:
            continue
        next_player = candidate
        wrapped = index + step >= len(order)
        break

    if next_player is None:
        return events
    if wrapped:
        state.turn_number += 1
    state.current_player = next_player
    events.extend(_begin_turn(state))
    return events


# --------------------------------------------------------------------------
# supply
# --------------------------------------------------------------------------


def is_supplied(state: GameState, unit: Unit) -> bool:
    """Whether a unit may attack: only units that are connected to a supply
    source (see :func:`_near_supply_source`) can fire.

    Capture and movement do not require supply, only attacks do, and a unit with
    no weapons to spend is therefore never "out of supply" as far as an order is
    concerned.  Fuel and replenishment use the stricter :func:`_near_supply_source`.
    """
    spec = state.spec_of(unit)
    if not spec.weapons:
        return True
    domain = spec.domain
    if domain not in SUPPLY_DOMAINS:
        return True
    return _near_supply_source(state, unit)


def _near_supply_source(state: GameState, unit: Unit) -> bool:
    """True when the unit stands on or next to one of its own bases, or next to
    a friendly supply vehicle that covers its domain.  A unit riding in a
    transport inherits whatever the transport itself is connected to."""
    spec = state.spec_of(unit)
    domain = spec.domain
    if unit.pos is None:
        carrier = state.units.get(unit.carried_by)
        if carrier is None:
            return False
        carrier_spec = state.spec_of(carrier)
        if carrier_spec.supply_domains and domain in carrier_spec.supply_domains:
            # Riding inside a supply vehicle is the same as standing next to it.
            return True
        # Inside an armed transport, the cargo shares whatever it is supplied by.
        return _near_supply_source(state, carrier)

    for pos in [unit.pos, *hex_neighbors(unit.pos)]:
        tile = state.map.tiles.get(pos)
        if tile is None or tile.owner != unit.owner:
            continue
        if domain in state.terrain_of(tile).supply_domains:
            return True

    for other in state.units_of(unit.owner, include_loaded=False):
        if other.id == unit.id or other.pos is None:
            continue
        other_spec = state.spec_of(other)
        if not other_spec.supply_domains or domain not in other_spec.supply_domains:
            continue
        if hex_distance(other.pos, unit.pos) <= 1:
            return True
    return False


# --------------------------------------------------------------------------
# victory
# --------------------------------------------------------------------------


def _can_still_fight(state: GameState, player_id: str) -> bool:
    if any(unit.owner == player_id for unit in state.units.values()):
        return True
    costs = [
        spec.cost
        for site in state.build_sites(player_id)
        for spec in state.specs.buildable_units(state.terrain_of(site).build)
    ]
    if not costs:
        return False
    return state.players[player_id].funds >= min(costs)


def check_victory(state: GameState) -> None:
    """Mark eliminations, a win, or a draw after the turn limit."""
    if state.status != "ongoing":
        return

    for player_id in state.player_ids():
        player = state.players[player_id]
        if player.eliminated or _can_still_fight(state, player_id):
            continue
        player.eliminated = True
        state.log.append(
            {
                "seq": state.action_seq,
                "turn": state.turn_number,
                "player": player_id,
                "action": {"action": "eliminated"},
                "describe": f"{player.name} is eliminated: no units left and no way to produce more",
                "events": [{"type": "eliminated", "player": player_id}],
            }
        )

    alive = state.alive_players()
    if not alive:
        state.status = "draw"
    elif len(alive) == 1:
        state.status = "victory"
        state.winner = alive[0]
    elif state.turn_number > int(state.config.get("max_turns", 200)):
        state.status = "draw"


# --------------------------------------------------------------------------
# legal actions, for agents and UIs
# --------------------------------------------------------------------------


def legal_moves(state: GameState, unit_id: str) -> list[dict]:
    """Every legal move destination with its cost and the cheapest path to it."""
    unit = state.unit(unit_id)
    spec = state.spec_of(unit)
    if unit.acted or unit.loaded or unit.deployed_this_turn or not spec.can_move or unit.pos is None:
        return []
    found = reachable_hexes(unit.pos, spec.move, _terrain_cost(state, unit), _occupied_hexes(state, ignore=unit))
    moves: list[dict] = []
    for pos, (cost, path) in sorted(found.items()):
        if pos == unit.pos:
            continue
        if unit.fuel is not None and unit.fuel < cost:
            continue
        moves.append(
            {
                "pos": list(pos),
                "terrain": state.map.tiles[pos].terrain,
                "move_cost": cost,
                "fuel_cost": cost if unit.fuel is not None else None,
                "path": [list(step) for step in path],
            }
        )
    return moves


def attack_options(state: GameState, unit_id: str) -> list[dict]:
    """Every legal attack, with a luck-free damage preview for planning."""
    unit = state.unit(unit_id)
    spec = state.spec_of(unit)
    if unit.acted or unit.loaded or unit.deployed_this_turn or unit.pos is None or not spec.weapons:
        return []
    if not is_supplied(state, unit):
        return []
    options: list[dict] = []
    for target in state.enemies_of(unit.owner):
        for weapon in combat.weapon_options(state, unit, target):
            dist = hex_distance(unit.pos, target.pos)
            entry: dict[str, Any] = {
                "target": target.id,
                "target_type": target.type_id,
                "weapon": weapon.id,
                "distance": dist,
                "preview_damage": combat.preview_damage(state, unit, weapon, target),
                "ammo_left": unit.ammo.get(weapon.id, 0),
            }
            counter = combat.available_weapon(state, target, unit, 1) if dist == 1 and weapon.is_direct else None
            if counter is not None:
                # the defender answers with what is left of it after this shot
                wounded_hp = max(0, target.hp - entry["preview_damage"])
                entry["counterattack"] = {
                    "weapon": counter.id,
                    "preview_damage": combat.preview_damage(
                        state, target, counter, unit, power_pct=combat.COUNTER_POWER_PCT, attacker_hp=wounded_hp
                    ),
                }
            options.append(entry)
    return sorted(options, key=lambda item: (item["target"], item["weapon"]))


def capture_options(state: GameState, unit_id: str) -> list[dict]:
    unit = state.unit(unit_id)
    spec = state.spec_of(unit)
    if unit.acted or unit.loaded or unit.deployed_this_turn or not spec.can_capture or unit.pos is None:
        return []
    tile = state.map.tiles[unit.pos]
    terrain = state.terrain_of(tile)
    if tile.capture_points is None or tile.owner == unit.owner:
        return []
    progress = tile.capture_progress if tile.capture_unit == unit.id else 0
    return [
        {
            "pos": list(tile.pos),
            "terrain": terrain.name,
            "progress": progress,
            "needed": tile.capture_points,
            "completes_now": progress + unit.hp >= tile.capture_points,
            "is_objective": tile.objective,
        }
    ]


def load_options(state: GameState, unit_id: str) -> list[dict]:
    unit = state.unit(unit_id)
    spec = state.spec_of(unit)
    if unit.acted or unit.loaded or unit.deployed_this_turn or not spec.is_transport or unit.pos is None:
        return []
    if len(unit.cargo) >= spec.capacity:
        return []
    options = []
    for pos in hex_neighbors(unit.pos):
        other = state.unit_at(pos)
        if other is None or other.owner != unit.owner or other.loaded:
            continue
        other_spec = state.spec_of(other)
        if other_spec.movement not in spec.carries:
            continue
        options.append({"cargo": other.id, "cargo_type": other.type_id, "pos": list(pos)})
    return sorted(options, key=lambda item: item["cargo"])


def unload_options(state: GameState, unit_id: str) -> list[dict]:
    unit = state.unit(unit_id)
    if unit.acted or unit.loaded or unit.deployed_this_turn or not unit.cargo or unit.pos is None:
        return []
    options: list[dict] = []
    for cargo_id in unit.cargo:
        cargo = state.units[cargo_id]
        cargo_spec = state.spec_of(cargo)
        for pos in hex_neighbors(unit.pos):
            tile = state.map.tiles.get(pos)
            if tile is None or state.unit_at(pos) is not None:
                continue
            if state.terrain_of(tile).cost_for(cargo_spec.movement) is None:
                continue
            options.append({"cargo": cargo.id, "at": list(pos), "terrain": tile.terrain})
    return sorted(options, key=lambda item: (item["cargo"], item["at"]))


def build_options(state: GameState, player_id: str) -> list[dict]:
    """What can be produced this turn, per owned empty build site."""
    player = state.players[player_id]
    options: list[dict] = []
    for site in state.build_sites(player_id):
        if state.unit_at(site.pos) is not None:
            continue
        terrain = state.terrain_of(site)
        for spec in state.specs.buildable_units(terrain.build):
            options.append(
                {
                    "site": list(site.pos),
                    "site_type": terrain.build,
                    "unit": spec.id,
                    "cost": spec.cost,
                    "affordable": player.funds >= spec.cost,
                }
            )
    return sorted(options, key=lambda item: (item["site"], item["cost"]))


def unit_options(state: GameState, unit_id: str) -> dict[str, Any]:
    """Everything this unit could be ordered to do right now, as one dictionary."""
    moves = legal_moves(state, unit_id)
    attacks = attack_options(state, unit_id)
    captures = capture_options(state, unit_id)
    loads = load_options(state, unit_id)
    unloads = unload_options(state, unit_id)
    return {
        "unit": unit_id,
        "can_move": bool(moves),
        "moves": moves,
        "attacks": attacks,
        "captures": captures,
        "loads": loads,
        "unloads": unloads,
        "can_hold": not state.unit(unit_id).acted,
        "blockers": explain_blockers(state, unit_id),
    }


def explain_blockers(state: GameState, unit_id: str) -> list[str]:
    """Why a unit cannot do certain things, phrased so an agent can act on it."""
    unit = state.unit(unit_id)
    spec = state.spec_of(unit)
    blockers: list[str] = []
    if unit.acted:
        blockers.append("already acted this turn")
    if unit.deployed_this_turn:
        blockers.append("newly produced: cannot act until next turn")
    if unit.loaded:
        blockers.append(f"loaded inside {unit.carried_by}")
    if spec.weapons and unit.pos is not None and not unit.acted and not unit.loaded:
        if not is_supplied(state, unit):
            blockers.append(
                "out of supply: attacks need an owned city/factory/port/airport/HQ or a supply "
                "vehicle adjacent to it"
            )
        elif unit.moved and not attack_options(state, unit_id) and _indirect_target_in_range(state, unit):
            blockers.append(
                "moved this turn: indirect weapons cannot fire after moving; attack with a direct "
                "weapon or fire without moving"
            )
        elif not attack_options(state, unit_id):
            blockers.append("no enemy unit is within range of a weapon it has ammo for")
    return blockers


def _indirect_target_in_range(state: GameState, unit: Unit) -> bool:
    """Whether an indirect weapon could have hit something if the unit had stayed put."""
    if unit.pos is None:
        return False
    for weapon in state.spec_of(unit).weapons:
        if weapon.is_direct or unit.ammo.get(weapon.id, 0) <= 0:
            continue
        for enemy in state.enemies_of(unit.owner):
            if enemy.pos is None:
                continue
            distance = hex_distance(unit.pos, enemy.pos)
            if weapon.covers(distance) and weapon.can_hit(state.spec_of(enemy).damage_class):
                return True
    return False
