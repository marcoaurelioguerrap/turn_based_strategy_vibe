"""In-memory game state.

Everything held by :class:`GameState` is plain data: exactly the structure that
gets saved to JSON, hashed for determinism tests and handed to UIs or agent
drivers.  Spec data (units, terrain) is *not* part of the state: it is loaded
from ``data/`` and attached as ``state.specs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .data import GameSpecs
from .hexgrid import Hex, axial_to_offset, hex_cells, offset_to_axial

SCHEMA_VERSION = 1

#: The keys a scenario unit placement may use.  Anything else is an authoring
#: mistake, so it is reported instead of quietly ignored.
UNIT_PLACEMENT_KEYS = frozenset(
    {"id", "type", "owner", "offset", "pos", "label", "hp", "fuel", "ammo"}
)

#: Damage taken each turn by an air or naval unit that ends its turn out of fuel.
OUT_OF_FUEL_DAMAGE = 15


@dataclass
class Unit:
    """One unit on the board.

    ``pos`` is ``None`` while the unit is loaded inside a transport.  The
    per-turn flags (``moved``, ``acted``, ``deployed_this_turn``) are reset at
    the start of the owner's turn.
    """

    id: str
    owner: str
    type_id: str
    pos: Hex | None
    hp: int
    ammo: dict[str, int] = field(default_factory=dict)
    fuel: int | None = None
    cargo: list[str] = field(default_factory=list)
    loaded: bool = False
    carried_by: str | None = None
    supplied: bool = True
    label: str = ""
    moved: bool = False
    acted: bool = False
    deployed_this_turn: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "type": self.type_id,
            "pos": list(self.pos) if self.pos is not None else None,
            "hp": self.hp,
            "ammo": dict(self.ammo),
            "fuel": self.fuel,
            "cargo": list(self.cargo),
            "loaded": self.loaded,
            "carried_by": self.carried_by,
            "supplied": self.supplied,
            "label": self.label,
            "moved": self.moved,
            "acted": self.acted,
            "deployed_this_turn": self.deployed_this_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Unit":
        pos = data["pos"]
        return cls(
            id=data["id"],
            owner=data["owner"],
            type_id=data["type"],
            pos=tuple(pos) if pos is not None else None,
            hp=int(data["hp"]),
            ammo=dict(data.get("ammo", {})),
            fuel=data.get("fuel"),
            cargo=list(data.get("cargo", [])),
            loaded=bool(data.get("loaded", False)),
            carried_by=data.get("carried_by"),
            supplied=bool(data.get("supplied", True)),
            label=data.get("label", ""),
            moved=bool(data.get("moved", False)),
            acted=bool(data.get("acted", False)),
            deployed_this_turn=bool(data.get("deployed_this_turn", False)),
        )


@dataclass
class Tile:
    """One hex of the map, including its ownership and capture progress."""

    pos: Hex
    terrain: str
    owner: str | None = None
    capture_points: int | None = None
    capture_progress: int = 0
    capture_unit: str | None = None
    objective: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": list(self.pos),
            "terrain": self.terrain,
            "owner": self.owner,
            "capture_points": self.capture_points,
            "capture_progress": self.capture_progress,
            "capture_unit": self.capture_unit,
            "objective": self.objective,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tile":
        return cls(
            pos=tuple(data["pos"]),
            terrain=data["terrain"],
            owner=data.get("owner"),
            capture_points=data.get("capture_points"),
            capture_progress=int(data.get("capture_progress", 0)),
            capture_unit=data.get("capture_unit"),
            objective=bool(data.get("objective", False)),
        )


@dataclass
class GameMap:
    width: int
    height: int
    tiles: dict[Hex, Tile]

    def tile(self, pos: Hex) -> Tile:
        pos = tuple(pos)
        try:
            return self.tiles[pos]
        except KeyError:
            raise KeyError(f"({pos[0]},{pos[1]}) is not part of the map") from None

    def in_bounds(self, pos: Hex) -> bool:
        return tuple(pos) in self.tiles

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "tiles": [tile.to_dict() for tile in sorted(self.tiles.values(), key=lambda t: (t.pos[1], t.pos[0]))],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameMap":
        tiles = {tuple(tile["pos"]): Tile.from_dict(tile) for tile in data["tiles"]}
        return cls(width=int(data["width"]), height=int(data["height"]), tiles=tiles)


@dataclass
class Player:
    id: str
    name: str
    funds: int
    eliminated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "funds": self.funds, "eliminated": self.eliminated}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Player":
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            funds=int(data["funds"]),
            eliminated=bool(data.get("eliminated", False)),
        )


@dataclass
class GameState:
    """The whole game: map, players, units and turn bookkeeping."""

    seed: int
    map: GameMap
    players: dict[str, Player]
    units: dict[str, Unit] = field(default_factory=dict)
    turn_number: int = 1
    current_player: str = ""
    status: str = "ongoing"
    winner: str | None = None
    initial_income: int = 1000
    config: dict[str, Any] = field(default_factory=dict)
    action_seq: int = 0
    unit_counter: dict[str, int] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    scenario_name: str = ""
    specs: GameSpecs | None = field(default=None, repr=False, compare=False)

    # --- accessors used by the rules and the views -------------------------

    def spec_of(self, unit: "Unit | str") -> Any:
        """The UnitSpec of a unit (or unit type id)."""
        assert self.specs is not None, "state.specs was not set (use new_game or load_game)"
        return self.specs.unit(unit.type_id if isinstance(unit, Unit) else unit)

    def terrain_of(self, tile: "Tile | str") -> Any:
        assert self.specs is not None, "state.specs was not set (use new_game or load_game)"
        return self.specs.terrain_spec(tile.terrain if isinstance(tile, Tile) else tile)

    def tile(self, pos: Hex) -> Tile:
        return self.map.tile(tuple(pos))

    def unit(self, unit_id: str) -> Unit:
        try:
            return self.units[unit_id]
        except KeyError:
            known = ", ".join(sorted(self.units))
            raise KeyError(f"unknown unit id {unit_id!r}; units in play: {known}") from None

    def units_at(self, pos: Hex) -> list[Unit]:
        """Every unit standing on a hex, in stable id order.

        More than one unit can be present when an air unit shares a hex with a
        surface unit (see :func:`can_share_hex`).  Loaded units have ``pos`` set
        to ``None`` and are therefore never returned here.
        """
        pos = tuple(pos)
        return sorted(
            (unit for unit in self.units.values() if unit.pos == pos),
            key=lambda unit: unit.id,
        )

    def unit_at(self, pos: Hex) -> Unit | None:
        """The unit on top of a hex: an air unit if one is stacked, else the
        surface unit.  Use :meth:`units_at` when every occupant matters."""
        units = self.units_at(pos)
        for unit in units:
            if self.spec_of(unit).domain == "air":
                return unit
        return units[0] if units else None

    def units_of(self, player_id: str, *, include_loaded: bool = True) -> list[Unit]:
        """A player's units, ordered by id so generated lists are reproducible."""
        units = [u for u in self.units.values() if u.owner == player_id]
        if not include_loaded:
            units = [u for u in units if not u.loaded]
        return sorted(units, key=lambda u: u.id)

    def enemies_of(self, player_id: str) -> list[Unit]:
        """Every unit that belongs to another player and can be attacked."""
        enemies = [
            unit
            for unit in self.units.values()
            if unit.owner != player_id and unit.pos is not None
        ]
        return sorted(enemies, key=lambda u: u.id)

    def player_ids(self) -> list[str]:
        return list(self.players)

    def alive_players(self) -> list[str]:
        return [pid for pid, player in self.players.items() if not player.eliminated]

    def owned_tiles(self, player_id: str) -> list[Tile]:
        return sorted(
            (tile for tile in self.map.tiles.values() if tile.owner == player_id),
            key=lambda t: (t.pos[1], t.pos[0]),
        )

    def build_sites(self, player_id: str) -> list[Tile]:
        assert self.specs is not None
        return [tile for tile in self.owned_tiles(player_id) if self.terrain_of(tile).build]

    # --- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "seed": self.seed,
            "scenario": self.scenario_name,
            "map": self.map.to_dict(),
            "players": [player.to_dict() for player in self.players.values()],
            "units": [unit.to_dict() for unit in sorted(self.units.values(), key=lambda u: u.id)],
            "turn_number": self.turn_number,
            "current_player": self.current_player,
            "status": self.status,
            "winner": self.winner,
            "initial_income": self.initial_income,
            "config": dict(self.config),
            "action_seq": self.action_seq,
            "unit_counter": dict(self.unit_counter),
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], specs: GameSpecs) -> "GameState":
        players = {player["id"]: Player.from_dict(player) for player in data["players"]}
        state = cls(
            seed=int(data["seed"]),
            map=GameMap.from_dict(data["map"]),
            players=players,
            units={unit["id"]: Unit.from_dict(unit) for unit in data["units"]},
            turn_number=int(data["turn_number"]),
            current_player=data["current_player"],
            status=data.get("status", "ongoing"),
            winner=data.get("winner"),
            initial_income=int(data.get("initial_income", 1000)),
            config=dict(data.get("config", {})),
            action_seq=int(data.get("action_seq", 0)),
            unit_counter=dict(data.get("unit_counter", {})),
            log=list(data.get("log", [])),
            scenario_name=data.get("scenario", ""),
            specs=specs,
        )
        return state


def can_share_hex(state: GameState, first: Unit | str, second: Unit | str) -> bool:
    """Whether two units may stand on the same hex.

    The board has two layers: air units fly above the surface.  An air unit may
    therefore share a hex with exactly one land or naval unit, but two air units
    and two surface units still block each other.
    """
    first_domain = state.spec_of(first).domain
    second_domain = state.spec_of(second).domain
    return (first_domain == "air") != (second_domain == "air")


def new_game(scenario: dict[str, Any], specs: GameSpecs) -> GameState:
    """Create a fresh game from a scenario dictionary.

    Scenario validation happens here: row lengths, terrain codes, unit types,
    legal spawn positions and unique ids all produce clear errors.
    """
    schema = int(scenario.get("schema", SCHEMA_VERSION))
    if schema != SCHEMA_VERSION:
        raise ValueError(f"scenario schema {schema} is not supported (expected {SCHEMA_VERSION})")

    rows = list(scenario.get("rows") or [])
    if not rows:
        raise ValueError("scenario has no 'rows' (the map as text rows)")
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"map row {index} is {len(row)} chars wide but row 0 is {width} wide")

    tiles: dict[Hex, Tile] = {}
    for pos, code in hex_cells(rows).items():
        terrain = specs.terrain_spec(code)
        tiles[pos] = Tile(
            pos=pos,
            terrain=code,
            capture_points=terrain.capture_points,
            objective=terrain.objective,
        )

    players: dict[str, Player] = {}
    for raw in scenario.get("players") or []:
        pid = raw["id"]
        if pid in players:
            raise ValueError(f"duplicate player id {pid!r}")
        players[pid] = Player(id=pid, name=raw.get("name", pid), funds=int(raw.get("funds", 8000)))
    if not players:
        raise ValueError("scenario needs at least one player")

    for region in scenario.get("regions", []):
        owner = region.get("owner")
        if owner not in players:
            raise ValueError(f"region refers to unknown player {owner!r}")
        if "offset_cols" in region:
            low, high = region["offset_cols"]
            for pos, tile in tiles.items():
                col, _ = axial_to_offset(pos)
                if low <= col <= high:
                    tile.owner = owner
        elif "cells" in region:
            for cell in region["cells"]:
                pos = tuple(cell)
                if pos not in tiles:
                    raise ValueError(f"region cell {list(pos)} is outside the map")
                tiles[pos].owner = owner
        else:
            raise ValueError(f"region for {owner!r} needs either 'offset_cols' or 'cells'")

    for override in scenario.get("tile_overrides", []):
        pos = _placement_position(override)
        tile = tiles.get(pos)
        if tile is None:
            raise ValueError(f"tile override {list(pos)} is outside the map")
        if "owner" in override:
            owner = override["owner"]
            if owner not in players:
                raise ValueError(f"tile override at {list(pos)} refers to unknown player {owner!r}")
            tile.owner = owner
        if "capture_points" in override:
            tile.capture_points = int(override["capture_points"])
        if "objective" in override:
            tile.objective = bool(override["objective"])

    state = GameState(
        seed=int(scenario.get("seed", 1)),
        map=GameMap(width=width, height=len(rows), tiles=tiles),
        players=players,
        initial_income=int(scenario.get("initial_income", 1000)),
        config={
            "fog_of_war": bool(scenario.get("fog_of_war", False)),
            "max_turns": int(scenario.get("max_turns", 200)),
            # "damage_luck": false makes combat fully predictable (used by tests
            # and by deterministic agent runs).
            "damage_luck": bool(scenario.get("damage_luck", True)),
        },
        scenario_name=scenario.get("name", ""),
        specs=specs,
    )

    for raw in scenario.get("units", []):
        unknown = sorted(set(raw) - UNIT_PLACEMENT_KEYS)
        if unknown:
            raise ValueError(
                f"unit placement {raw.get('id') or raw.get('type')!r} uses unknown keys "
                f"{', '.join(unknown)}; a placement may only set {', '.join(sorted(UNIT_PLACEMENT_KEYS))}"
            )
        create_unit(
            state,
            type_id=raw["type"],
            owner=raw["owner"],
            pos=_placement_position(raw),
            unit_id=raw.get("id"),
            label=raw.get("label", ""),
            hp=raw.get("hp"),
            fuel=raw.get("fuel"),
            ammo=raw.get("ammo"),
        )

    first = scenario.get("first_player") or next(iter(players))
    if first not in players:
        raise ValueError(f"first_player {first!r} is not one of the scenario players")
    state.current_player = first
    return state


def _placement_position(raw: dict[str, Any]) -> Hex:
    if "offset" in raw:
        col, row = raw["offset"]
        return offset_to_axial(int(col), int(row))
    if "pos" in raw:
        return tuple(raw["pos"])
    raise ValueError(f"unit placement {raw.get('id') or raw.get('type')!r} needs 'offset' [col,row] or 'pos' [q,r]")


def create_unit(
    state: GameState,
    type_id: str,
    owner: str,
    pos: Hex | None,
    unit_id: str | None = None,
    label: str = "",
    hp: int | None = None,
    fuel: int | None = None,
    ammo: dict[str, int] | None = None,
    deployed_this_turn: bool = False,
) -> Unit:
    """Add a unit to the game, validating its type, owner, spawn hex and the
    partial equipment (HP, fuel, ammo) a scenario may start it with."""
    spec = state.spec_of(type_id)
    if owner not in state.players:
        raise ValueError(f"cannot create a unit for unknown player {owner!r}")
    if unit_id is None:
        unit_id = _next_unit_id(state, owner)
    if unit_id in state.units:
        raise ValueError(f"duplicate unit id {unit_id!r}")
    if pos is not None:
        pos = tuple(pos)
        tile = state.map.tiles.get(pos)
        if tile is None:
            raise ValueError(f"{spec.name} cannot be placed at {list(pos)}: it is not part of the map")
        cost = state.terrain_of(tile).cost_for(spec.movement)
        if cost is None:
            raise ValueError(
                f"{spec.name} cannot be placed on {state.terrain_of(tile).name} at {list(pos)} "
                f"({spec.movement} units cannot enter that terrain)"
            )
        occupant = next(
            (
                other
                for other in state.units_at(pos)
                if not can_share_hex(state, spec.id, other)
            ),
            None,
        )
        if occupant is not None:
            raise ValueError(f"cannot place {spec.name} at {list(pos)}: {occupant.id} is already there")

    if hp is not None:
        hp = int(hp)
        if not 1 <= hp <= spec.max_hp:
            raise ValueError(
                f"unit placement {unit_id!r}: hp {hp} is not between 1 and {spec.max_hp} "
                f"({spec.name} maximum)"
            )
    if fuel is not None:
        if spec.fuel is None:
            raise ValueError(
                f"unit placement {unit_id!r}: {spec.name} carries no fuel tank; drop the 'fuel' override"
            )
        fuel = int(fuel)
        if not 0 <= fuel <= spec.fuel:
            raise ValueError(f"unit placement {unit_id!r}: fuel {fuel} is not between 0 and {spec.fuel}")
    weapon_ammo = {weapon.id: weapon.ammo for weapon in spec.weapons}
    if ammo is not None:
        if not isinstance(ammo, dict):
            raise ValueError(
                f"unit placement {unit_id!r}: 'ammo' must be a mapping like {{\"mg\": 2}}, not {ammo!r}"
            )
        for weapon_id, amount in ammo.items():
            if weapon_id not in weapon_ammo:
                raise ValueError(
                    f"unit placement {unit_id!r}: {spec.name} has no weapon {weapon_id!r}; "
                    f"it has: {', '.join(weapon_ammo) or 'no weapons'}"
                )
            amount = int(amount)
            if not 0 <= amount <= weapon_ammo[weapon_id]:
                raise ValueError(
                    f"unit placement {unit_id!r}: {weapon_id} ammo {amount} is not between 0 "
                    f"and {weapon_ammo[weapon_id]}"
                )
            weapon_ammo[weapon_id] = amount

    unit = Unit(
        id=unit_id,
        owner=owner,
        type_id=type_id,
        pos=pos,
        hp=hp if hp is not None else spec.max_hp,
        ammo=weapon_ammo,
        fuel=fuel if fuel is not None else spec.fuel,
        label=label or f"{owner} {spec.name}",
        deployed_this_turn=deployed_this_turn,
    )
    state.units[unit.id] = unit
    return unit


def _next_unit_id(state: GameState, owner: str) -> str:
    next_number = state.unit_counter.get(owner, 0) + 1
    while f"{owner}{next_number}" in state.units:
        next_number += 1
    state.unit_counter[owner] = next_number
    return f"{owner}{next_number}"


def remove_unit(state: GameState, unit_id: str) -> Unit:
    """Delete a unit from the game.  Cargo carried by it is destroyed as well
    (the same rule Advance Wars uses for a sunk transport)."""
    unit = state.unit(unit_id)
    for cargo_id in list(unit.cargo):
        remove_unit(state, cargo_id)
    if unit.carried_by is not None:
        carrier = state.units.get(unit.carried_by)
        if carrier is not None:
            carrier.cargo = [cid for cid in carrier.cargo if cid != unit.id]
    state.units.pop(unit.id)
    for tile in state.map.tiles.values():
        if tile.capture_unit == unit.id:
            tile.capture_unit = None
            tile.capture_progress = 0
    return unit
