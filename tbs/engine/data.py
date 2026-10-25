"""Spec data: units, terrain and scenarios, loaded from ``data/``.

Everything gameplay-facing lives in JSON so that adding or renaming units,
retuning the damage matrix or shipping a new unit pack is a data edit rather
than a code change.  Specs are immutable and never stored inside the game
state: they are attached to a :class:`~tbs.engine.state.GameState` as
``state.specs``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

#: The scenario a bare launch starts: border_skirmish is the introduction map.
#: It is named explicitly rather than taken from ``available_scenarios()[0]``
#: so that adding a map whose filename sorts earlier never changes the default.
DEFAULT_SCENARIO_NAME = "border_skirmish"

#: How a unit moves, which decides which terrain it may enter.
MOVEMENT_TYPES = frozenset({"foot", "wheel", "tread", "air", "sea"})

#: What a unit counts as for damage purposes (see the damage matrix in units.json).
DAMAGE_CLASSES = frozenset({"infantry", "light", "heavy", "air", "naval", "sub"})

#: Supply domains used by terrain and supply vehicles.
SUPPLY_DOMAINS = frozenset({"land", "air", "naval"})

#: Movement type -> supply domain (a unit needs a supply source of its domain).
DOMAIN_OF_MOVEMENT = {
    "foot": "land",
    "wheel": "land",
    "tread": "land",
    "air": "air",
    "sea": "naval",
}


@dataclass(frozen=True)
class WeaponSpec:
    id: str
    name: str
    power: int
    min_range: int
    max_range: int
    ammo: int
    #: Damage multipliers in percent per damage class; 0 means "cannot hit".
    multipliers: dict[str, int]

    @property
    def is_direct(self) -> bool:
        """Direct weapons act at range 1 and can be counterattacked."""
        return self.min_range == 1

    def can_hit(self, damage_class: str) -> bool:
        return self.multipliers.get(damage_class, 0) > 0

    def covers(self, dist: int) -> bool:
        return self.min_range <= dist <= self.max_range

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaponSpec":
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            power=int(data["power"]),
            min_range=int(data["min_range"]),
            max_range=int(data["max_range"]),
            ammo=int(data["ammo"]),
            multipliers={k: int(v) for k, v in data["multipliers"].items()},
        )


@dataclass(frozen=True)
class UnitSpec:
    id: str
    name: str
    glyph: str  # one-character marker used by text views
    damage_class: str
    movement: str
    move: int
    vision: int
    cost: int
    max_hp: int
    fuel: int | None
    weapons: tuple[WeaponSpec, ...]
    capacity: int
    carries: tuple[str, ...]
    can_capture: bool
    build_at: tuple[str, ...]
    supply_domains: tuple[str, ...]
    comment: str = ""

    @property
    def domain(self) -> str:
        return DOMAIN_OF_MOVEMENT[self.movement]

    @property
    def is_transport(self) -> bool:
        return self.capacity > 0

    @property
    def is_fuelled(self) -> bool:
        return self.fuel is not None

    @property
    def can_move(self) -> bool:
        return self.move > 0

    def can_enter(self, move_cost: int | None) -> bool:
        return move_cost is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnitSpec":
        return cls(
            id=data["id"],
            name=data["name"],
            glyph=data["glyph"],
            damage_class=data["damage_class"],
            movement=data["movement"],
            move=int(data["move"]),
            vision=int(data["vision"]),
            cost=int(data["cost"]),
            max_hp=int(data.get("max_hp", 100)),
            fuel=data.get("fuel"),
            weapons=tuple(WeaponSpec.from_dict(w) for w in data.get("weapons", [])),
            capacity=int(data.get("capacity", 0)),
            carries=tuple(data.get("carries", [])),
            can_capture=bool(data.get("can_capture", False)),
            build_at=tuple(data.get("build_at", [])),
            supply_domains=tuple(data.get("supply_domains", [])),
            comment=data.get("comment", ""),
        )


@dataclass(frozen=True)
class TerrainSpec:
    code: str
    name: str
    defense_pct: int
    move_cost: dict[str, int]
    build: str | None
    supply_domains: tuple[str, ...]
    capture_points: int | None
    income: int
    objective: bool

    def cost_for(self, movement: str) -> int | None:
        """Movement points needed to enter, or ``None`` if impassable."""
        return self.move_cost.get(movement)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainSpec":
        move_cost = {k: int(v) for k, v in data["move_cost"].items() if v is not None}
        return cls(
            code=data["code"],
            name=data["name"],
            defense_pct=int(data.get("defense_pct", 0)),
            move_cost=move_cost,
            build=data.get("build"),
            supply_domains=tuple(data.get("supply_domains", [])),
            capture_points=data.get("capture_points"),
            income=int(data.get("income", 0)),
            objective=bool(data.get("objective", False)),
        )


class GameSpecs:
    """Loaded and validated spec data for one running game."""

    def __init__(self, units: dict[str, UnitSpec], terrain: dict[str, TerrainSpec]) -> None:
        self.units = units
        self.terrain = terrain

    def unit(self, unit_type_id: str) -> UnitSpec:
        try:
            return self.units[unit_type_id]
        except KeyError:
            known = ", ".join(sorted(self.units))
            raise KeyError(f"unknown unit type {unit_type_id!r}; known types: {known}") from None

    def terrain_spec(self, code: str) -> TerrainSpec:
        try:
            return self.terrain[code]
        except KeyError:
            known = ", ".join(sorted(self.terrain))
            raise KeyError(f"unknown terrain code {code!r}; known codes: {known}") from None

    def buildable_units(self, build_key: str) -> list[UnitSpec]:
        """Units that can be produced at a build site of ``build_key``."""
        return sorted(
            (spec for spec in self.units.values() if build_key in spec.build_at),
            key=lambda spec: (spec.cost, spec.id),
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if the spec data is internally inconsistent.

        Run by the test suite so that a bad edit to ``data/`` fails loudly.
        """
        glyphs: dict[str, str] = {}
        for spec in self.units.values():
            if spec.movement not in MOVEMENT_TYPES:
                raise ValueError(f"unit {spec.id}: movement {spec.movement!r} is not one of {sorted(MOVEMENT_TYPES)}")
            if spec.damage_class not in DAMAGE_CLASSES:
                raise ValueError(f"unit {spec.id}: damage_class {spec.damage_class!r} is not one of {sorted(DAMAGE_CLASSES)}")
            if len(spec.glyph) != 1:
                raise ValueError(f"unit {spec.id}: glyph must be exactly one character, got {spec.glyph!r}")
            if spec.glyph in glyphs:
                raise ValueError(f"unit {spec.id} and {glyphs[spec.glyph]} share view glyph {spec.glyph!r}")
            glyphs[spec.glyph] = spec.id
            if spec.max_hp <= 0:
                raise ValueError(f"unit {spec.id}: max_hp must be positive")
            if spec.move < 0 or spec.vision < 0:
                raise ValueError(f"unit {spec.id}: move and vision must not be negative")
            if spec.is_fuelled and spec.fuel is not None and spec.fuel < 0:
                raise ValueError(f"unit {spec.id}: fuel must not be negative")
            for domain in spec.supply_domains:
                if domain not in SUPPLY_DOMAINS:
                    raise ValueError(f"unit {spec.id}: supply domain {domain!r} is not one of {sorted(SUPPLY_DOMAINS)}")
            for key in spec.build_at:
                if not any(terrain.build == key for terrain in self.terrain.values()):
                    raise ValueError(f"unit {spec.id}: build_at {key!r} matches no terrain with that build key")
            seen: set[str] = set()
            for weapon in spec.weapons:
                if weapon.id in seen:
                    raise ValueError(f"unit {spec.id}: duplicate weapon id {weapon.id!r}")
                seen.add(weapon.id)
                if weapon.power <= 0:
                    raise ValueError(f"unit {spec.id}/{weapon.id}: power must be positive")
                if weapon.ammo < 0:
                    raise ValueError(f"unit {spec.id}/{weapon.id}: ammo must not be negative")
                if weapon.min_range < 1 or weapon.max_range < weapon.min_range:
                    raise ValueError(f"unit {spec.id}/{weapon.id}: bad range {weapon.min_range}-{weapon.max_range}")
                if not weapon.multipliers:
                    raise ValueError(f"unit {spec.id}/{weapon.id}: no damage multipliers")
                for klass, mult in weapon.multipliers.items():
                    if klass not in DAMAGE_CLASSES:
                        raise ValueError(f"unit {spec.id}/{weapon.id}: unknown damage class {klass!r}")
                    if mult < 0:
                        raise ValueError(f"unit {spec.id}/{weapon.id}: multiplier for {klass} is negative")
        for terrain in self.terrain.values():
            for movement, cost in terrain.move_cost.items():
                if movement not in MOVEMENT_TYPES:
                    raise ValueError(f"terrain {terrain.code}: movement {movement!r} is not a known movement type")
                if cost <= 0:
                    raise ValueError(f"terrain {terrain.code}: move cost for {movement} must be positive")
            for domain in terrain.supply_domains:
                if domain not in SUPPLY_DOMAINS:
                    raise ValueError(f"terrain {terrain.code}: supply domain {domain!r} is not one of {sorted(SUPPLY_DOMAINS)}")
            if terrain.capture_points is not None and terrain.capture_points <= 0:
                raise ValueError(f"terrain {terrain.code}: capture_points must be positive")

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> "GameSpecs":
        root = Path(data_dir).resolve() if data_dir else DATA_ROOT
        units_raw = json.loads((root / "units.json").read_text(encoding="utf-8"))
        terrain_raw = json.loads((root / "terrain.json").read_text(encoding="utf-8"))
        units = {spec.id: spec for spec in (UnitSpec.from_dict(u) for u in units_raw["units"])}
        terrain = {spec.code: spec for spec in (TerrainSpec.from_dict(t) for t in terrain_raw["terrains"])}
        specs = cls(units, terrain)
        specs.validate()
        return specs


_specs_cache: dict[Path, GameSpecs] = {}


def load_specs(data_dir: Path | str | None = None) -> GameSpecs:
    """Load ``data/units.json`` and ``data/terrain.json`` (cached per directory)."""
    root = Path(data_dir).resolve() if data_dir else DATA_ROOT
    specs = _specs_cache.get(root)
    if specs is None:
        specs = GameSpecs.load(root)
        _specs_cache[root] = specs
    return specs


def load_scenario(path: Path | str) -> dict[str, Any]:
    """Load a scenario JSON file from ``data/maps/`` or anywhere else."""
    path = Path(path)
    if not path.exists():
        known = ", ".join(sorted(p.name for p in available_scenarios()))
        raise FileNotFoundError(f"scenario {path} not found; bundled scenarios: {known}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_scenarios(data_dir: Path | str | None = None) -> list[Path]:
    root = (Path(data_dir).resolve() if data_dir else DATA_ROOT) / "maps"
    return sorted(root.glob("*.json")) if root.exists() else []


def default_scenario(data_dir: Path | str | None = None) -> Path:
    """The scenario a launch without ``--scenario`` picks.

    Resolved by name, not by sort order, so a newly added map cannot silently
    become the default.  The name is the scenario file stem (``border_skirmish``).
    """
    scenarios = available_scenarios(data_dir)
    for path in scenarios:
        if path.stem == DEFAULT_SCENARIO_NAME:
            return path
    known = ", ".join(path.name for path in scenarios) or "none"
    raise FileNotFoundError(
        f"default scenario {DEFAULT_SCENARIO_NAME!r} is missing from data/maps; found: {known}"
    )
