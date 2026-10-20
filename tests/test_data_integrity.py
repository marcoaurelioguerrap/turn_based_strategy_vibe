"""``data/`` is the design.  These tests guard the promises the engine's own validator cannot make.

``GameSpecs.validate()`` runs on every load and already checks everything a
spec must promise about itself: glyph uniqueness, weapon ranges, multiplier
keys, move-cost sanity.  What it structurally cannot see are the cross-file
traps:

* a unit buildable at a site its movement type could never drive off of -
  a factory that swallows everything produced in it;
* a scenario map placing a unit on ground it cannot cross, which is exactly
  the bug that once put a tread ``aa_missile`` on a mountain;
* a transport whose carries list matches no unit that exists;
* capturable objectives that no buildable capturer can ever walk onto;
* a movement graph where roads are not an upgrade over open ground.

And it checks the validator itself: for every rule ``validate()`` enforces, a
tampered copy of the shipped data must still raise the error a modder would
need to read.
"""

from __future__ import annotations

import json
import re

import pytest

from tbs.engine.data import (
    DATA_ROOT,
    MOVEMENT_TYPES,
    GameSpecs,
    TerrainSpec,
    UnitSpec,
    available_scenarios,
    load_scenario,
    load_specs,
)

SPECS = load_specs()

#: The v1 roster.  A floor, not a ceiling: variants may be added after the base
#: game works; these units are the game and may not vanish from it.
V1_ROSTER = {
    "infantry",
    "apc",
    "truck",
    "tank",
    "artillery",
    "aa_missile",
    "helicopter",
    "attack_plane",
    "bomber",
    "destroyer",
    "cruiser",
    "submarine",
    "transport_ship",
    "hq",
}

#: The map vocabulary every authored map is written in.
MAP_VOCABULARY = {".", "r", "F", "#", "M", "C", "f", "p", "A", "H", "~"}

LAND_MOVEMENTS = ("foot", "wheel", "tread")


# ---------------------------------------------------------------------------
# the roster and the damage matrix it promises
# ---------------------------------------------------------------------------


def test_the_v1_roster_is_all_present():
    missing = V1_ROSTER - set(SPECS.units)
    assert not missing, f"the shipped roster is missing core units: {sorted(missing)}"
    for spec in SPECS.units.values():
        assert spec.name and spec.glyph, f"unit {spec.id} must carry a display name and glyph"


# ---------------------------------------------------------------------------
# cross-file invariants the engine's own validate() cannot see
# ---------------------------------------------------------------------------


def test_no_unit_is_stranded_by_the_build_site_it_is_built_on():
    for spec in SPECS.units.values():
        for build_key in spec.build_at:
            sites = [terrain for terrain in SPECS.terrain.values() if terrain.build == build_key]
            assert sites, f"{spec.id} names build site {build_key!r}, but no terrain provides it"
            for site in sites:
                assert spec.can_enter(site.cost_for(spec.movement)), (
                    f"{spec.id} ({spec.movement}) can be built on {site.code!r}, but could "
                    "never drive off it: the build site must be passable for the unit's own movement"
                )


def test_every_scenario_map_places_units_on_ground_they_can_stand_on():
    scenario_paths = available_scenarios()
    assert scenario_paths, "data/maps ships no scenarios at all"
    for path in scenario_paths:
        data = load_scenario(path)
        rows = data["rows"]
        assert len({len(row) for row in rows}) == 1, f"{path.name}: every row must be the same length"
        unknown = {code for row in rows for code in row} - set(SPECS.terrain)
        assert not unknown, f"{path.name}: rows use terrain codes the terrain file does not define: {sorted(unknown)}"

        players = {player["id"] for player in data["players"]}
        for region in data.get("regions", []):
            assert region["owner"] in players, f"{path.name}: a region is owned by {region['owner']!r}, who is not a player"
            if "offset_cols" in region:
                low, high = region["offset_cols"]
                assert 0 <= low <= high < len(rows[0]), f"{path.name}: region columns {low}-{high} lie outside the map"

        seen_ids: set[str] = set()
        for placement in data["units"]:
            unit_id = placement.get("id", "?")
            assert placement["owner"] in players, f"{path.name}: unit {unit_id} belongs to {placement['owner']!r}, who is not a player"
            if "id" in placement:
                assert placement["id"] not in seen_ids, f"{path.name}: unit id {placement['id']!r} is used twice"
                seen_ids.add(placement["id"])
            spec = SPECS.units.get(placement["type"])
            assert spec is not None, f"{path.name}: unit {unit_id} uses unit type {placement['type']!r}, which is not in the roster"
            col, row = placement["offset"]
            assert 0 <= row < len(rows) and 0 <= col < len(rows[row]), f"{path.name}: unit {unit_id} is placed outside the map"
            terrain = SPECS.terrain[rows[row][col]]
            assert spec.can_enter(terrain.cost_for(spec.movement)), (
                f"{path.name}: unit {unit_id} ({spec.name}, {spec.movement}) is placed on {terrain.code!r}, "
                "ground it cannot cross - this is the bug class that once put a tread AA Guard on a mountain"
            )


def test_transports_carry_movement_types_that_exist():
    for spec in SPECS.units.values():
        if not spec.is_transport:
            continue
        assert spec.carries, f"{spec.id} has capacity {spec.capacity} but carries nothing: it could never unload anything"
        for carried in spec.carries:
            assert carried in MOVEMENT_TYPES, f"{spec.id} carries {carried!r}, which is not a movement type"
        assert any(other.movement in spec.carries for other in SPECS.units.values()), (
            f"{spec.id} carries {list(spec.carries)}, a class no unit in the roster belongs to"
        )


def test_capturable_ground_can_be_taken_by_someone_buildable():
    capturable = [terrain for terrain in SPECS.terrain.values() if terrain.capture_points]
    assert capturable, "the map vocabulary defines no capturable site at all"
    assert any(terrain.objective for terrain in capturable), "no site is an objective: the HQ victory path has nothing to stand on"
    capturers = [spec for spec in SPECS.units.values() if spec.can_capture and spec.build_at]
    assert capturers, "there is capturable ground, but no capturable unit can be built to take it"
    for capturer in capturers:
        for terrain in capturable:
            assert capturer.can_enter(terrain.cost_for(capturer.movement)), (
                f"{capturer.id} is the capture unit, but it cannot even enter {terrain.code!r}"
            )


def test_the_movement_graph_fights_land_wars():
    for code in MAP_VOCABULARY:
        assert code in SPECS.terrain, f"the map vocabulary is missing {code!r}"
    plains, road, mountain, sea = (SPECS.terrain[code] for code in (".", "r", "M", "~"))

    for movement in LAND_MOVEMENTS:
        assert plains.cost_for(movement) is not None, f"open plains must be crossable by {movement}"
        assert road.cost_for(movement) is not None and road.cost_for(movement) <= plains.cost_for(movement), (
            f"roads must be an upgrade over the open ground they replace for {movement}"
        )
        assert sea.cost_for(movement) is None, f"sea is not an opening for {movement}"

    assert mountain.cost_for("foot") is not None
    assert mountain.cost_for("wheel") is None and mountain.cost_for("tread") is None, (
        "mountains are foot-only: this is the exact rule the scenario tests rely on, and the reason "
        "AA Guards belong on cities"
    )

    assert sea.cost_for("sea") is not None
    for terrain in SPECS.terrain.values():
        assert terrain.cost_for("air") is not None, f"terrain {terrain.code!r} blocks aircraft: aircraft fly over everything here"
        if terrain.cost_for("sea") is not None:
            assert terrain.code == "~" or terrain.build == "port", (
                f"terrain {terrain.code!r} is sea-passable, but naval units could never be built on it"
            )


# ---------------------------------------------------------------------------
# the validator itself: every rule it enforces must still fire on bad data
# ---------------------------------------------------------------------------


def _raw_data() -> tuple[list[dict], list[dict]]:
    units = json.loads((DATA_ROOT / "units.json").read_text(encoding="utf-8"))["units"]
    terrains = json.loads((DATA_ROOT / "terrain.json").read_text(encoding="utf-8"))["terrains"]
    return units, terrains


def _specs(units_raw: list[dict], terrains_raw: list[dict]) -> GameSpecs:
    units = {spec.id: spec for spec in map(UnitSpec.from_dict, units_raw)}
    terrains = {spec.code: spec for spec in map(TerrainSpec.from_dict, terrains_raw)}
    specs = GameSpecs(units, terrains)
    specs.validate()
    return specs


def _tamper_unit(unit_id: str, field: str, value: object) -> GameSpecs:
    units, terrains = _raw_data()
    next(unit for unit in units if unit["id"] == unit_id)[field] = value
    return _specs(units, terrains)


def _tamper_weapon(unit_id: str, weapon_id: str, field: str, value: object) -> GameSpecs:
    units, terrains = _raw_data()
    unit = next(u for u in units if u["id"] == unit_id)
    next(weapon for weapon in unit["weapons"] if weapon["id"] == weapon_id)[field] = value
    return _specs(units, terrains)


def _duplicate_weapon(unit_id: str, weapon_id: str) -> GameSpecs:
    units, terrains = _raw_data()
    unit = next(u for u in units if u["id"] == unit_id)
    weapon = next(w for w in unit["weapons"] if w["id"] == weapon_id)
    units_by_id = {u["id"]: u for u in units}
    units_by_id[unit_id]["weapons"].append(json.loads(json.dumps(weapon)))
    return _specs(units, terrains)


def _tamper_terrain(code: str, field: str, value: object) -> GameSpecs:
    units, terrains = _raw_data()
    next(terrain for terrain in terrains if terrain["code"] == code)[field] = value
    return _specs(units, terrains)


def _worse_move_cost() -> GameSpecs:
    units, terrains = _raw_data()
    next(terrain for terrain in terrains if terrain["code"] == ".")["move_cost"] = {"foot": 0}
    return _specs(units, terrains)


def _alien_move_cost() -> GameSpecs:
    units, terrains = _raw_data()
    terrain = next(t for t in terrains if t["code"] == ".")
    terrain["move_cost"]["hovercraft"] = 1
    return _specs(units, terrains)


def _bad_range() -> GameSpecs:
    units, terrains = _raw_data()
    unit = next(u for u in units if u["id"] == "infantry")
    rifle = next(w for w in unit["weapons"] if w["id"] == "rifle")
    rifle["min_range"], rifle["max_range"] = 3, 2
    return _specs(units, terrains)


CASES = [
    ("unknown movement type", lambda: _tamper_unit("infantry", "movement", "hop"), "is not one of"),
    ("unknown damage class", lambda: _tamper_unit("infantry", "damage_class", "armoured"), "damage_class"),
    ("glyph not one character", lambda: _tamper_unit("infantry", "glyph", "ii"), "exactly one character"),
    ("glyph already taken", lambda: _tamper_unit("tank", "glyph", "i"), "share view glyph"),
    ("max_hp not positive", lambda: _tamper_unit("infantry", "max_hp", 0), "max_hp must be positive"),
    ("negative move", lambda: _tamper_unit("tank", "move", -1), "must not be negative"),
    ("negative fuel", lambda: _tamper_unit("tank", "fuel", -5), "fuel must not be negative"),
    ("unit supply domain unknown", lambda: _tamper_unit("truck", "supply_domains", ["underground"]), "supply domain"),
    ("build site matches no terrain", lambda: _tamper_unit("tank", "build_at", ["mothership_pad"]), "matches no terrain"),
    ("weapon power not positive", lambda: _tamper_weapon("infantry", "rifle", "power", 0), "power must be positive"),
    ("weapon ammo negative", lambda: _tamper_weapon("infantry", "rifle", "ammo", -3), "ammo must not be negative"),
    ("weapon range inverted", _bad_range, "bad range"),
    ("weapon multiplier negative", lambda: _tamper_weapon("infantry", "rifle", "multipliers", {"infantry": 100, "heavy": -10}), "is negative"),
    ("weapon multiplier class unknown", lambda: _tamper_weapon("infantry", "rifle", "multipliers", {"ghost": 100}), "unknown damage class"),
    ("weapon has no multipliers", lambda: _tamper_weapon("infantry", "rifle", "multipliers", {}), "no damage multipliers"),
    ("duplicate weapon id", lambda: _duplicate_weapon("infantry", "rifle"), "duplicate weapon id"),
    ("terrain move cost not positive", _worse_move_cost, "must be positive"),
    ("terrain movement type unknown", _alien_move_cost, "not a known movement type"),
    ("terrain supply domain unknown", lambda: _tamper_terrain("C", "supply_domains", ["orbit"]), "supply domain"),
    ("capture points not positive", lambda: _tamper_terrain("C", "capture_points", 0), "capture_points must be positive"),
]


@pytest.mark.parametrize("name, build, message", CASES, ids=[case[0] for case in CASES])
def test_the_validator_catches_tampered_data(name: str, build, message: str):
    with pytest.raises(ValueError, match=re.escape(message)):
        build()
