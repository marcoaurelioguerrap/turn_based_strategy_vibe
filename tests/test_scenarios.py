"""Scenario authoring is part of the game data, so the engine validates it loudly.

This file pins two things.

First, the bundled ``border_skirmish`` scenario really starts a game that is
fair to play: both sides own the same kinds of bases and build sites, earn the
same income, command the same army, and begin with every unit connected to
supply.  Fairness is verified as *material* symmetry - the same functional
situation on both sides - rather than a pixel-perfect map mirror: a hex-true
mirror is an axial reflection, which shifts by an extra column on the odd
authoring rows, so "reverse the row strings" would be the wrong test.  The
bundled map is built to be materially fair, and this file is what keeps it
that way.

Second, everything a scenario author can get wrong - a wrong schema, ragged
rows, a terrain character nobody defined, regions claiming for unknown players,
units spawned off the map or onto ground they cannot stand on, duplicate ids -
is refused with a readable error before anyone takes a turn.  (The equipment a
placement may start with - hp, fuel, ammo - is validated and asserted in
``test_logistics``.)
"""

from __future__ import annotations

from collections import Counter

import pytest

from tbs.engine.data import (
    DATA_ROOT,
    available_scenarios,
    default_scenario,
    load_scenario,
)
from tbs.engine.rules import _near_supply_source, is_supplied, turn_income
from tbs.engine.state import new_game
from .helpers import PLAIN_ROWS, SPECS, bundled_path, bundled_scenario, scenario, unit

SEA_ROWS = ["~~~~......", "..........", "..........", ".........."]

# What every side of the bundled scenario is expected to own and to field.
# A fair start must be equal on both sides *and* nonzero: comparing empty
# inventories against each other would prove nothing.
START_BELONGINGS = Counter({"City": 3, "Factory": 1, "Port": 1, "Airport": 1, "Headquarters": 1})
START_INCOME = 800 + 3 * 300 + 500 + 300 + 200 + 400  # initial income plus every owned site
START_ARMY = Counter(
    {"hq": 1, "infantry": 2, "tank": 1, "artillery": 1, "truck": 1, "aa_missile": 1, "attack_plane": 1, "destroyer": 1}
)


def functional_bases(state, player_id: str) -> Counter:
    """Terrain names of the sites this player owns that actually do something.

    Bare mountain and open sea are deliberately ignored: they supply nobody,
    build nothing, capture nothing and pay nothing, so owning one is not an
    advantage to balance.  Everything a material-fairness check must balance -
    supply sources, build sites, income, captureable and objective tiles - is.
    """
    counts: Counter = Counter()
    for tile in state.owned_tiles(player_id):
        terrain = state.terrain_of(tile)
        if terrain.income or terrain.supply_domains or terrain.build or terrain.capture_points or terrain.objective:
            counts[terrain.name] += 1
    return counts


# ---------------------------------------------------------------------------
# the shipped scenario
# ---------------------------------------------------------------------------


def test_the_bundled_scenario_ships_with_the_game_and_loads_by_name():
    assert "border_skirmish.json" in [path.name for path in available_scenarios()]
    assert default_scenario().name == "border_skirmish.json"
    assert load_scenario(bundled_path()) == bundled_scenario()

    with pytest.raises(FileNotFoundError) as info:
        load_scenario(DATA_ROOT / "maps" / "no_such_map.json")
    message = str(info.value)
    assert "not found; bundled scenarios:" in message
    assert "border_skirmish.json" in message


def test_border_skirmish_starts_a_legal_game():
    state = new_game(bundled_scenario(), SPECS)
    assert state.status == "ongoing"
    assert state.map.width == 14 and state.map.height == 8
    assert len(state.units) == 18
    assert len(state.units_of("A")) == 9 and len(state.units_of("B")) == 9
    assert state.players["A"].funds == state.players["B"].funds == 8000

    # Every side owns exactly one objective, and its HQ unit stands on it.
    for player_id in ("A", "B"):
        objectives = [tile for tile in state.owned_tiles(player_id) if tile.objective]
        assert len(objectives) == 1
        guard = next(u for u in state.units_of(player_id) if u.type_id == "hq")
        assert guard.pos == objectives[0].pos, "each side's flag stands on its own HQ hex"


def test_border_skirmish_gives_both_sides_the_same_borders():
    state = new_game(bundled_scenario(), SPECS)
    assert functional_bases(state, "A") == functional_bases(
        state, "B"
    ), "the mirrored side owns the same kinds of sites in the same numbers"
    assert functional_bases(state, "A") == START_BELONGINGS


def test_both_sides_of_border_skirmish_begin_with_the_same_army():
    state = new_game(bundled_scenario(), SPECS)
    armies = {pid: Counter(u.type_id for u in state.units_of(pid)) for pid in ("A", "B")}
    assert armies["A"] == armies["B"] == START_ARMY


def test_both_sides_of_border_skirmish_earn_the_same_from_the_first_turn():
    state = new_game(bundled_scenario(), SPECS)
    assert turn_income(state, "A") == START_INCOME
    assert turn_income(state, "B") == START_INCOME


def test_every_unit_on_the_border_begins_connected_to_supply():
    state = new_game(bundled_scenario(), SPECS)
    # The strict fuel-level check, not just the lenient "may attack" one:
    # the same supply situation is part of what makes this start fair.
    stranded = [u.id for u in state.units.values() if not _near_supply_source(state, u)]
    assert stranded == []
    assert all(is_supplied(state, u) for u in state.units.values())


# ---------------------------------------------------------------------------
# the rest of the map pack
# ---------------------------------------------------------------------------


def fair_map_paths() -> list:
    return [path for path in available_scenarios() if load_scenario(path).get("fair")]


FAIR_MAPS = fair_map_paths()


@pytest.mark.parametrize("path", FAIR_MAPS, ids=lambda path: path.stem)
def test_every_fair_map_starts_a_balanced_game(path):
    """A map that advertises fairness really is materially symmetric.

    This is the generic version of the border_skirmish assertions: the same
    functional sites, the same army, the same income and every unit in supply.
    The terrain itself is hex-true mirrored by the authoring script, so each
    side sees the same ground under its own half of the board.
    """
    data = load_scenario(path)
    state = new_game(data, SPECS)

    assert state.status == "ongoing"
    assert len(state.players) == 2, f"{path.stem}: fair maps are two-sided"
    first, second = state.player_ids()
    assert state.players[first].funds == state.players[second].funds
    assert functional_bases(state, first) == functional_bases(state, second)
    assert functional_bases(state, first), f"{path.stem}: both sides own functional sites"

    armies = {pid: Counter(u.type_id for u in state.units_of(pid)) for pid in (first, second)}
    assert armies[first] == armies[second], f"{path.stem}: the two armies differ"

    assert turn_income(state, first) == turn_income(state, second)

    stranded = [u.id for u in state.units.values() if not _near_supply_source(state, u)]
    assert stranded == [], f"{path.stem}: units start out of supply: {stranded}"

    for player_id in (first, second):
        objectives = [tile for tile in state.owned_tiles(player_id) if tile.objective]
        assert len(objectives) == 1, f"{path.stem}: {player_id} owns {len(objectives)} objectives"
        guard = next(unit for unit in state.units_of(player_id) if unit.type_id == "hq")
        assert guard.pos == objectives[0].pos


@pytest.mark.parametrize("path", FAIR_MAPS, ids=lambda path: path.stem)
def test_every_fair_map_is_hex_true_mirrored(path):
    """The two halves are the same ground, not a reversed string.

    On odd-r authoring rows the mirror of column ``c`` is ``width - 2 - c``
    (not ``width - 1 - c``), so a plain ``row[::-1]`` would be wrong.  Pinning
    the real transform keeps the visual symmetry the fairness tests assume.
    """
    rows = load_scenario(path)["rows"]
    width = len(rows[0])
    for row_index, row in enumerate(rows):
        for col in range(width):
            mirror = width - 1 - col if row_index % 2 == 0 else width - 2 - col
            if 0 <= mirror < width:
                assert row[col] == row[mirror], (
                    f"{path.stem}: ({col},{row_index}) is {row[col]!r} but its mirror "
                    f"({mirror},{row_index}) is {row[mirror]!r}"
                )


def test_last_stand_is_intentionally_asymmetric_but_still_starts_cleanly():
    path = next(path for path in available_scenarios() if path.stem == "last_stand")
    data = load_scenario(path)
    assert data["fair"] is False

    state = new_game(data, SPECS)
    assert state.status == "ongoing"
    assert len(state.units_of("B")) > len(state.units_of("A")), "the attacker fields the larger army"
    assert [unit.id for unit in state.units.values() if not _near_supply_source(state, unit)] == []


# ---------------------------------------------------------------------------
# scenario validation
# ---------------------------------------------------------------------------


def test_maps_are_refused_before_anyone_can_play_them():
    with pytest.raises(ValueError) as info:
        new_game(scenario(schema=2), SPECS)
    assert "scenario schema 2 is not supported (expected 1)" in str(info.value)

    no_rows = scenario()
    no_rows["rows"] = []
    with pytest.raises(ValueError) as info:
        new_game(no_rows, SPECS)
    assert "scenario has no 'rows' (the map as text rows)" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(rows=["....", "..."]), SPECS)
    assert "map row 1 is 3 chars wide but row 0 is 4 wide" in str(info.value)

    unmapped = list(PLAIN_ROWS)
    unmapped[0] = ".........Q"  # row 0 now contains a terrain nobody defined
    with pytest.raises(KeyError) as info:
        new_game(scenario(rows=unmapped), SPECS)
    assert "unknown terrain code 'Q'" in str(info.value)


def test_ownership_claims_that_do_not_resolve_are_refused():
    with pytest.raises(ValueError) as info:
        new_game(scenario(regions=[{"owner": "C", "cells": [[0, 0]]}]), SPECS)
    assert "region refers to unknown player 'C'" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(regions=[{"owner": "A"}]), SPECS)
    assert "region for 'A' needs either 'offset_cols' or 'cells'" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(regions=[{"owner": "A", "cells": [[99, 99]]}]), SPECS)
    assert "region cell [99, 99] is outside the map" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(tile_overrides=[{"pos": [99, 99], "owner": "A"}]), SPECS)
    assert "tile override [99, 99] is outside the map" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(tile_overrides=[{"pos": [0, 0], "owner": "C"}]), SPECS)
    assert "tile override at [0, 0] refers to unknown player 'C'" in str(info.value)


def test_units_cannot_be_spawned_into_impossible_places():
    with pytest.raises(KeyError) as info:
        new_game(scenario(units=[unit("knight", "A", 1, 1)]), SPECS)
    assert "unknown unit type 'knight'" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(units=[unit("infantry", "C", 1, 1)]), SPECS)
    assert "cannot create a unit for unknown player 'C'" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(units=[unit("infantry", "A", 1, 1, "A1"), unit("infantry", "A", 2, 1, "A1")]), SPECS)
    assert "duplicate unit id 'A1'" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(units=[unit("infantry", "A", 99, 0, "A1")]), SPECS)
    assert "Infantry cannot be placed at" in str(info.value) and "it is not part of the map" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(rows=SEA_ROWS, units=[unit("infantry", "A", 0, 0, "A1")]), SPECS)
    assert "Infantry cannot be placed on Sea at" in str(info.value)
    assert "foot units cannot enter that terrain" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(units=[unit("infantry", "A", 1, 1, "A1"), unit("tank", "A", 1, 1, "A2")]), SPECS)
    assert "cannot place Tank at" in str(info.value) and "A1 is already there" in str(info.value)

    with pytest.raises(ValueError) as info:  # a key no placement understands
        new_game(scenario(units=[{"id": "A1", "type": "infantry", "owner": "A", "offset": [1, 1], "speed": 9}]), SPECS)
    assert "uses unknown keys speed" in str(info.value)

    with pytest.raises(ValueError) as info:  # a unit nobody gave a hex to stand on
        new_game(scenario(units=[{"id": "A1", "type": "infantry", "owner": "A"}]), SPECS)
    assert "needs 'offset' [col,row] or 'pos' [q,r]" in str(info.value)


def test_the_player_table_itself_is_checked():
    with pytest.raises(ValueError) as info:
        new_game(scenario(players=[{"id": "A", "funds": 0}, {"id": "A", "funds": 0}]), SPECS)
    assert "duplicate player id 'A'" in str(info.value)

    nobody = scenario()
    nobody["players"] = []
    with pytest.raises(ValueError) as info:
        new_game(nobody, SPECS)
    assert "scenario needs at least one player" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(scenario(first_player="C"), SPECS)
    assert "first_player 'C' is not one of the scenario players" in str(info.value)


def test_a_minimal_scenario_gets_sane_defaults():
    bare = {"rows": PLAIN_ROWS, "players": [{"id": "Solo"}]}
    state = new_game(bare, SPECS)
    assert state.initial_income == 1000
    assert state.players["Solo"].funds == 8000  # a player with no stated purse starts funded
    assert state.current_player == "Solo", "the first listed player simply starts"
    assert state.turn_number == 1 and state.status == "ongoing"
    assert state.config == {"fog_of_war": False, "max_turns": 200, "damage_luck": True}
