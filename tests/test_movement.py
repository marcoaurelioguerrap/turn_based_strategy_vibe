"""Movement: movement points, terrain costs, blocking, fuel and legal-move lists."""

from __future__ import annotations

import pytest

from tbs.engine.errors import ActionError
from tbs.engine.hexgrid import hex_distance, hex_range
from tbs.engine.rules import legal_moves, unit_options

from .helpers import (
    anchor,
    at,
    cell,
    column_path,
    do,
    end_turn,
    hold,
    line,
    make_state,
    move,
    unit,
)

INFANTRY = [unit("infantry", "A", 5, 2, "A1"), anchor()]

ROUGH_AND_PEAK = ["..........", "..........", "..#MM.....", ".........."]
FOREST_ROW = ["..........", "..........", ".....FFFF.", ".........."]
SEA_AND_PEAK = ["..........", "..........", "....M~....", ".........."]
ENCIRCLED = ["..........", "...MM.....", "...M.M....", "...MM....."]


def test_infantry_crosses_open_ground_in_one_turn():
    state = make_state(units=INFANTRY)
    events = do(state, move("A1", line(5, 2, 3)))

    assert events[0]["type"] == "move"
    assert events[0]["move_cost"] == 3, "plains cost one movement point for foot movement"
    assert at(state, 8, 2).id == "A1"
    assert at(state, 8, 2).hp == 100, "moving by itself does not cost HP"
    assert at(state, 8, 2).moved is True


def test_a_unit_cannot_move_further_than_its_movement_points():
    state = make_state(units=INFANTRY)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(5, 2, 4)))

    assert "movement points" in str(info.value)
    assert "Infantry A1" in str(info.value)
    assert at(state, 5, 2).id == "A1", "a rejected move leaves the unit where it was"
    assert at(state, 5, 2).moved is False


def test_terrain_costs_come_from_the_map_not_the_unit():
    # a tank crosses three forests: six movement points, exactly its limit
    state = make_state(units=[unit("tank", "A", 4, 2, "A1"), anchor()], rows=FOREST_ROW)
    events = do(state, move("A1", line(4, 2, 3)))
    assert events[0]["move_cost"] == 6

    # the fourth forest in the row is out of reach
    state = make_state(units=[unit("tank", "A", 4, 2, "A1"), anchor()], rows=FOREST_ROW)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(4, 2, 4)))
    assert "costs 8" in str(info.value)
    assert "only has 6" in str(info.value)


def test_roads_and_cities_are_cheaper_for_vehicles():
    road = ["..........", "..........", "..rrrrr...", ".........."]
    plains = ["..........", "..........", "..........", ".........."]

    state = make_state(units=[unit("truck", "A", 1, 2, "A1"), anchor()], rows=road)
    events = do(state, move("A1", line(1, 2, 5)))
    assert events[0]["move_cost"] == 5, "a truck spends one movement point per road hex"
    assert at(state, 6, 2).id == "A1"

    # the same five hexes on plains would cost twice as much
    state = make_state(units=[unit("truck", "A", 1, 2, "A1"), anchor()], rows=plains)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(1, 2, 5)))
    assert "costs 10" in str(info.value)


def test_wheeled_units_cannot_enter_rough_or_mountain():
    rough = ["..........", "..........", "..#.......", ".........."]
    peak = ["..........", "..........", "...M......", ".........."]

    state = make_state(units=[unit("truck", "A", 1, 2, "A1"), anchor()], rows=rough)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(1, 2, 1)))
    assert "cannot enter Rough" in str(info.value)

    state = make_state(units=[unit("truck", "A", 2, 2, "A1"), anchor()], rows=peak)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(2, 2, 1)))
    assert "cannot enter Mountain" in str(info.value)

    # they cannot even be deployed on such a hex
    with pytest.raises(ValueError) as info:
        make_state(units=[unit("truck", "A", 2, 2, "A1"), anchor()], rows=ROUGH_AND_PEAK)
    assert "cannot be placed on Rough" in str(info.value)


def test_tracked_units_cross_rough_but_not_mountains():
    state = make_state(units=[unit("tank", "A", 1, 2, "A1"), anchor()], rows=ROUGH_AND_PEAK)
    events = do(state, move("A1", line(1, 2, 1)))
    assert events[0]["move_cost"] == 2, "tracked vehicles do cross rough, just slowly"

    state = make_state(units=[unit("tank", "A", 2, 2, "A1"), anchor()], rows=ROUGH_AND_PEAK)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(2, 2, 1)))
    assert "cannot enter Mountain" in str(info.value)


def test_infantry_crosses_rough_and_mountains_at_double_cost():
    # rough(2) then mountain(2) then mountain(2): one step is affordable, three is not
    state = make_state(units=[unit("infantry", "A", 1, 2, "A1"), anchor()], rows=ROUGH_AND_PEAK)
    events = do(state, move("A1", line(1, 2, 1)))
    assert events[0]["move_cost"] == 2
    assert at(state, 2, 2).id == "A1"

    state = make_state(units=[unit("infantry", "A", 1, 2, "A1"), anchor()], rows=ROUGH_AND_PEAK)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(1, 2, 3)))
    assert "costs 6" in str(info.value)


def test_air_units_cross_sea_and_mountains_for_one_movement_point():
    state = make_state(units=[unit("helicopter", "A", 3, 2, "A1"), anchor()], rows=SEA_AND_PEAK)
    events = do(state, move("A1", line(3, 2, 2)))
    assert events[0]["move_cost"] == 2, "air pays one point for any terrain it may overfly"
    assert at(state, 5, 2).type_id == "helicopter", "it is standing on a sea hex"
    assert at(state, 5, 2).hp == 100, "the sea does not drown aircraft"


def test_units_cannot_move_through_or_into_an_occupied_hex():
    state = make_state(units=[unit("infantry", "A", 5, 2, "A1"), unit("infantry", "B", 7, 2, "B1")])
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(5, 2, 2)))
    assert "occupied by B1" in str(info.value)
    assert at(state, 5, 2).id == "A1"

    # friendly units block exactly like enemy ones
    state = make_state(
        units=[unit("infantry", "A", 5, 2, "A1"), unit("infantry", "A", 6, 2, "A2"), anchor()]
    )
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(5, 2, 1)))
    assert "occupied by A2" in str(info.value)

    # walking around a blocking unit is legal
    state = make_state(units=[unit("infantry", "A", 5, 2, "A1"), unit("infantry", "B", 7, 2, "B1")])
    around = [cell(5, 2), cell(6, 2), *column_path(6, 2, 1)[1:]]
    do(state, move("A1", around))
    assert at(state, 6, 3).id == "A1"


def test_paths_must_start_at_the_unit_and_step_one_hex_at_a_time():
    state = make_state(units=INFANTRY)

    with pytest.raises(ActionError) as info:
        do(state, move("A1", [cell(4, 2), cell(5, 2)]))
    assert "must start at" in str(info.value)
    assert str(list(state.unit("A1").pos)) in str(info.value)

    with pytest.raises(ActionError) as info:
        do(state, move("A1", [cell(5, 2), cell(7, 2)]))
    assert "not adjacent" in str(info.value)
    assert at(state, 5, 2).id == "A1"

    edge = make_state(units=[unit("infantry", "A", 9, 2, "A9"), unit("infantry", "B", 0, 0, "B1")])
    with pytest.raises(ActionError) as info:
        do(edge, move("A9", [cell(9, 2), cell(10, 2)]))
    assert "not part of the map" in str(info.value)
    assert at(edge, 9, 2).id == "A9", "the rejected move does not half-happen"


def test_a_one_hex_path_is_not_a_move():
    state = make_state(units=INFANTRY)
    with pytest.raises(ActionError) as info:
        do(state, move("A1", [cell(5, 2)]))
    assert "at least two hexes" in str(info.value)
    assert "hold" in str(info.value), "the error tells the agent what to do instead"


def test_a_unit_may_move_once_and_act_once_in_a_turn():
    state = make_state(units=INFANTRY)
    do(state, move("A1", line(5, 2, 1)))
    do(state, hold("A1"))

    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(6, 2, 1)))
    assert "already acted" in str(info.value)


def test_moving_twice_in_a_turn_is_rejected():
    state = make_state(units=INFANTRY)
    do(state, move("A1", line(5, 2, 1)))
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(6, 2, 1)))
    assert "already moved" in str(info.value)
    assert at(state, 6, 2).id == "A1"


def test_holding_spends_the_turn_and_removes_the_unit_from_the_option_lists():
    state = make_state(units=INFANTRY)
    assert legal_moves(state, "A1")
    do(state, hold("A1"))
    assert legal_moves(state, "A1") == []
    options = unit_options(state, "A1")
    assert options["can_hold"] is False
    assert "already acted this turn" in options["blockers"]


def test_an_hq_never_moves():
    state = make_state(
        rows=["..........", "..........", "..H.......", ".........."],
        units=[unit("hq", "A", 2, 2, "A1"), anchor()],
    )
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(2, 2, 1)))
    assert "cannot move" in str(info.value)
    assert legal_moves(state, "A1") == []
    assert unit_options(state, "A1")["can_move"] is False


def test_fuel_limits_movement_and_is_spent_by_moving():
    state = make_state(units=[unit("tank", "A", 4, 2, "A1"), anchor()])
    state.units["A1"].fuel = 3

    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(4, 2, 2)))
    assert "needs 4 fuel" in str(info.value)
    assert "has 3 left" in str(info.value)
    affordable = legal_moves(state, "A1")
    assert affordable, "one step of plains costs 2 fuel, which it can pay"
    assert all(option["move_cost"] <= 3 for option in affordable)
    assert all(option["fuel_cost"] == option["move_cost"] for option in affordable)
    assert state.units["A1"].fuel == 3, "a rejected move spends nothing"

    events = do(state, move("A1", line(4, 2, 1)))
    assert events[0]["fuel_used"] == 2
    assert state.units["A1"].fuel == 1


def test_a_stranded_land_unit_is_not_destroyed_but_cannot_move():
    state = make_state(units=[unit("tank", "A", 4, 2, "A1"), anchor()])
    state.units["A1"].fuel = 0
    assert legal_moves(state, "A1") == []
    do(state, end_turn())
    assert "A1" in state.units, "only air and naval units are damaged by an empty tank"
    assert state.units["A1"].hp == 100


def test_legal_moves_lists_every_reachable_hex_and_nothing_else():
    """On open plains a foot unit reaches exactly the hex range its movement
    allows, minus hexes that are off the map or occupied."""
    state = make_state(units=INFANTRY)
    mover = state.unit("A1")
    options = legal_moves(state, "A1")

    expected = {
        pos for pos in hex_range(mover.pos, 3) if pos in state.map.tiles and state.unit_at(pos) is None
    }
    expected.discard(mover.pos)
    assert {tuple(option["pos"]) for option in options} == expected

    for option in options:
        path = [tuple(pos) for pos in option["path"]]
        assert path[0] == mover.pos and path[-1] == tuple(option["pos"])
        assert all(hex_distance(a, b) == 1 for a, b in zip(path, path[1:]))
        assert all(pos in state.map.tiles for pos in path)
        assert state.unit_at(path[-1]) is None
        assert option["move_cost"] == hex_distance(mover.pos, path[-1]) <= 3
        assert option["terrain"] == state.map.tile(path[-1]).terrain

    assert options == legal_moves(state, "A1"), "the same position always produces the same list"


def test_every_listed_move_is_accepted_by_the_rules():
    """legal_moves must never advertise a move the rules then refuse."""
    rows = ["..........", ".....F....", "....#.....", ".........."]
    for index in range(4):
        state = make_state(rows=rows, units=[unit("infantry", "A", 3, 2, "A1"), anchor("B")])
        options = legal_moves(state, "A1")
        assert options, "this unit has somewhere to go"
        chosen = options[index % len(options)]
        do(state, move("A1", chosen["path"]))
        assert state.unit("A1").pos == tuple(chosen["pos"])


def test_a_unit_walled_in_by_impassable_terrain_has_no_legal_moves():
    state = make_state(rows=ENCIRCLED, units=[unit("tank", "A", 4, 2, "A1"), anchor()])
    assert legal_moves(state, "A1") == []
    assert unit_options(state, "A1")["can_move"] is False
    with pytest.raises(ActionError) as info:
        do(state, move("A1", column_path(4, 2, 1)))
    assert "cannot enter Mountain" in str(info.value)


def test_unit_options_advertise_a_capture_the_unit_could_start():
    state = make_state(
        rows=["..........", "..C.......", "..........", ".........."],
        units=[unit("infantry", "A", 2, 1, "A1"), anchor()],
    )
    options = unit_options(state, "A1")
    assert [capture_option["pos"] for capture_option in options["captures"]] == [cell(2, 1)]
    assert options["captures"][0]["needed"] == 200
    assert options["captures"][0]["completes_now"] is False
    assert options["can_move"] is True, "standing on a site does not freeze the unit"
