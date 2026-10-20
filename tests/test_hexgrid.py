"""Hex-grid maths, offset authoring conversions and pathfinding.

These are the pieces every other rule depends on, and the neighbour order in
``AXIAL_DIRECTIONS`` is part of the game contract, so it is pinned here.
"""

from __future__ import annotations

import pytest

from tbs.engine.hexgrid import (
    AXIAL_DIRECTIONS,
    axial_to_offset,
    axial_to_pixel,
    cube,
    find_path,
    hex_add,
    hex_cells,
    hex_distance,
    hex_line,
    hex_neighbors,
    hex_range,
    offset_to_axial,
    pixel_to_axial,
    reachable_hexes,
)

ROWS = [
    "bbbbbbbb",
    "b......b",
    "b..F...b",
    "bbbbbbbb",
]


def test_cube_coordinates_satisfy_q_plus_r_plus_s_zero():
    for pos in [(0, 0), (3, -1), (-4, 7), (2, 2)]:
        q, r, s = cube(pos)
        assert (q, r) == pos
        assert q + r + s == 0


def test_neighbour_order_is_the_documented_contract():
    assert AXIAL_DIRECTIONS == ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def test_each_hex_has_exactly_six_neighbours_at_distance_one():
    here = (2, -1)
    around = hex_neighbors(here)
    assert len(around) == 6
    assert len(set(around)) == 6
    assert all(hex_distance(here, other) == 1 for other in around)
    assert around == [hex_add(here, direction) for direction in AXIAL_DIRECTIONS]


def test_distance_and_range_are_consistent():
    center = (0, 0)
    assert hex_distance(center, center) == 0
    for dist in (0, 1, 2, 4):
        inside = hex_range(center, dist)
        assert len(inside) == 1 + 3 * dist * (dist + 1)
        assert all(hex_distance(center, hex_) <= dist for hex_ in inside)
        assert all(hex_ in inside for hex_ in inside)
        if dist:
            for hex_ in inside:
                if hex_distance(center, hex_) == dist:
                    assert any(hex_distance(step, hex_) == 1 for step in hex_range(center, dist - 1))
    assert hex_range(center, -1) == []


def test_distance_is_symmetric_and_matches_a_walk():
    a, b = (-3, 5), (4, -2)
    steps = hex_distance(a, b)
    assert hex_distance(b, a) == steps
    walked = {a}
    for _ in range(steps):
        walked = {n for hex_ in walked for n in hex_neighbors(hex_)}
    assert b in walked


def test_straight_lines_between_two_hexes_are_contiguous():
    for a, b in [((0, 0), (3, 0)), ((0, 0), (0, 3)), ((0, 0), (3, -3)), ((1, 1), (-2, 4))]:
        between = hex_line(a, b)
        assert hex_distance(a, b) == len(between) + 1
        assert between[0] != a and between[-1] != b
        for from_hex, to_hex in zip([a, *between], [*between, b]):
            assert hex_distance(from_hex, to_hex) == 1
    assert hex_line((0, 0), (1, 0)) == []


def test_horizontal_and_vertical_neighbours_are_always_adjacent():
    """In odd-r offset authoring, same-row and same-column cells touch; only
    diagonals depend on row parity.  Map authors (and tests) rely on this."""
    for row in range(4):
        for col in range(4):
            here = offset_to_axial(col, row)
            for neighbour in (offset_to_axial(col + 1, row), offset_to_axial(col, row + 1)):
                assert hex_distance(here, neighbour) == 1, (col, row)


def test_offset_and_axial_conversions_round_trip():
    for row in range(5):
        for col in range(6):
            axial = offset_to_axial(col, row)
            assert axial[1] == row, "the authored row is also the axial r"
            assert axial_to_offset(axial) == (col, row)


def test_odd_rows_are_shifted_right_by_half_a_hex():
    """A cell in an odd row sits between two cells of the even row above it, so
    which diagonal is a neighbour depends on row parity."""
    for col in range(1, 4):
        shifted = offset_to_axial(col, 1)
        assert hex_distance(shifted, offset_to_axial(col, 0)) == 1
        assert hex_distance(shifted, offset_to_axial(col + 1, 0)) == 1
        assert hex_distance(shifted, offset_to_axial(col - 1, 0)) == 2

    # an even row is not shifted, so its diagonals point the other way
    for col in range(1, 4):
        here = offset_to_axial(col, 2)
        assert hex_distance(here, offset_to_axial(col, 3)) == 1
        assert hex_distance(here, offset_to_axial(col - 1, 3)) == 1
        assert hex_distance(here, offset_to_axial(col + 1, 3)) == 2


def test_text_maps_turn_into_cells_at_the_right_positions():
    cells = hex_cells(ROWS)
    assert len(cells) == len(ROWS) * len(ROWS[0])
    assert cells[offset_to_axial(3, 2)] == "F"
    assert cells[offset_to_axial(1, 1)] == "."
    assert cells[offset_to_axial(0, 0)] == "b"
    # every authored position round-trips back to the character that was typed
    for row, text in enumerate(ROWS):
        for col, code in enumerate(text):
            assert cells[offset_to_axial(col, row)] == code


def test_pixel_round_trip_finds_the_same_hex():
    size = 34.0
    for col in range(8):
        for row in range(4):
            axial = offset_to_axial(col, row)
            x, y = axial_to_pixel(axial, size)
            # sampling the centre of a hex, and a point a third of a hex away
            # from it, must both land on that same hex.
            assert pixel_to_axial(x, y, size) == axial
            assert pixel_to_axial(x + size * 0.3, y + size * 0.2, size) == axial


def uniform_cost(_hex):
    return 1


def test_reachable_hexes_cover_the_whole_hex_range_when_everything_is_passable():
    start = (0, 0)
    found = reachable_hexes(start, 3, uniform_cost)
    assert set(found) == set(hex_range(start, 3))
    for hex_, (cost, path) in found.items():
        assert cost == hex_distance(start, hex_)
        assert path[0] == start and path[-1] == hex_
        assert all(hex_distance(a, b) == 1 for a, b in zip(path, path[1:]))
    assert found[start] == (0, [start])


def test_reachable_hexes_respect_costs_budgets_and_impassable_terrain():
    # entering column 1 is expensive, entering column 2 is impossible
    def cost(hex_):
        if hex_[0] == 2:
            return None
        if hex_[0] == 1:
            return 4
        return 1

    found = reachable_hexes((0, 0), 5, cost)
    assert (0, 1) in found and found[(0, 1)][0] == 1
    assert (1, 0) in found and found[(1, 0)][0] == 4
    assert not any(hex_[0] >= 2 for hex_ in found), "impassable and over-budget hexes are excluded"
    assert max(spent for spent, _ in found.values()) <= 5, "the budget is never exceeded"


def test_reachable_hexes_cannot_cross_blocked_hexes():
    start, goal = (0, 0), (3, 0)
    ring = set(hex_neighbors(start)) - {goal}
    found = reachable_hexes(start, 6, uniform_cost, blocked=ring)
    assert found == {start: (0, [start])}
    assert find_path(start, goal, 6, uniform_cost, blocked=ring) is None
    assert find_path(start, goal, 6, uniform_cost) == [start, (1, 0), (2, 0), (3, 0)]


def test_pathfinding_is_deterministic_and_picks_the_cheapest_route():
    def cost(hex_):
        return 5 if hex_ == (1, 0) else 1

    first = reachable_hexes((0, 0), 4, cost)
    assert first == reachable_hexes((0, 0), 4, cost), "the same input always gives the same answer"

    spent, path = first[(2, 0)]
    assert spent == 3, "three cheap steps, instead of one step through the expensive hex"
    assert (1, 0) not in path
    assert all(steps[0] == (0, 0) for _, steps in first.values())

    # the expensive hex is only reachable by paying for it
    assert (1, 0) not in first
    assert reachable_hexes((0, 0), 5, cost)[(1, 0)] == (5, [(0, 0), (1, 0)])


def test_blocked_argument_accepts_a_frozenset_as_well_as_a_set():
    found = reachable_hexes((0, 0), 1, uniform_cost, blocked=frozenset({(1, 0)}))
    assert (1, 0) not in found, "the blocked hex is not entered"
    assert set(found) == {hex_ for hex_ in hex_range((0, 0), 1)} - {(1, 0)}


def test_reordering_the_directions_would_change_generated_paths():
    """Guards the documented assumption that search order is stable: a path to
    an equidistant hex depends on AXIAL_DIRECTIONS, so tests must not silently
    start depending on a different order."""
    paths = {tuple(path) for _, path in reachable_hexes((0, 0), 2, uniform_cost).values()}
    assert len(paths) == len(hex_range((0, 0), 2))
    assert tuple([(0, 0), (1, 0), (2, 0)]) in paths
