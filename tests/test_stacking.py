"""Stacking: aircraft fly above exactly one surface unit.

These tests pin the two-layer board:

* an air unit may share a hex with one land or naval unit;
* two aircraft or two surface units still block each other;
* movement, attacks, unloading, production and the views all see both layers.
"""

from __future__ import annotations

import pytest

from tbs.engine import view
from tbs.engine.data import load_specs
from tbs.engine.errors import ActionError
from tbs.engine.rules import attack_options, build_options, legal_moves, unload_options
from tbs.engine.state import new_game
from tbs.ui import info

from .helpers import (
    anchor,
    build,
    cell,
    do,
    end_turn,
    line,
    load,
    make_state,
    move,
    scenario,
    unit,
    unload,
)

SPECS = load_specs()

SEA_ROW = ["..........", "...~~~~...", "..........", ".........."]
FACTORY_ROW = ["..........", "..f.......", "..........", ".........."]


def test_aircraft_can_share_a_hex_with_a_ground_unit_in_either_authoring_order():
    first_ground = make_state(
        units=[unit("infantry", "A", 3, 1, "A1"), unit("attack_plane", "A", 3, 1, "A2"), anchor()]
    )
    assert [item.id for item in first_ground.units_at(cell(3, 1))] == ["A1", "A2"]
    assert first_ground.unit("A1").pos == first_ground.unit("A2").pos
    assert first_ground.unit_at(cell(3, 1)).id == "A2", "the aircraft is the top layer"

    first_air = make_state(
        units=[unit("attack_plane", "A", 3, 1, "A2"), unit("infantry", "A", 3, 1, "A1"), anchor()]
    )
    assert [item.id for item in first_air.units_at(cell(3, 1))] == ["A1", "A2"]


def test_aircraft_can_share_a_hex_with_a_naval_unit():
    state = make_state(
        rows=SEA_ROW,
        units=[unit("destroyer", "A", 4, 1, "A1"), unit("attack_plane", "A", 4, 1, "A2"), anchor()],
    )
    assert [item.id for item in state.units_at(cell(4, 1))] == ["A1", "A2"]


def test_two_aircraft_or_two_surface_units_still_cannot_share():
    with pytest.raises(ValueError) as info:
        new_game(
            scenario(
                units=[
                    unit("attack_plane", "A", 3, 1, "A1"),
                    unit("helicopter", "A", 3, 1, "A2"),
                ]
            ),
            SPECS,
        )
    assert "cannot place Helicopter" in str(info.value) and "A1 is already there" in str(info.value)

    with pytest.raises(ValueError) as info:
        new_game(
            scenario(
                units=[
                    unit("infantry", "A", 3, 1, "A1"),
                    unit("tank", "A", 3, 1, "A2"),
                ]
            ),
            SPECS,
        )
    assert "cannot place Tank" in str(info.value) and "A1 is already there" in str(info.value)


def test_aircraft_moves_onto_and_through_ground_units():
    state = make_state(
        units=[unit("attack_plane", "A", 1, 1, "A1"), unit("infantry", "A", 2, 1, "A2"), anchor()]
    )
    assert cell(2, 1) in [move["pos"] for move in legal_moves(state, "A1")]
    do(state, move("A1", line(1, 1, 1)))
    assert state.unit("A1").pos == tuple(cell(2, 1))
    assert state.unit("A2").pos == tuple(cell(2, 1))

    state = make_state(
        units=[unit("attack_plane", "A", 1, 1, "A1"), unit("infantry", "A", 2, 1, "A2"), anchor()]
    )
    do(state, move("A1", line(1, 1, 3)))
    assert state.unit("A1").pos == tuple(cell(4, 1))
    assert state.unit("A2").pos == tuple(cell(2, 1)), "the aircraft flew over the infantry"


def test_ground_moves_under_an_aircraft():
    state = make_state(
        units=[unit("infantry", "A", 1, 1, "A1"), unit("attack_plane", "A", 2, 1, "A2"), anchor()]
    )
    assert cell(2, 1) in [move["pos"] for move in legal_moves(state, "A1")]
    do(state, move("A1", line(1, 1, 1)))
    assert state.unit("A1").pos == tuple(cell(2, 1))
    assert state.unit("A2").pos == tuple(cell(2, 1))


def test_same_layer_units_still_block_movement():
    state = make_state(
        units=[unit("attack_plane", "A", 1, 1, "A1"), unit("attack_plane", "B", 2, 1, "B1"), anchor("A")]
    )
    assert cell(2, 1) not in [option["pos"] for option in legal_moves(state, "A1")]
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(1, 1, 1)))
    assert "occupied by B1" in str(info.value)

    state = make_state(
        units=[unit("infantry", "A", 1, 1, "A1"), unit("infantry", "B", 2, 1, "B1"), anchor("A")]
    )
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(1, 1, 1)))
    assert "occupied by B1" in str(info.value)


def test_stacked_enemies_are_separate_attack_targets():
    state = make_state(
        units=[
            unit("aa_missile", "A", 2, 1, "A1"),
            unit("infantry", "B", 3, 1, "B1"),
            unit("attack_plane", "B", 3, 1, "B2"),
            anchor("A"),
        ]
    )
    options = attack_options(state, "A1")
    assert {option["target"] for option in options} == {"B1", "B2"}
    assert {option["weapon"] for option in options if option["target"] == "B2"} == {"sam", "autocannon"}


def test_cargo_can_unload_under_an_aircraft():
    state = make_state(
        units=[
            unit("truck", "A", 3, 1, "A1"),
            unit("infantry", "A", 2, 1, "A2"),
            unit("attack_plane", "A", 4, 1, "A3"),
            anchor("B"),
        ]
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())
    do(state, end_turn())
    assert cell(4, 1) in [option["at"] for option in unload_options(state, "A1")]
    do(state, unload("A1", "A2", cell(4, 1)))
    assert state.unit("A2").pos == tuple(cell(4, 1))
    assert state.unit("A3").pos == tuple(cell(4, 1))


def test_an_aircraft_over_a_factory_does_not_block_ground_production():
    state = make_state(
        rows=FACTORY_ROW,
        units=[unit("attack_plane", "A", 2, 1, "A1"), anchor("B")],
        regions=[{"owner": "A", "cells": [cell(2, 1)]}],
    )
    assert "tank" in {option["unit"] for option in build_options(state, "A")}
    do(state, build(2, 1, "tank"))
    built = next(item for item in state.units.values() if item.type_id == "tank")
    assert built.pos == tuple(cell(2, 1))
    assert state.unit("A1").pos == tuple(cell(2, 1))


def test_views_expose_both_layers_and_the_stack():
    state = make_state(
        units=[unit("infantry", "A", 3, 1, "A1"), unit("attack_plane", "A", 3, 1, "A2"), anchor()]
    )
    structured = view.state_view(state, "A")
    assert {"pos": cell(3, 1), "units": ["A1", "A2"]} in structured["unit_stacks"]
    assert structured["map_ground_unit_rows"][1][3] == "i"
    assert structured["map_air_unit_rows"][1][3] == "P"
    assert structured["map_unit_rows"][1][3] == "P", "the air layer is drawn on top"

    coords = view.cell_coordinates(state, 1, 3)
    assert coords["units"] == ["A1", "A2"]
    assert coords["unit"] == "A2"


def test_hover_text_lists_every_unit_in_a_stack():
    state = make_state(
        units=[unit("infantry", "A", 3, 1, "A1"), unit("attack_plane", "A", 3, 1, "A2"), anchor()]
    )
    text = "\n".join(info.unit_lines(state, cell(3, 1)))
    assert "A1  Infantry" in text
    assert "A2  Attack Plane" in text
