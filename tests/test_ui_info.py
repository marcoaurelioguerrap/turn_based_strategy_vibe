"""Hover text: tile and unit facts come from the engine, not the renderer."""

from __future__ import annotations

from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.state import new_game
from tbs.ui import info


def fresh_state():
    state = new_game(load_scenario(available_scenarios()[0]), load_specs())
    rules.start_game(state)
    return state


def test_tile_lines_describe_terrain_owner_and_cover():
    state = fresh_state()
    lines = info.tile_lines(state, state.unit("A1").pos)  # the Blue HQ hex
    text = "\n".join(lines)
    assert "Headquarters" in text
    assert "Blue Republic" in text
    assert "cover 40%" in text
    assert "OBJECTIVE" in text


def test_unit_lines_describe_condition_ammo_rearm_and_status():
    state = fresh_state()
    lines = info.unit_lines(state, state.unit("A2").pos)
    text = "\n".join(lines)
    assert "A2  Infantry" in text
    assert "hp 100/100" in text
    assert "rearm yes" in text
    assert "rifle 10/10" in text
    assert "status: ready" in text


def test_an_acted_unit_says_so_and_a_moved_unit_says_moved():
    state = fresh_state()
    unit = state.unit("A2")
    unit.acted = True
    assert "acted" in "\n".join(info.unit_lines(state, unit.pos))
    unit.acted = False
    unit.moved = True
    assert "moved" in "\n".join(info.unit_lines(state, unit.pos))


def test_hover_lines_stack_tile_and_unit_with_a_separator():
    state = fresh_state()
    lines = info.hover_lines(state, state.unit("A2").pos)
    assert "Plains" in lines[0]
    assert "" in lines
    assert any("A2  Infantry" in line for line in lines)


def test_empty_ground_has_tile_text_but_no_unit_text():
    state = fresh_state()
    empty = next(pos for pos in state.map.tiles if state.unit_at(pos) is None)
    lines = info.hover_lines(state, empty)
    assert lines and not any("hp " in line for line in lines)
