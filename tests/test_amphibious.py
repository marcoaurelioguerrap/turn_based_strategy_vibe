"""Amphibious hints, the scripted ferry policy and the coordinator's move-then-act seam."""

from __future__ import annotations

import json

from tbs.agents.codec import encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.llm import build_messages
from tbs.agents.protocol import TurnRequest
from tbs.agents.turn import apply_plan
from tbs.engine import rules, view
from tbs.engine.amphibious import landmass_index
from tbs.engine.data import DATA_ROOT, load_scenario
from tbs.engine.state import new_game

from .helpers import SPECS


def archipelago():
    scenario = load_scenario(DATA_ROOT / "maps" / "archipelago.json")
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    return state


def test_landmass_index_separates_the_two_home_islands():
    state = archipelago()
    islands = landmass_index(state)
    a3 = islands.of(state.unit("A3").pos)
    b3 = islands.of(state.unit("B3").pos)
    a_hq = islands.of(state.unit("A1").pos)
    enemy_hq = islands.of(state.unit("B1").pos)
    assert a3 is not None and b3 is not None
    assert a3 == a_hq
    assert b3 == enemy_hq
    assert a3 != b3
    assert islands.of(state.unit("A10").pos) is None, "a sea transport is not on a landmass"


def test_view_marks_stranded_units_and_names_the_pickup():
    state = archipelago()
    snapshot = view.state_view(state, "A")
    ferry = snapshot["ferry"]
    assert ferry["transports"] == ["A10"]
    assert "A3" in ferry["stranded"]
    assert next(unit for unit in snapshot["units"] if unit["id"] == "A3")["stranded"] is True
    assert next(unit for unit in snapshot["units"] if unit["id"] == "A1").get("stranded") is not True

    operation = next(
        item
        for item in ferry["operations"]
        if item["transport"] == "A10" and item["cargo"] == "A3"
    )
    assert operation["action"] == "pickup"
    assert operation["cargo_to"] == [3, 5], "the opening infantry can reach the beach beside the ship"
    assert operation["target"] == [15, 3], "the enemy HQ is the strategic target"

    site = next(site for site in snapshot["sites"] if site["objective"] and site["owner"] == "B")
    assert site["island"] == next(
        unit for unit in snapshot["units"] if unit["id"] == "B1"
    )["island"]


def test_greedy_agent_boards_then_lands_on_the_enemy_island():
    state = archipelago()
    agent = GreedyAgent()

    first = agent.plan(TurnRequest(player="A", view=view.state_view(state, "A")))
    result = apply_plan(state, first)
    assert result.errors == []
    assert state.unit("A3").loaded is True
    assert state.unit("A10").cargo == ["A3"]

    rules.perform(state, {"action": "end_turn"})
    rules.perform(state, {"action": "end_turn"})
    assert state.current_player == "A"

    second = agent.plan(TurnRequest(player="A", view=view.state_view(state, "A")))
    result = apply_plan(state, second)
    assert result.errors == []
    assert state.unit("A3").loaded is False
    assert state.unit("A3").pos is not None
    islands = landmass_index(state)
    assert islands.of(state.unit("A3").pos) == islands.of(state.unit("B1").pos), (
        "the landing force reached the enemy home island on turn two"
    )


def test_ferry_anchors_survive_the_large_map_move_sample():
    scenario = load_scenario(DATA_ROOT / "maps" / "archipelago_m.json")
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    snapshot = view.state_view(state, "A")
    for operation in snapshot["ferry"]["operations"]:
        for unit_id, key in (
            (operation.get("transport"), "transport_to"),
            (operation.get("cargo"), "cargo_to"),
        ):
            destination = operation.get(key)
            if unit_id is None or destination is None:
                continue
            moves = snapshot["unit_actions"][unit_id].get("move_to") or []
            assert any(list(destination) == row[0] for row in moves), (
                f"{unit_id} must be able to see its ferry destination {destination}"
            )


def test_codec_and_llm_prompt_expose_the_recommended_crossing():
    state = archipelago()
    encoded = encode_state(view.state_view(state, "A"))
    assert "ferry" in encoded
    assert "A10 A3 p 3,5 - - 15,3" in encoded

    messages = build_messages(TurnRequest(player="A", view=view.state_view(state, "A")))
    assert "AMPHIBIOUS OPERATIONS" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["view"]["ferry"]["operations"]
