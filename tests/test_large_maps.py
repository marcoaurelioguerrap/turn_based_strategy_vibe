"""The large ``_m`` map pack: ten-times-area maps, the 50-unit cap and long games.

The `_m` files are deliberately separate from the originals.  They keep the
same names plus the suffix, are about ten times the tile area, and are held to
the same fairness and supply checks by ``test_scenarios``.  This file pins the
new dimensions, the per-team cap and a longer greedy game with an exact replay.
"""

from __future__ import annotations

import json

import pytest

from tbs.agents.codec import encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.llm import build_messages
from tbs.agents.protocol import TurnRequest
from tbs.agents.turn import run_turn
from tbs.engine import rules, view
from tbs.engine.data import DATA_ROOT, load_scenario
from tbs.engine.serialize import replay, state_hash
from tbs.engine.state import new_game

from .helpers import SPECS

LARGE_MAPS = {
    "border_skirmish_m": (44, 25, 112),
    "mountain_pass_m": (56, 32, 180),
    "archipelago_m": (64, 38, 240),
    "urban_delta_m": (50, 32, 160),
    "last_stand_m": (50, 32, 160),
}


def path_for(stem: str):
    return DATA_ROOT / "maps" / f"{stem}.json"


def test_the_large_map_pack_ships_beside_the_originals_with_the_m_suffix():
    paths = {path.stem: path for path in (DATA_ROOT / "maps").glob("*.json")}
    for stem, (width, height, old_area) in LARGE_MAPS.items():
        assert stem in paths, f"{stem}.json is missing"
        data = load_scenario(paths[stem])
        assert data["name"] == stem
        assert len(data["rows"]) == height
        assert all(len(row) == width for row in data["rows"])
        assert width * height >= old_area * 9, f"{stem} is not about 10x the old area"
        assert data["max_turns"] >= 100, f"{stem} should allow a longer game"


@pytest.mark.parametrize("stem", LARGE_MAPS)
def test_large_maps_start_with_at_most_fifty_units_per_team(stem):
    state = new_game(load_scenario(path_for(stem)), SPECS)
    assert state.config["max_units_per_player"] == 50
    assert state.config["view_move_limit"] == 12
    assert len(state.units_of("A")) <= 50
    assert len(state.units_of("B")) <= 50

    snapshot = view.state_view(state, state.current_player)
    for actions in snapshot["unit_actions"].values():
        assert len(actions.get("move_to", [])) <= 12


@pytest.mark.parametrize("stem", LARGE_MAPS)
def test_a_large_map_plays_forty_turns_and_replays_exactly(stem):
    scenario = load_scenario(path_for(stem))
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    orders: list[dict] = []

    for _ in range(40):
        if state.status != "ongoing":
            break
        result = run_turn(state, GreedyAgent())
        orders.extend(result.orders)

    rebuilt = replay(scenario, json.loads(json.dumps(orders)))
    assert state_hash(rebuilt) == state_hash(state), f"{stem}: replay diverged"
    assert state.turn_number >= 20, f"{stem}: the long game did not reach turn 20"


def test_the_large_map_prompt_stays_within_the_compact_budget():
    scenario = load_scenario(path_for("archipelago_m"))
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    messages = build_messages(
        TurnRequest(player=state.current_player, view=view.state_view(state, state.current_player))
    )
    payload = messages[1]["content"]
    assert len(messages[0]["content"]) + len(payload) <= 26_000
    compact = json.loads(payload)["view"]
    assert "map_ownership_rows" not in compact, "large prompts drop the duplicate territory layer"
    assert "map_terrain_rows" in compact and "units" in compact

    snapshot = view.state_view(state, state.current_player)
    assert len(encode_state(snapshot)) <= 6_000, "the NCM codec must fit the league prefix"
