"""The agent protocol the engine promises: a prompt-sized view in, validated JSON orders out.

The scripted agent in this file is deliberately the dumbest legal policy one
could write: attack whatever the best preview names, capture what is under its
own feet, otherwise walk toward the nearest enemy or uncaptured objective, and
hold when even that is futile.  It never hard-codes a single unit id, map cell
or damage number - every order it gives is derived from what the engine itself
lists as legal right now.

If even that script can drive the bundled skirmish for six rounds without the
engine rejecting one order, if the whole game then rebuilds exactly from its
recorded orders, and if a reader who sees nothing but the view JSON can pick
valid attacks from it - then a real LLM agent only has to be better at choosing,
not at anything else.
"""

from __future__ import annotations

import json

from tbs.engine import rules, view
from tbs.engine.hexgrid import hex_distance
from tbs.engine.serialize import log_hash, replay, state_hash
from tbs.engine.state import new_game

from .helpers import SPECS, bundled_scenario, do, end_turn, events_of


def start_bundled():
    scenario = bundled_scenario()
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    return state


# ---------------------------------------------------------------------------
# the scripted greedy policy - and what it does in the bundled scenario
# ---------------------------------------------------------------------------


def _approach_targets(state, unit) -> list[tuple[int, int]]:
    """Positions this unit wants to close in on: enemies and unowned capture sites."""
    spots = [enemy.pos for enemy in state.enemies_of(unit.owner)]
    for tile in state.map.tiles.values():
        terrain = state.terrain_of(tile)
        if terrain.capture_points is not None and tile.owner != unit.owner:
            spots.append(tile.pos)
    return spots


def greedy_order(state, unit_id: str) -> dict:
    """The next order for one unit, derived only from ``rules.unit_options``."""
    options = rules.unit_options(state, unit_id)
    if options["attacks"]:
        best = max(options["attacks"], key=lambda a: (a["preview_damage"], a["target"], a["weapon"]))
        return {"action": "attack", "unit": unit_id, "target": best["target"], "weapon": best["weapon"]}
    if options["captures"]:
        return {"action": "capture", "unit": unit_id}
    if options["moves"]:
        unit = state.unit(unit_id)
        spots = _approach_targets(state, unit)
        nearest_now = min(hex_distance(unit.pos, spot) for spot in spots)

        def score(move):
            reached = min(hex_distance(tuple(move["pos"]), spot) for spot in spots)
            return (reached, move["move_cost"], move["pos"])

        best = min(options["moves"], key=score)
        if min(hex_distance(tuple(best["pos"]), spot) for spot in spots) < nearest_now:
            return {"action": "move", "unit": unit_id, "path": best["path"]}
    return {"action": "hold", "unit": unit_id}


def play_greedy(state, half_turns: int) -> list[dict]:
    """Play ``half_turns`` player-turns with the greedy policy, recording every order."""
    orders: list[dict] = []
    for _ in range(half_turns):
        if state.status != "ongoing":
            break
        for unit in sorted(state.units_of(state.current_player), key=lambda u: u.id):
            if state.status != "ongoing":
                break
            order = greedy_order(state, unit.id)
            do(state, order)
            orders.append(order)
        if state.status == "ongoing":
            do(state, end_turn())
            orders.append(end_turn())
    return orders


def battle_events(state) -> list[dict]:
    return [event for entry in state.log for event in entry["events"]]


def test_the_greedy_agent_fights_the_bundled_war_without_rejecting_a_single_order():
    state = start_bundled()
    play_greedy(state, 12)  # six full rounds

    events = battle_events(state)
    attacks = sum(1 for event in events if event["type"] == "attack")
    destroyed = sum(1 for event in events if event["type"] == "destroyed")
    assert attacks >= 10, "the script found the enemy and traded blows"
    assert destroyed >= 4, "greedy trading of blows resolves into actual losses"
    assert state.status == "ongoing", "six rounds of a border skirmish does not decide the whole war"
    for player_id in ("A", "B"):
        assert state.players[player_id].funds > 8000, "income keeps arriving to both sides even mid-battle"


def test_the_agents_whole_game_rebuilds_from_its_recorded_orders_alone():
    played = start_bundled()
    orders = play_greedy(played, 12)

    stored = json.loads(json.dumps(orders))  # an order log must be plain JSON to be stored or shipped
    rebuilt = replay(bundled_scenario(), stored)
    assert state_hash(rebuilt) == state_hash(played)
    assert log_hash(rebuilt) == log_hash(played), "the recorded orders are the game, event for event"


# ---------------------------------------------------------------------------
# the view: prompt-sized JSON that agrees with the rules API
# ---------------------------------------------------------------------------


def test_the_view_is_prompt_sized_json_that_agrees_with_the_rules_api():
    state = start_bundled()
    text = view.view_json(state, "A")
    data = json.loads(text)
    assert data == view.state_view(state, "A")
    assert len(text) < 30_000, f"the whole view is {len(text)} characters - comfortably a prompt, and pinned so it cannot quietly double"

    width, height = data["map_size"]
    assert (width, height) == (state.map.width, state.map.height)
    for key in ("map_terrain_rows", "map_ownership_rows", "map_unit_rows"):
        assert len(data[key]) == height, f"{key} must have one row per map row"
        assert all(len(row) == width for row in data[key]), f"{key} rows must match the map width"

    assert data["asking_about"] == state.current_player == "A"
    own_ids = {unit["id"] for unit in data["units"] if unit["owner"] == "A"}
    assert set(data["unit_actions"]) == own_ids, "only the asking player's units get action lists"
    assert all(summary.get("hold") is True for summary in data["unit_actions"].values()), (
        "every unit can always be ordered to hold: a unit is never left with no legal order"
    )

    assert data["rules"] and data["coordinates"], "the view explains its own rules and coordinates in-band"
    assert any(site["objective"] and site["owner"] == "A" for site in data["sites"]), (
        "the objective that ends the game is visible in the view: agents can aim at winning, not just at shooting"
    )

    for unit_id, summary in data["unit_actions"].items():
        api_attacks = [
            [o["target"], o["weapon"], o["distance"], o["preview_damage"]]
            for o in rules.attack_options(state, unit_id)
        ]
        api_moves = [[m["pos"], m["move_cost"]] for m in rules.legal_moves(state, unit_id)]
        assert summary.get("attack", []) == api_attacks, "the view lists exactly the attacks the rules API offers"
        assert summary.get("move_to", []) == api_moves, "the view lists exactly the moves the rules API offers"
        for target_id, _weapon, _dist, _preview in api_attacks:
            assert state.unit(target_id).owner == "B", "attack options only ever name real enemies"


# ---------------------------------------------------------------------------
# acting on the view alone: JSON in, validated orders out
# ---------------------------------------------------------------------------


def test_an_agent_can_act_on_the_view_json_alone():
    state = start_bundled()
    play_greedy(state, 4)  # advance the armies until the position genuinely offers choices
    data = json.loads(view.view_json(state, state.current_player))

    chosen = 0
    attacked_targets: set[str] = set()
    for unit_id, summary in data["unit_actions"].items():
        if unit_id not in state.units:
            continue  # died as a defender earlier in this same ordering
        rows = [row for row in summary.get("attack", []) if row[0] not in attacked_targets]
        if not rows:
            continue
        target_id, weapon, _distance, preview = max(rows, key=lambda row: (row[3], row[0], row[1]))
        order = {"action": "attack", "unit": unit_id, "target": target_id, "weapon": weapon}

        events = do(state, order)  # an order chosen from view JSON alone must validate
        hit = events_of(events, "attack")[0]
        assert hit["attacker"] == unit_id and hit["defender"] == target_id and hit["weapon"] == weapon
        # the preview is honest: the real roll never leaves the luck band around it
        assert preview * 88 // 100 <= hit["damage"] <= preview * 112 // 100, (
            f"{unit_id} vs {target_id}: previewed {preview}, dealt {hit['damage']}"
        )
        attacked_targets.add(target_id)
        chosen += 1

    assert chosen >= 1, "the position really offered attacks that a view-only reader could find and order"
