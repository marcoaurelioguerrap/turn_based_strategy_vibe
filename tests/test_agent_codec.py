"""The compact agent codecs: board text, exact order round-trips, a fixed budget."""

from __future__ import annotations

import json

import pytest

from tbs.agents.codec import (
    CodecError,
    canonical_order,
    decode_order,
    encode_order,
    encode_state,
)
from tbs.agents.greedy import GreedyAgent
from tbs.agents.protocol import AgentPlan
from tbs.agents.turn import normalize_order, run_turn, run_turn_stepwise
from tbs.engine import rules, view
from tbs.engine.serialize import replay, state_hash
from tbs.engine.state import new_game

from .helpers import (
    SPECS,
    anchor,
    bundled_scenario,
    cell,
    column_path,
    do,
    end_turn,
    hold,
    load,
    make_state,
    move,
    unit,
)

#: The reference scenario encodes to 1,061-1,125 characters (511-541 GPT-2
#: tokens).  The cap leaves room for capture progress, cargo flags and a larger
#: army without silently losing the budget.
STATE_CHAR_BUDGET = 1300


def played_state(turns: int = 3):
    """The bundled scenario after ``turns`` greedy half-turns."""
    state = new_game(bundled_scenario(), SPECS)
    rules.start_game(state)
    for _ in range(turns):
        if state.status != "ongoing":
            break
        run_turn(state, GreedyAgent())
    return state


def sections(text: str) -> dict[str, list[str]]:
    """Split encoded state text into its named sections."""
    names = {"players", "terrain", "sites", "units", "builds", "ferry"}
    found: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line in names:
            current = line
            found[current] = []
        elif current is not None:
            found[current].append(line)
    return found


def legal_orders(state) -> list[dict]:
    """Every order the engine currently offers, in agent and engine forms."""
    player = state.current_player
    orders: list[dict] = []
    for current in state.units_of(player):
        unit_id = current.id
        options = rules.unit_options(state, unit_id)
        for option in options["moves"]:
            orders.append({"action": "move", "unit": unit_id, "to": list(option["pos"])})
            orders.append({"action": "move", "unit": unit_id, "path": option["path"]})
        for option in options["attacks"]:
            orders.append(
                {
                    "action": "attack",
                    "unit": unit_id,
                    "target": option["target"],
                    "weapon": option["weapon"],
                }
            )
            orders.append({"action": "attack", "unit": unit_id, "target": option["target"]})
        for _option in options["captures"]:
            orders.append({"action": "capture", "unit": unit_id})
        for option in options["loads"]:
            orders.append({"action": "load", "unit": unit_id, "cargo": option["cargo"]})
        for option in options["unloads"]:
            orders.append(
                {
                    "action": "unload",
                    "unit": unit_id,
                    "cargo": option["cargo"],
                    "target": option["at"],
                }
            )
        if not (current.acted or current.loaded or current.deployed_this_turn or current.pos is None):
            orders.append({"action": "hold", "unit": unit_id})
    for option in rules.build_options(state, player):
        orders.append({"action": "build", "site": option["site"], "unit": option["unit"]})
    orders.append({"action": "end_turn"})
    return orders


# ---------------------------------------------------------------------------
# state text
# ---------------------------------------------------------------------------


def test_encoded_state_is_canonical_bounded_and_keeps_every_board_fact():
    state = played_state(3)
    snapshot = view.state_view(state, state.current_player)
    encoded = encode_state(snapshot)

    assert encode_state(snapshot) == encoded, "the same view always encodes the same text"
    assert len(encoded) <= STATE_CHAR_BUDGET
    full_json = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    assert len(encoded) < len(full_json) // 5, "the codec is a large reduction, not a rename"

    parts = sections(encoded)
    assert parts["terrain"] == snapshot["map_terrain_rows"], "terrain rows are verbatim"
    assert len(parts["terrain"]) == snapshot["map_size"][1]
    assert len(parts["units"]) == len(snapshot["units"])
    assert len(parts["sites"]) == len(snapshot["sites"])
    assert len(parts["builds"]) == len(snapshot["build_options"])
    assert len(parts["players"]) == len(snapshot["players"])

    header = encoded.splitlines()[0]
    assert header.startswith(f"#ncm1 {snapshot['current_player']} t{snapshot['turn']}")

    unit_lines = {line.split()[0]: line.split() for line in parts["units"]}
    assert set(unit_lines) == {current["id"] for current in snapshot["units"]}
    for current in snapshot["units"]:
        fields = unit_lines[current["id"]]
        expected_pos = "-" if current["pos"] is None else f"{current['pos'][0]},{current['pos'][1]}"
        assert fields[1] == current["type"]
        assert fields[2] == expected_pos
        assert fields[3] == str(current["hp"])

    site_lines = {" ".join(line.split()[:2]) for line in parts["sites"]}
    for site in snapshot["sites"]:
        assert f"{site['pos'][0]},{site['pos'][1]} {site['terrain']}" in site_lines


def test_encoding_is_canonical_under_input_order():
    state = played_state(2)
    snapshot = view.state_view(state, state.current_player)
    shuffled = dict(snapshot)
    for key in ("players", "sites", "units", "build_options"):
        shuffled[key] = list(reversed(snapshot[key]))
    assert encode_state(shuffled) == encode_state(snapshot)


def test_a_finished_game_and_capture_progress_are_visible():
    state = played_state(2)
    snapshot = dict(view.state_view(state, state.current_player), status="finished")
    assert encode_state(snapshot).splitlines()[0].endswith(" finished")

    site = snapshot["sites"][0]
    snapshot["sites"] = [dict(site, capture_progress=40, capture_needed=200)]
    line = next(
        item for item in sections(encode_state(snapshot))["sites"] if item.startswith(
            f"{site['pos'][0]},{site['pos'][1]} "
        )
    )
    assert line.endswith(" 40/200")


def test_unit_flags_show_acted_moved_and_cargo():
    state = make_state(
        units=[
            unit("truck", "A", 1, 1, "A1"),
            unit("infantry", "A", 3, 1, "A2"),
            anchor("B"),
        ]
    )
    before = {line.split()[0]: line for line in sections(encode_state(view.state_view(state, "A")))["units"]}
    assert before["A1"] == "A1 truck 1,1 100 50 -"
    assert before["A2"] == "A2 infantry 3,1 100 - rifle:10"

    do(state, move("A1", column_path(1, 1, 1)))
    do(state, hold("A2"))
    encoded = encode_state(view.state_view(state, "A"))
    after = {line.split()[0]: line.split() for line in sections(encoded)["units"]}
    landing = cell(1, 2)
    assert after["A1"][2] == f"{landing[0]},{landing[1]}"
    assert after["A1"][-1] == ">", "a move leaves the unit able to fire, so only moved is set"
    assert after["A2"][-1] == "!", "a hold marks the unit acted"


def test_a_loaded_cargo_is_marked_and_its_position_is_absent():
    state = make_state(
        units=[
            unit("truck", "A", 1, 1, "A1"),
            unit("infantry", "A", 2, 1, "A2"),
            anchor("B"),
        ]
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())
    do(state, end_turn())
    assert state.current_player == "A"

    lines = {
        line.split()[0]: line.split()
        for line in sections(encode_state(view.state_view(state, "A")))["units"]
    }
    assert lines["A1"][-1] == "cA2", "the transport lists its cargo"
    assert lines["A2"][2] == "-", "loaded cargo has no board position"
    assert lines["A2"][-1] == "^", "the cargo is marked loaded"


# ---------------------------------------------------------------------------
# orders
# ---------------------------------------------------------------------------


def test_every_legal_order_round_trips_exactly():
    state = played_state(2)
    orders = legal_orders(state)
    assert len(orders) > 50, "the enumeration is not vacuous"

    for order in orders:
        line = encode_order(order)
        assert line and "\n" not in line
        assert decode_order(line) == canonical_order(order)


def test_engine_move_paths_collapse_to_the_destination_they_expand_back_to():
    state = played_state(2)
    checked = 0
    for current in state.units_of(state.current_player):
        for option in rules.legal_moves(state, current.id):
            engine_order = {"action": "move", "unit": current.id, "path": option["path"]}
            decoded = decode_order(encode_order(engine_order))
            assert normalize_order(state, decoded)["path"] == option["path"]
            checked += 1
    assert checked > 20


def test_a_real_load_then_turn_flip_exposes_unload_orders_that_round_trip():
    state = make_state(
        units=[
            unit("truck", "A", 1, 1, "A1"),
            unit("infantry", "A", 2, 1, "A2"),
            anchor("B"),
        ]
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())
    do(state, end_turn())

    checked = 0
    for option in rules.unload_options(state, "A1"):
        order = {"action": "unload", "unit": "A1", "cargo": option["cargo"], "target": option["at"]}
        assert decode_order(encode_order(order)) == order
        checked += 1
    assert checked > 0, "the fixture did not create an unload choice"


def test_the_wire_form_is_pinned_for_every_action():
    assert encode_order({"action": "move", "unit": "A2", "to": [3, 4]}) == "m A2 3,4"
    assert encode_order({"action": "move", "unit": "A2", "path": [[1, 1], [2, 3], [3, 4]]}) == "m A2 3,4"
    assert encode_order({"action": "attack", "unit": "A5", "target": "B5"}) == "a A5 B5"
    assert (
        encode_order({"action": "attack", "unit": "A5", "target": "B5", "weapon": "howitzer"})
        == "a A5 B5 howitzer"
    )
    assert encode_order({"action": "capture", "unit": "A4"}) == "c A4"
    assert encode_order({"action": "load", "unit": "A6", "cargo": "A2"}) == "l A6 A2"
    assert (
        encode_order({"action": "unload", "unit": "A6", "cargo": "A2", "target": [2, 3]})
        == "u A6 A2 2,3"
    )
    assert encode_order({"action": "build", "site": [1, 1], "unit": "infantry"}) == "b 1,1 infantry"
    assert encode_order({"action": "hold", "unit": "A1"}) == "h A1"
    assert encode_order({"action": "end_turn"}) == "e"

    assert decode_order("e") == {"action": "end_turn"}
    assert decode_order("m A2 -1,4") == {"action": "move", "unit": "A2", "to": [-1, 4]}


def test_extra_keys_are_dropped_in_the_canonical_form():
    order = {
        "action": "attack",
        "unit": "A5",
        "target": "B5",
        "weapon": "howitzer",
        "target_type": "tank",
        "preview_damage": 70,
    }
    assert canonical_order(order) == {
        "action": "attack",
        "unit": "A5",
        "target": "B5",
        "weapon": "howitzer",
    }


@pytest.mark.parametrize(
    "line",
    ["", "   ", "q A1", "m A1", "m A1 3", "u A1 A2", "e A1", "b 1,1"],
)
def test_malformed_lines_are_rejected_with_a_codec_error(line):
    with pytest.raises(CodecError):
        decode_order(line)


@pytest.mark.parametrize(
    "order",
    [
        {"action": "move", "unit": "A1"},
        {"action": "fly", "unit": "A1"},
        {"action": "attack", "unit": "A1"},
        {"action": "build", "unit": "infantry"},
        {"action": "move", "unit": "", "to": [1, 1]},
        {"action": "attack", "unit": "A1", "target": "B1", "weapon": 3},
    ],
)
def test_malformed_orders_are_rejected_with_a_codec_error(order):
    with pytest.raises(CodecError):
        encode_order(order)


class CodecAgent:
    """Picks one view-derived order and sends it through the text codec."""

    name = "codec"

    def __init__(self):
        self.sizes: list[int] = []
        self.lines: list[str] = []

    def plan(self, request):
        self.sizes.append(len(encode_state(request.view)))
        for unit_id, options in sorted(request.view["unit_actions"].items()):
            if options.get("attack"):
                target, weapon, _distance, _preview = options["attack"][0]
                order = {"action": "attack", "unit": unit_id, "target": target, "weapon": weapon}
            elif options.get("move_to"):
                order = {"action": "move", "unit": unit_id, "to": options["move_to"][0][0]}
            elif options.get("capture_remaining_hp") is not None:
                order = {"action": "capture", "unit": unit_id}
            else:
                continue
            line = encode_order(order)
            self.lines.append(line)
            return AgentPlan(orders=[decode_order(line)])
        return AgentPlan(orders=[])


def test_a_stepwise_agent_whose_orders_travel_through_the_codec_plays_and_replays():
    state = played_state(0)
    agent = CodecAgent()

    result = run_turn_stepwise(state, agent, max_steps=40)

    assert agent.lines, "the agent never had a legal order to send"
    assert all("\n" not in line for line in agent.lines)
    assert max(agent.sizes) <= STATE_CHAR_BUDGET
    assert result.applied and result.ended_turn
    rebuilt = replay(bundled_scenario(), json.loads(json.dumps(result.orders)))
    assert state_hash(rebuilt) == state_hash(state)
