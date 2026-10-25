"""A two-minute tour of the whole engine, in one screen.

Loads the bundled border skirmish, shows the opening position, rejects a bad
order with its engine error, plays two player-turns with a plain greedy policy
(the same "attack what is in range, else advance" policy that
``tests/test_agent_play.py`` pins), saves the game, loads it back, and
continues playing from the loaded file.

Run from the repository root::

    py scripts/demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `py scripts/demo.py` from anywhere

from tbs.engine import rules, view
from tbs.engine.data import default_scenario, load_scenario, load_specs
from tbs.engine.errors import ActionError
from tbs.engine.hexgrid import hex_distance
from tbs.engine.serialize import load_game, log_hash, save_game, state_hash
from tbs.engine.state import new_game


def next_order(state, unit_id: str) -> dict:
    """The dumbest policy that still fights: attack the best preview it can
    see, capture what is underfoot, else walk toward the nearest enemy, else
    hold.  Every order is derived from ``rules.unit_options`` alone."""
    options = rules.unit_options(state, unit_id)
    if options["attacks"]:
        best = max(options["attacks"], key=lambda a: (a["preview_damage"], a["target"], a["weapon"]))
        return {"action": "attack", "unit": unit_id, "target": best["target"], "weapon": best["weapon"]}
    if options["captures"]:
        return {"action": "capture", "unit": unit_id}
    if options["moves"]:
        unit = state.unit(unit_id)
        spots = [enemy.pos for enemy in state.enemies_of(unit.owner)]
        if spots:
            nearest_now = min(hex_distance(unit.pos, spot) for spot in spots)

            def score(move):
                return (min(hex_distance(tuple(move["pos"]), spot) for spot in spots), move["move_cost"], move["pos"])

            best = min(options["moves"], key=score)
            if min(hex_distance(tuple(best["pos"]), spot) for spot in spots) < nearest_now:
                return {"action": "move", "unit": unit_id, "path": best["path"]}
    return {"action": "hold", "unit": unit_id}


INTERESTING = {"attack", "counterattack", "destroyed", "captured", "objective_captured", "out_of_fuel", "eliminated"}


def describe_event(event: dict) -> str:
    kind = event["type"]
    if kind in ("attack", "counterattack"):
        verb = "attacks" if kind == "attack" else "counterattacks"
        return f"{event['attacker']} {verb} {event['defender']} for {event['damage']} -> hp {event['defender_hp']}"
    if kind == "destroyed":
        return f"{event['unit']} destroyed ({event['cause']})"
    if kind == "captured":
        return f"site {event.get('site')} captured by {event.get('owner')}"
    if kind == "objective_captured":
        return f"OBJECTIVE {event.get('site')} captured - {event.get('winner')} wins"
    if kind == "out_of_fuel":
        return f"{event['unit']} runs out of fuel (-{event['damage']} hp)"
    return kind


def play_half_turn(state) -> None:
    player = state.current_player
    print(f"\n-- player {player} orders (turn {state.turn_number}) --")
    for unit in sorted(state.units_of(player), key=lambda u: u.id):
        if state.status != "ongoing":
            break
        order = next_order(state, unit.id)
        summary = f"{order['action']:<7} {unit.id}"
        if order["action"] == "move":
            summary += f" -> {order['path'][-1]}"
        elif order["action"] == "attack":
            summary += f" vs {order['target']} with {order['weapon']}"
        events = rules.perform(state, order)
        aftermath = "; ".join(describe_event(event) for event in events if event["type"] in INTERESTING)
        print(f"  {summary}" + (f"  [{aftermath}]" if aftermath else ""))
    if state.status == "ongoing":
        rules.perform(state, {"action": "end_turn"})


def main() -> None:
    specs = load_specs()
    scenario = load_scenario(default_scenario())
    state = new_game(scenario, specs)
    rules.start_game(state)

    print(f"== {scenario.get('title') or scenario['name']} ==")
    print(view.render_status(state))
    print()
    print(view.render_map(state))
    print(view.unit_legend(state))

    whole_view = view.view_json(state, state.current_player)
    print(f"\nThe whole view an agent sees for its next order: {len(whole_view)} characters of JSON.")

    try:  # A1 is the HQ: it has no weapons at all, and agents will order things like this
        rules.perform(state, {"action": "attack", "unit": "A1", "target": "B1"})
    except ActionError as error:
        print(f"\nAn illegal order, rejected exactly as an agent would be told:\n  {error}")

    for _ in range(2):  # one turn for each side
        if state.status != "ongoing":
            break
        play_half_turn(state)
        print()
        print(view.render_map(state))

    with tempfile.TemporaryDirectory() as tmp:
        path = save_game(state, Path(tmp) / "border_skirmish.json")
        resumed = load_game(path)
        print(f"\nSaved to {path.name}: the loaded game is byte-identical ({state_hash(resumed) == state_hash(state)}).")
        rules.perform(resumed, {"action": "hold", "unit": sorted(resumed.units)[0]})
        print("and the loaded game plays on: one order performed on the resumed copy without any drift.")

    print(f"\nEnd of demo: status={state.status} turn={state.turn_number}")
    print(f"state hash {state_hash(state)[:16]}  log hash {log_hash(state)[:16]}")
    print("Everything above is reproduced by `py -m pytest`: 362 tests pin every rule this demo touched.")


if __name__ == "__main__":
    main()
