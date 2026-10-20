"""The M7 agent server: JSON objects in, JSON objects out, one per line.

The wire layer must never invent, soften or hide an engine rule: every answer
comes straight from the tested view/options/order API, every rejection is a
one-sentence ``{"ok": false, "error": ...}`` the agent can retry from, and bad
input of any kind is answered instead of crashing the process.  As everywhere
else in this suite, nothing here hard-codes an opening attack: positions are
played first and orders are derived from what the server itself reports.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tbs
from tbs.api.server import ACTION_SHAPES, COMMANDS, PROTOCOL, Session
from tbs.engine.serialize import state_hash


def served_session():
    return Session.new()


def served_view(session) -> dict:
    reply = session.handle({"cmd": "state"})
    assert reply["ok"], reply["error"]
    return reply["view"]


def first_attack_order(view: dict) -> tuple[str, list]:
    """An attack derived from the served view, never from a hard-coded board."""
    for unit_id, summary in view["unit_actions"].items():
        rows = summary.get("attack") or []
        if rows:
            return unit_id, max(rows, key=lambda row: row[3])
    raise AssertionError("the served view offered no attack to order")


# ---------------------------------------------------------------------------
# the protocol itself
# ---------------------------------------------------------------------------


def test_the_server_introduces_itself_and_serves_its_whole_protocol():
    session = served_session()
    hello = session.handle({"cmd": "hello"})
    assert hello["ok"] is True
    assert hello["protocol"] == PROTOCOL
    assert hello["scenario"] == "border_skirmish"
    assert sorted(hello["players"]) == ["A", "B"]
    assert hello["commands"] == COMMANDS, "the documented command list is the served one, verbatim"
    assert hello["status"] == "ongoing" and hello["turn"] == 1 and hello["current_player"] == "A"

    help_reply = session.handle({"cmd": "help"})
    assert help_reply["ok"] and help_reply["orders"] == ACTION_SHAPES, "the order schema lives in the server, not only in the README"


def test_state_serves_the_prompt_view_defaulting_to_the_player_to_move():
    session = served_session()
    view = served_view(session)
    width, height = view["map_size"]
    assert (width, height) == (session.state.map.width, session.state.map.height)
    for key in ("map_terrain_rows", "map_ownership_rows", "map_unit_rows"):
        assert len(view[key]) == height and all(len(row) == width for row in view[key])
    assert view["asking_about"] == "A" == session.state.current_player

    other = session.handle({"cmd": "state", "player": "B"})
    assert other["ok"] and other["view"]["asking_about"] == "B", "any side may ask for its own view"

    refused = session.handle({"cmd": "state", "player": "Z"})
    assert refused["ok"] is False and "no player" in refused["error"]


# ---------------------------------------------------------------------------
# playing over the wire
# ---------------------------------------------------------------------------


def test_an_agent_can_act_from_the_served_view_alone():
    session = served_session()
    view = served_view(session)
    unit_id, (target_id, weapon, _distance, _preview) = first_attack_order(view)

    reply = session.handle({"cmd": "order", "action": "attack", "unit": unit_id, "target": target_id, "weapon": weapon})
    assert reply["ok"], reply["error"]
    hits = [event for event in reply["events"] if event["type"] == "attack"]
    assert hits and hits[0]["attacker"] == unit_id and hits[0]["defender"] == target_id
    assert reply["state_hash"] == state_hash(session.state), "the hash the agent is told is the real position hash"
    assert reply["status"] == "ongoing"

    passed = session.handle({"cmd": "order", "action": "end_turn"})
    assert passed["ok"] and passed["current_player"] == "B", "turn flow is visible in every reply"

    status = session.handle({"cmd": "status"})
    assert status["ok"] and status["current_player"] == "B" and status["turn"] == 1
    assert all(side["units"] > 0 for side in status["players"].values())


def test_options_answer_what_a_unit_may_do_and_rejections_cost_the_game_nothing():
    session = served_session()
    view = served_view(session)
    idle_id = next(
        unit_id for unit_id, summary in view["unit_actions"].items() if not summary.get("attack")
    )  # something with no targets in range (the HQ, a ship without ships...)
    enemy_id = next(unit["id"] for unit in view["units"] if unit["owner"] == "B")

    options = session.handle({"cmd": "options", "unit": idle_id})
    assert options["ok"] and options["options"]["attacks"] == []
    assert options["options"]["can_hold"] is True and isinstance(options["options"]["blockers"], list)

    before = state_hash(session.state)
    refused = session.handle({"cmd": "order", "action": "attack", "unit": idle_id, "target": enemy_id})
    assert refused["ok"] is False and refused["error"], "a bad order comes back as a sentence, not a stack trace"
    assert state_hash(session.state) == before, "a rejected order changed nothing at all"

    held = session.handle({"cmd": "order", "action": "hold", "unit": idle_id})
    assert held["ok"], "the fallback every unit always has is served over the wire too"

    again = session.handle({"cmd": "options", "unit": idle_id})
    assert again["ok"] and again["options"]["can_hold"] is False, "the served options track the served game"


# ---------------------------------------------------------------------------
# save, load, continuation
# ---------------------------------------------------------------------------


def test_a_game_played_and_saved_through_the_server_continues_identically_after_load(tmp_path):
    played = Session.new(root=tmp_path)
    assert played.handle({"cmd": "order", "action": "hold", "unit": "A1"})["ok"]

    saved = played.handle({"cmd": "save", "path": "mid_game.json"})
    assert saved["ok"] and saved["path"] == "mid_game.json" and (tmp_path / "mid_game.json").exists()

    continuing = Session.new(root=tmp_path)
    loaded = continuing.handle({"cmd": "load", "path": "mid_game.json"})
    assert loaded["ok"] and state_hash(continuing.state) == state_hash(played.state), "the loaded game is the saved one, played state included"

    repeated = continuing.handle({"cmd": "order", "action": "hold", "unit": "A1"})
    assert repeated["ok"] is False and "already acted" in repeated["error"], "the loaded copy remembers what the original did"

    assert played.handle({"cmd": "save", "path": "../escape.json"})["ok"] is False
    assert played.handle({"cmd": "save", "path": str(tmp_path / 'absolute.json')})["ok"] is False
    assert played.handle({"cmd": "load", "path": "missing.json"})["ok"] is False


# ---------------------------------------------------------------------------
# bad input is answered, never fatal
# ---------------------------------------------------------------------------


def test_broken_lines_and_unknown_commands_are_answered_never_crashed():
    session = served_session()
    for raw_line in ["not json at all", "[]", '{"cmd": 42}', '{"action":"move"}', '{"cmd": "teleport"}']:
        reply = json.loads(session.handle_line(raw_line))
        assert reply["ok"] is False and reply["error"], raw_line

    unknown = json.loads(session.handle_line('{"cmd": "teleport"}'))
    assert "unknown command" in unknown["error"] and "hello" in unknown["error"], "rejections name the verbs an agent may use instead"

    status = session.handle({"cmd": "status"})
    assert status["ok"] and status["status"] == "ongoing", "no bad input can wound the game behind the wire"


# ---------------------------------------------------------------------------
# the process really runs as `py -m tbs.api.server`
# ---------------------------------------------------------------------------


def test_the_server_answers_five_lines_over_real_stdio(tmp_path):
    repo_root = Path(tbs.__file__).resolve().parents[1]
    requests = [
        {"cmd": "hello"},
        {"cmd": "state"},
        {"cmd": "order", "action": "end_turn"},
        {"cmd": "save", "path": "wire_check.json"},
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "tbs.api.server", "--root", str(tmp_path)],
        input="\n".join(json.dumps(request) for request in requests) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
    )
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == len(requests) + 1, "one greeting plus one reply per request"
    greeting, hello, state_reply, turn_passed, saved = (json.loads(line) for line in lines)

    assert greeting.get("event") == "hello" and hello["ok"] and hello["protocol"] == PROTOCOL
    assert state_reply["view"]["asking_about"] == "A"
    assert turn_passed["ok"] and turn_passed["current_player"] == "B", "the game on the wire really advanced"
    assert saved["ok"] and (tmp_path / "wire_check.json").exists()
    for line in lines:
        assert line.isascii(), "JSON lines must survive any pipe encoding"
