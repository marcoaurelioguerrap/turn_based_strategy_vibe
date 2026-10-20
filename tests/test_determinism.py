"""The engine is deterministic by construction, and this file keeps it that way.

Nothing random is ever stored in a game state.  Every dice roll is derived from
the game seed plus the exact context of the act that rolls it (turn number,
action sequence number, unit ids, weapon id), so:

* the same scenario driven by the same ordered actions always produces the same
  events, the same positions and the same hashes - even with damage luck ON;
* ``replay()`` can rebuild a game from its action log alone, and a seed override
  gives identical what-if runs;
* save files are plain JSON: a loaded game continues exactly where the original
  left off;
* asking the engine "what could this unit do?" (options, previews) consumes
  nothing and changes nothing - frontends and agents may enumerate everything,
  twice, before every order;
* the position hash and the history hash are separate: a log-free copy of a
  game has an identical position hash and a different history hash - the log is
  the only part the position hash ignores, so an acted unit, a spent fuel gauge
  or a different seed all count as new positions.

The brawl used here is two APCs trading shots with their own supply Trucks
parked right behind them (supply rules are pinned in ``test_logistics``).  APC
machine guns are weak against their own light class (6 damage a shot), so the
trade survives a full three rounds; Tank main guns do 80 against heavies and
would bury one Tank in two rounds, and a script that ends in a corpse is no
use for comparing runs.
"""

from __future__ import annotations

import json

import pytest

from tbs.engine import rules
from tbs.engine.rng import LUCK_RANGE, luck_percent, roll_percent
from tbs.engine.serialize import (
    canonical_json,
    load_game,
    log_hash,
    replay,
    save_game,
    state_from_dict,
    state_hash,
    state_to_dict,
)
from .helpers import (
    anchor,
    at,
    attack,
    bundled_scenario,
    column_path,
    do,
    end_turn,
    line,
    make_state,
    move,
    scenario,
    unit,
)

OPEN_GROUND_ROWS = ["..........", "..........", "..........", ".........."]


def brawl_units() -> list[dict]:
    return [
        unit("truck", "A", 3, 1, "A2"),  # each APC fights with its own supply vehicle at hand
        unit("apc", "A", 4, 1, "A1"),
        unit("apc", "B", 5, 1, "B1"),
        unit("truck", "B", 6, 1, "B2"),
        anchor("A"),
        anchor("B"),
    ]


def brawl_kwargs(**overrides) -> dict:
    kwargs: dict = dict(rows=OPEN_GROUND_ROWS, units=brawl_units(), seed=7, damage_luck=True)
    kwargs.update(overrides)
    return kwargs


def brawl_actions() -> list[dict]:
    """Three rounds of trade: one shot and its counter on each side, per turn."""
    actions: list[dict] = []
    for _ in range(3):
        actions += [attack("A1", "B1"), end_turn(), attack("B1", "A1"), end_turn()]
    return actions


def play_brawl(**overrides):
    state = make_state(**brawl_kwargs(**overrides))
    for action in brawl_actions():
        do(state, action)
    return state


def damages(state) -> list[int]:
    """Every damage number the log recorded, in order."""
    return [event["damage"] for entry in state.log for event in entry["events"] if "damage" in event]


# ---------------------------------------------------------------------------
# replays
# ---------------------------------------------------------------------------


def test_the_same_seed_and_orders_rebuild_the_exact_same_game():
    interactive = play_brawl()
    from_log = replay(scenario(**brawl_kwargs()), brawl_actions())
    assert state_hash(from_log) == state_hash(interactive)
    assert log_hash(from_log) == log_hash(interactive), "replays are event-for-event exact"


def test_replays_are_exact_even_with_damage_luck_running():
    # The brawl scenario leaves damage luck ON; the dice come from seed plus
    # action context, never from a hidden stream, so nothing about the replay
    # depends on how many times you ask for it.
    first = replay(scenario(**brawl_kwargs()), brawl_actions())
    second = replay(scenario(**brawl_kwargs()), brawl_actions())
    assert damages(first) == damages(second)
    assert state_hash(first) == state_hash(second)
    assert log_hash(first) == log_hash(second)


def test_a_replay_with_a_chosen_seed_is_its_own_exact_game():
    base = scenario(**brawl_kwargs())
    alt_first = replay(base, brawl_actions(), seed=4242)
    alt_again = replay(base, brawl_actions(), seed=4242)
    assert alt_first.seed == 4242
    assert log_hash(alt_first) == log_hash(alt_again)
    assert base["seed"] == 7, "the scenario dict itself is not touched by a seed override"


def test_different_seeds_are_different_games_when_the_dice_matter():
    outcomes = set()
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        first = replay(scenario(**brawl_kwargs()), brawl_actions(), seed=seed)
        again = replay(scenario(**brawl_kwargs()), brawl_actions(), seed=seed)
        assert log_hash(first) == log_hash(again), "each seed is exact for itself"
        outcomes.add(log_hash(first))
    assert len(outcomes) > 1, "the seed genuinely drives the dice, it is not decoration"


# ---------------------------------------------------------------------------
# luck, and the absence of hidden state
# ---------------------------------------------------------------------------


def test_damage_luck_stays_inside_its_band():
    lucky = play_brawl()  # damage_luck=True
    exact = play_brawl(damage_luck=False)  # the same script with the dice switched off

    lucky_damages = damages(lucky)
    exact_damages = damages(exact)
    assert len(lucky_damages) == len(exact_damages), "luck never changes which events happen"

    low_pct, high_pct = 100 - LUCK_RANGE, 100 + LUCK_RANGE
    for lucked, plain in zip(lucky_damages, exact_damages):
        assert lucked >= plain * low_pct // 100, f"{lucked} fell below {plain} minus {LUCK_RANGE}%"
        assert lucked <= plain * high_pct // 100 + 1, f"{lucked} rose above {plain} plus {LUCK_RANGE}%"
    assert any(lucked != plain for lucked, plain in zip(lucky_damages, exact_damages)), "the dice actually rolled"


def test_the_dice_are_pure_functions_of_their_context():
    rolls = [roll_percent(seed, "luck", turn, seq) for seed in range(20) for turn in range(20) for seq in range(5)]
    assert all(0 <= roll <= 99 for roll in rolls)
    assert roll_percent(17, "luck", 3, 2) == roll_percent(17, "luck", 3, 2), "the same context rolls the same value"
    assert len(set(rolls)) > 90, "different contexts must not collapse onto the same roll"

    # Two unrelated contexts collide only as often as pure chance allows.
    collisions = sum(roll_percent(s, "luck") == roll_percent(s, "combat") for s in range(400))
    assert collisions < 40, "roll_percent must not be secretly the same value for everything"

    jitters = {
        luck_percent(seed, "luck", turn, seq, "attack", "A1", "B1", "cannon")
        for seed in range(10)
        for turn in range(30)
        for seq in range(5)
    }
    assert jitters == set(range(-LUCK_RANGE, LUCK_RANGE + 1)), "jitter covers its whole band, bounded exactly"


def test_asking_what_each_unit_could_do_changes_nothing_about_the_game():
    plain_run = make_state(**brawl_kwargs())
    curious_run = make_state(**brawl_kwargs())
    for action in brawl_actions():
        if action["action"] != "end_turn":
            # An agent is allowed to enumerate every legal option, with previews,
            # before ordering - that must not consume randomness or mutate state.
            for unit_ in curious_run.units_of(curious_run.current_player):
                rules.unit_options(curious_run, unit_.id)
        do(plain_run, action)
        do(curious_run, action)
    assert state_hash(curious_run) == state_hash(plain_run)
    assert log_hash(curious_run) == log_hash(plain_run)


# ---------------------------------------------------------------------------
# saves, hashes and canonical JSON
# ---------------------------------------------------------------------------


def test_state_dicts_are_plain_json_and_round_trip_exactly():
    state = play_brawl()
    data = state_to_dict(state)
    assert json.loads(json.dumps(data)) == data, "a game state must be pure JSON-able data"

    restored = state_from_dict(data)
    assert state_hash(restored) == state_hash(state)
    assert log_hash(restored) == log_hash(state)

    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert '"a":[1,2]' in canonical_json({"a": (1, 2)}), "key order and separators are canonical, tuples become lists"


def test_a_saved_game_continues_exactly_where_its_original_left_off(tmp_path):
    original = make_state(**brawl_kwargs())
    do(original, attack("A1", "B1"))
    do(original, end_turn())

    path = save_game(original, tmp_path / "saves" / "mid_brawl.json")
    assert json.loads(path.read_text(encoding="utf-8")) == state_to_dict(original)

    resumed = load_game(path)
    for state in (original, resumed):
        do(state, attack("B1", "A1"))
        do(state, end_turn())
    assert state_hash(resumed) == state_hash(original)
    assert log_hash(resumed) == log_hash(original)


def test_the_position_hash_and_the_history_hash_track_different_things():
    state = play_brawl()
    assert state.log, "a played game carries its history inside its state dict"

    # The position hash is exactly the state minus the log: strip the history out
    # of a save dict and the position is byte-identical, while the history hash
    # tracks that the game was played at all.
    data = state_to_dict(state)
    log_free = state_from_dict({**data, "log": []})
    assert state_hash(log_free) == state_hash(state), "the log is the only part the position hash ignores"
    assert log_hash(log_free) != log_hash(state), "the history hash tracks it"

    moved = state_hash(state)
    do(state, move("A2", column_path(3, 1, 1)))  # now the truck actually drives off
    assert state_hash(state) != moved, "and a unit that really moves is a new position"


# ---------------------------------------------------------------------------
# the bundled scenario replays as a whole maneuver game too
# ---------------------------------------------------------------------------


def test_the_bundled_scenario_replays_a_two_turn_maneuver_identically():
    border = bundled_scenario()
    maneuver = [
        move("A2", line(4, 2, 1, step=-1)),  # ridge infantry give way
        end_turn(),
        move("B2", line(9, 2, 1)),
        end_turn(),
        move("A3", column_path(5, 3, 1)),  # the second guards step down off the ridge
        move("A4", column_path(4, 3, 1)),  # and the Tank starts round the ridge by the bay
        end_turn(),
        move("B4", column_path(8, 3, 1)),
        end_turn(),
    ]

    first = replay(border, maneuver)
    again = replay(border, maneuver)
    assert state_hash(first) == state_hash(again)
    assert log_hash(first) == log_hash(again)
    assert first.status == "ongoing" and len(first.units) == 18

    assert at(first, 3, 2).id == "A2" and at(first, 5, 4).id == "A3"
    assert at(first, 4, 4).id == "A4" and at(first, 8, 4).id == "B4", "the maneuver really happened"

    # The maneuver rolls no dice, so its outcome does not depend on the seed at
    # all - but the seed is part of the stored state (it is what any future roll
    # would derive from), so a resseeded replay differs in exactly that one field.
    resseeded = replay(border, maneuver, seed=99999)
    assert log_hash(resseeded) == log_hash(first), "the whole recorded history is identical"
    reseeded_data = state_to_dict(resseeded)
    original_data = state_to_dict(first)
    reseeded_data.pop("seed")
    original_data.pop("seed")
    assert reseeded_data == original_data, "everything except the seed field is identical"
