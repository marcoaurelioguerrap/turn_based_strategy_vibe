"""Capturing sites, the income they bring, and what wins a game.

The rules this file pins down:

* Only Infantry capture, and only the site its own hex is standing on.  Each
  capture attempt adds the unit's current HP to the site's progress (so a
  city needs two turns of a fully fit infantry, and a wounded one takes
  longer).  The progress belongs to that one unit on that one site: it waits
  there patiently through the enemy's turns, but it goes back to zero when
  the capturer steps off, dies, or a different unit takes over the job.
* Owning cities, factories, ports and airfields pays income at the start of
  your turn; production spends that money at a factory, port or airport you
  own and that nothing is standing on.  New units cannot act until your next
  turn.
* Capturing an enemy HQ - or a neutral one - ends the game.  So does wiping
  out an enemy who has no way to produce more; a player with a factory and
  funds is never out of the war on units alone.  Passing the scenario's turn
  limit ends it as a draw.
"""

from __future__ import annotations

import pytest

from tbs.engine.errors import ActionError
from tbs.engine.rules import build_options, turn_income
from .helpers import (
    anchor,
    at,
    attack,
    build,
    capture,
    cell,
    column_path,
    do,
    end_turn,
    events_of,
    hold,
    line,
    load,
    make_state,
    move,
    owned_cells,
    tile_at,
    unit,
)

CITY_ROWS = ["..........", "..C.......", "..........", ".........."]  # neutral city at column 2
FACTORY_ROWS = ["..........", "..f.......", "..........", ".........."]
AIRPORT_ROWS = ["..........", "..A.......", "..........", ".........."]
PORT_ROWS = ["..........", "..p.......", "..........", ".........."]
HQ_ROWS = ["..........", "........H.", "..........", ".........."]  # HQ at column 8, row 1
OPEN_GROUND_ROWS = ["..........", "..........", "..........", ".........."]

A_CITY = [owned_cells("A", [(2, 1)])]
B_CITY = [owned_cells("B", [(2, 1)])]
A_FACTORY = [owned_cells("A", [(2, 1)])]
A_HQ = [owned_cells("A", [(2, 1)])]


def turn_start_of(events: list[dict], player_id: str) -> dict:
    """The turn-start event a finished turn handed to ``player_id``."""
    starts = events_of(events, "turn_start")
    assert len(starts) == 1 and starts[0]["player"] == player_id
    return starts[0]


# ---------------------------------------------------------------------------
# capturing sites
# ---------------------------------------------------------------------------


def test_capturing_a_neutral_city_is_two_turns_of_a_fully_fit_infantry():
    state = make_state(
        rows=CITY_ROWS,
        units=[unit("infantry", "A", 2, 1, "A1"), anchor("B")],  # standing on the city
    )

    events = do(state, capture("A1"))
    assert events_of(events, "captured") == [], "one turn of work is not ownership yet"
    progress = events_of(events, "capture_progress")[0]
    assert progress["unit"] == "A1" and progress["site"] == cell(2, 1)
    assert progress["progress"] == 100 and progress["needed"] == 200, "a capture pays the unit's HP"

    city = tile_at(state, 2, 1)
    assert city.capture_unit == "A1" and city.capture_progress == 100

    # The work waits on the site through the enemy's turn, untouched.
    events = do(state, end_turn())
    assert turn_start_of(events, "B")["capture_progress_reset"] == []
    do(state, end_turn())
    assert city.capture_progress == 100, "the capturer is still standing there, so its work stands"

    events = do(state, capture("A1"))
    assert events_of(events, "capture_progress")[0]["progress"] == 200
    assert events_of(events, "captured") == [
        {
            "type": "captured",
            "unit": "A1",
            "site": cell(2, 1),
            "terrain": "City",
            "previous_owner": None,
            "new_owner": "A",
        }
    ]
    assert city.owner == "A"
    assert city.capture_progress == 0 and city.capture_unit is None, "the site starts clean for its new owner"
    assert state.status == "ongoing", "a city is not an objective"


def test_capture_work_resets_when_the_capturer_steps_off_the_site():
    state = make_state(
        rows=CITY_ROWS,
        units=[unit("infantry", "A", 2, 1, "A1"), anchor("B")],
    )
    do(state, capture("A1"))
    do(state, end_turn())  # Red's turn; the work on the city is untouched
    do(state, end_turn())

    do(state, move("A1", line(2, 1, 1)))  # walks away before finishing
    events = do(state, end_turn())
    assert turn_start_of(events, "B")["capture_progress_reset"] == [cell(2, 1)]
    city = tile_at(state, 2, 1)
    assert city.capture_progress == 0 and city.capture_unit is None, "half-done work is abandoned, not saved"

    do(state, end_turn())
    do(state, move("A1", line(3, 1, 1, step=-1)))  # comes back, and starts over
    events = do(state, capture("A1"))
    assert events_of(events, "capture_progress")[0]["progress"] == 100, "not 200: the reset was real"
    assert events_of(events, "captured") == []


def test_a_different_unit_takes_the_capture_job_back_to_zero():
    state = make_state(
        rows=CITY_ROWS,
        units=[
            unit("infantry", "A", 2, 1, "A1"),  # on the city
            unit("infantry", "A", 4, 1, "A2"),  # waiting behind
            anchor("B"),
        ],
    )
    do(state, capture("A1"))
    do(state, end_turn())
    do(state, end_turn())

    do(state, move("A1", column_path(2, 1, 1)))  # step aside
    do(state, move("A2", [cell(4, 1), cell(3, 1), cell(2, 1)]))  # climb onto the city
    events = do(state, capture("A2"))
    progress = events_of(events, "capture_progress")[0]
    assert progress["unit"] == "A2" and progress["progress"] == 100
    assert events_of(events, "captured") == []
    city = tile_at(state, 2, 1)
    assert city.capture_unit == "A2", "the site now works for the second infantry's progress, not the first's"


def test_wounded_infantry_capture_at_half_speed():
    state = make_state(
        rows=CITY_ROWS,
        units=[unit("infantry", "A", 2, 1, "A1", hp=50), anchor("B")],
    )
    events = do(state, capture("A1"))
    assert events_of(events, "capture_progress")[0]["progress"] == 50, "capture work is paid for in HP"

    do(state, end_turn())
    do(state, end_turn())
    events = do(state, capture("A1"))
    assert events_of(events, "capture_progress")[0]["progress"] == 100
    assert tile_at(state, 2, 1).owner is None, "it will be a slow afternoon for this squad"


def test_capture_is_something_only_infantry_do_while_standing_on_the_site():
    # A Tank on the city cannot capture.
    with pytest.raises(ActionError) as info:
        do(
            make_state(rows=CITY_ROWS, units=[unit("tank", "A", 2, 1, "A1"), anchor("B")]),
            capture("A1"),
        )
    assert "Tank A1 cannot capture sites; only Infantry can" in str(info.value)

    # An infantry on ordinary ground has no site under it.
    with pytest.raises(ActionError) as info:
        do(
            make_state(rows=CITY_ROWS, units=[unit("infantry", "A", 3, 1, "A1"), anchor("B")]),
            capture("A1"),
        )
    assert "Plains at" in str(info.value) and "cannot be captured" in str(info.value)

    # Its own city is not "capturing", it is already home.
    with pytest.raises(ActionError) as info:
        do(
            make_state(rows=CITY_ROWS, regions=A_CITY, units=[unit("infantry", "A", 2, 1, "A1"), anchor("B")]),
            capture("A1"),
        )
    assert "is already yours" in str(info.value)

    # A cargo cannot capture anything.
    state = make_state(
        rows=OPEN_GROUND_ROWS,
        units=[unit("truck", "A", 4, 1, "A1"), unit("infantry", "A", 3, 1, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    with pytest.raises(ActionError) as info:
        do(state, capture("A2"))
    assert "is loaded inside A1" in str(info.value)


# ---------------------------------------------------------------------------
# the HQ: the site that ends the war
# ---------------------------------------------------------------------------


def test_capturing_the_enemy_hq_wins_the_game():
    state = make_state(
        rows=HQ_ROWS,
        regions=[owned_cells("B", [(8, 1)])],  # Red owns the HQ
        units=[unit("infantry", "A", 7, 1, "A1"), anchor("B")],  # Blue's infantry at the gate
    )
    hq = tile_at(state, 8, 1)

    do(state, move("A1", line(7, 1, 1)))  # steps onto the HQ hex
    do(state, capture("A1"))
    do(state, end_turn())
    assert hq.capture_progress == 100, "the enemy's own turn does not cancel work on its ground"
    do(state, end_turn())

    do(state, capture("A1"))
    assert events_of(do(state, end_turn()), "captured") == []
    assert hq.capture_progress == 200
    do(state, end_turn())

    events = do(state, capture("A1"))
    assert events_of(events, "captured") == [
        {
            "type": "captured",
            "unit": "A1",
            "site": cell(8, 1),
            "terrain": "Headquarters",
            "previous_owner": "B",
            "new_owner": "A",
        }
    ]
    assert events_of(events, "objective_captured") == [
        {"type": "objective_captured", "site": cell(8, 1), "winner": "A"}
    ]
    assert state.status == "victory" and state.winner == "A"

    with pytest.raises(ActionError) as info:
        do(state, end_turn())
    assert str(info.value) == "the game is already victory (winner: A)"


def test_a_neutral_hq_is_an_objective_for_whichever_side_gets_there_first():
    state = make_state(rows=HQ_ROWS, units=[unit("infantry", "A", 7, 1, "A1"), anchor("B")])  # nobody owns the HQ
    do(state, move("A1", line(7, 1, 1)))
    do(state, capture("A1"))
    do(state, end_turn())
    do(state, end_turn())
    do(state, capture("A1"))
    do(state, end_turn())
    do(state, end_turn())
    events = do(state, capture("A1"))
    assert events_of(events, "captured") == [
        {
            "type": "captured",
            "unit": "A1",
            "site": cell(8, 1),
            "terrain": "Headquarters",
            "previous_owner": None,
            "new_owner": "A",
        }
    ]
    assert events_of(events, "objective_captured") == [
        {"type": "objective_captured", "site": cell(8, 1), "winner": "A"}
    ]
    assert state.status == "victory" and state.winner == "A", "an unowned HQ still ends the war when taken"


# ---------------------------------------------------------------------------
# ownership, income, production
# ---------------------------------------------------------------------------


def test_your_bases_pay_you_at_the_start_of_your_turn():
    rows = ["..........", "..C.f.....", "..........", ".........."]  # city at 2, factory at 4
    state = make_state(
        rows=rows,
        regions=[owned_cells("A", [(2, 1), (4, 1)])],
        units=[anchor("B")],
    )
    assert turn_income(state, "A") == 300 + 500, "initial income plus every owned site"
    assert turn_income(state, "B") == 0, "owning nothing pays nothing"
    assert state.players["A"].funds == 20000 + 800, "start_game already paid turn 1"
    assert state.players["B"].funds == 20000

    do(state, end_turn())
    events = do(state, end_turn())
    start = turn_start_of(events, "A")
    assert start["income"] == 800 and start["funds"] == 20000 + 1600, "every turn, same rule"

    # The scenario's initial income is part of the same accounting.
    richer = make_state(rows=rows, regions=[owned_cells("A", [(2, 1), (4, 1)])], units=[anchor("B")], initial_income=1500)
    assert turn_income(richer, "A") == 1500 + 800
    assert richer.players["A"].funds == 20000 + 2300


def test_losing_a_city_moves_its_income_across_the_front_line():
    state = make_state(
        rows=CITY_ROWS,
        regions=A_CITY,
        units=[unit("infantry", "B", 2, 1, "B1"), anchor("A")],  # Red's infantry stands on Blue's city
    )
    city = tile_at(state, 2, 1)

    do(state, end_turn())  # Blue idles; Red begins its turn
    do(state, capture("B1"))  # a hundred of the two hundred it needs
    events = do(state, end_turn())  # Red's turn ends: Blue still owns the city, so Blue is paid
    assert turn_start_of(events, "A")["income"] == 300, "Blue was still paid while it held the city"
    assert city.capture_progress == 100, "Blue's turn did not cancel Red's work on Blue's ground"

    do(state, end_turn())  # Blue idle; Red's infantry finishes the job
    events = do(state, capture("B1"))
    assert events_of(events, "captured")[0]["previous_owner"] == "A"
    assert city.owner == "B" and state.status == "ongoing"

    events = do(state, end_turn())  # Red's turn ends: Blue begins with empty till
    assert turn_start_of(events, "A")["income"] == 0
    events = do(state, end_turn())
    start = turn_start_of(events, "B")
    assert start["income"] == 300, "and Red is paid for it instead"


def test_production_costs_money_and_needs_a_free_site_of_the_right_kind():
    state = make_state(rows=FACTORY_ROWS, regions=A_FACTORY, units=[anchor("B")])
    funds_before = state.players["A"].funds  # 20000 plus the factory's own income
    assert funds_before == 20500

    events = do(state, build(2, 1, "infantry"))
    assert events == [
        {"type": "build", "unit": "A1", "site": cell(2, 1), "cost": 700, "funds_left": funds_before - 700}
    ]
    fresh = state.unit("A1")
    assert at(state, 2, 1).id == "A1" and fresh.deployed_this_turn is True

    with pytest.raises(ActionError) as info:
        do(state, move("A1", column_path(2, 1, 1)))
    assert "was produced this turn and cannot act until your next turn" in str(info.value)
    with pytest.raises(ActionError) as info:
        do(state, capture("A1"))
    assert "was produced this turn" in str(info.value)

    do(state, end_turn())
    do(state, end_turn())
    do(state, move("A1", column_path(2, 1, 1)))  # next turn it walks off its own factory
    assert at(state, 2, 2).id == "A1"

    # Ports build ships, and a Tank is not one.
    state = make_state(rows=PORT_ROWS, regions=[owned_cells("A", [(2, 1)])], units=[anchor("B")])
    funds_before = state.players["A"].funds
    events = do(state, build(2, 1, "transport_ship"))
    assert events[0]["type"] == "build" and events[0]["cost"] == 1200
    assert events[0]["funds_left"] == funds_before - 1200
    assert at(state, 2, 1).type_id == "transport_ship", "a harbour builds the navy"


def test_production_refuses_the_wrong_site_and_every_kind_of_impossible_order():
    bad_orders = [
        # not a build site at all
        (dict(rows=CITY_ROWS, regions=A_CITY, units=[anchor("B")]), build(2, 1, "infantry"),
         "is not a build site (factory, port or airport)"),
        # someone else's factory - nobody's, or the enemy's
        (dict(rows=FACTORY_ROWS, units=[anchor("B")]), build(2, 1, "infantry"),
         "you do not own the Factory at"),
        (dict(rows=FACTORY_ROWS, regions=[owned_cells("B", [(2, 1)])], units=[anchor("B")]), build(2, 1, "infantry"),
         "(owner: B)"),
        # a unit is standing on the site
        (dict(rows=FACTORY_ROWS, regions=A_FACTORY, units=[unit("infantry", "A", 2, 1, "A1"), anchor("B")]),
         build(2, 1, "infantry"), "is occupied by A1; production needs an empty build site"),
        # wrong kind of unit for the site
        (dict(rows=FACTORY_ROWS, regions=A_FACTORY, units=[anchor("B")]), build(2, 1, "helicopter"),
         "Helicopter can only be produced at: airport (this site is factory)"),
        (dict(rows=AIRPORT_ROWS, regions=[owned_cells("A", [(2, 1)])], units=[anchor("B")]), build(2, 1, "tank"),
         "Tank can only be produced at: factory (this site is airport)"),
        (dict(rows=PORT_ROWS, regions=[owned_cells("A", [(2, 1)])], units=[anchor("B")]), build(2, 1, "infantry"),
         "Infantry can only be produced at: factory (this site is port)"),
    ]
    for scenario_kwargs, action, message in bad_orders:
        with pytest.raises(ActionError) as info:
            do(make_state(**scenario_kwargs), action)
        assert message in str(info.value)

    # Off the map.
    with pytest.raises(ActionError) as info:
        do(make_state(rows=FACTORY_ROWS, regions=A_FACTORY, units=[anchor("B")]), build(15, 9, "infantry"))
    assert "is not part of the map" in str(info.value)

    # Out of money: a factory's own income is never enough for a Tank.
    poor = make_state(
        rows=FACTORY_ROWS,
        regions=A_FACTORY,
        units=[anchor("B")],
        players=[{"id": "A", "name": "Alpha", "funds": 0}, {"id": "B", "name": "Bravo", "funds": 0}],
    )
    assert poor.players["A"].funds == 500  # the factory paid its owner, still not enough
    with pytest.raises(ActionError) as info:
        do(poor, build(2, 1, "tank"))
    assert str(info.value) == "Tank costs 1200 but you only have 500"


# ---------------------------------------------------------------------------
# winning, surviving, and running out of turns
# ---------------------------------------------------------------------------


def test_wiping_out_an_enemy_that_cannot_rebuild_wins_the_game():
    state = make_state(
        rows=CITY_ROWS,
        regions=A_CITY,
        units=[unit("infantry", "A", 3, 1, "A1"), unit("infantry", "B", 4, 1, "B1", hp=1)],
        # Red has money but no factory and no other soldier: this map has neither.
    )
    events = do(state, attack("A1", "B1"))
    assert "B1" not in state.units

    assert state.players["B"].eliminated is True, "funds in the bank cannot buy soldiers without a build site"
    assert state.status == "victory" and state.winner == "A"
    assert any("no units left and no way to produce more" in entry["describe"] for entry in state.log)

    with pytest.raises(ActionError) as info:
        do(state, hold("A1"))
    assert str(info.value) == "the game is already victory (winner: A)"


def test_a_player_with_a_factory_and_funds_is_never_out_of_the_war_on_units_alone():
    rows = ["..........", ".f......C.", "..........", ".........."]  # Blue factory at 1, Red city at 8
    state = make_state(
        rows=rows,
        regions=[owned_cells("A", [(1, 1)]), owned_cells("B", [(8, 1)])],
        units=[unit("infantry", "A", 6, 1, "A1", hp=1), unit("tank", "B", 7, 1, "B1")],
    )
    do(state, end_turn())  # Blue's turn ends; the soldier stands no arms factory nearby
    do(state, attack("B1", "A1"))  # the Tank shells it
    assert "A1" not in state.units
    assert state.players["A"].eliminated is False, "Blue can still man the factory"
    assert state.status == "ongoing"

    do(state, end_turn())
    events = do(state, build(1, 1, "infantry"))  # and does, next turn
    assert events[0]["type"] == "build"
    assert state.status == "ongoing"


def test_the_team_unit_cap_stops_production_and_oversized_starting_armies():
    state = make_state(
        rows=FACTORY_ROWS,
        regions=A_FACTORY,
        units=[unit("infantry", "A", 4, 1, "A1"), anchor("B")],
        max_units_per_player=1,
    )
    assert build_options(state, "A") == [], "a full team is offered no production"

    with pytest.raises(ActionError) as info:
        do(state, build(2, 1, "infantry"))
    assert "caps each team at 1" in str(info.value)

    with pytest.raises(ValueError) as info:
        make_state(
            rows=OPEN_GROUND_ROWS,
            units=[unit("infantry", "A", 0, 0, "A1"), unit("infantry", "A", 1, 0, "A2"), anchor("B")],
            max_units_per_player=1,
        )
    assert "max_units_per_player is 1" in str(info.value)


def test_running_past_the_scenario_turn_limit_ends_the_game_as_a_draw():
    state = make_state(rows=CITY_ROWS, units=[unit("infantry", "A", 4, 1, "A1"), anchor("B")], max_turns=1)
    do(state, end_turn())  # Red's turn
    assert state.status == "ongoing"
    do(state, end_turn())  # back to Blue: that is round 2 of a 1-round scenario
    assert state.status == "draw" and state.winner is None

    with pytest.raises(ActionError) as info:
        do(state, end_turn())
    assert str(info.value) == "the game is already draw"
