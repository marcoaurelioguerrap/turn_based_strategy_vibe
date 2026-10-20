"""Combat: the damage formula, weapon ranges and classes, cover, counterattacks.

Every expected number here comes from the published formula with the seeded luck
switched off (``damage_luck: false``), so the arithmetic can be checked by hand:

    base   = power * multiplier // 100
    base   = base * attacker_hp // 100                  # attacker condition
    base   = base * counter_power_pct // 100            # 50 % for a counterattack
    damage = base * (100 - defender cover) // 100
    damage = max(1, damage)
"""

from __future__ import annotations

import pytest

from tbs.engine.combat import COUNTER_POWER_PCT, compute_damage
from tbs.engine.data import WeaponSpec
from tbs.engine.errors import ActionError
from tbs.engine.rng import LUCK_RANGE
from tbs.engine.rules import attack_options, explain_blockers, is_supplied

from .helpers import anchor, at, attack, do, end_turn, line, make_state, move, owned_cells, tile_at, unit

# A city on column 2 of row 1.  In the scenarios that pass A_CITY, A owns it, so
# a unit on it or next to it is in supply; where A_CITY is omitted the city is
# neutral and supplies nobody.
SUPPLY_ROWS = ["..........", "..C.......", "..........", ".........."]
A_CITY = [owned_cells("A", [(2, 1)])]

FOREST_ROWS = ["..........", "..C.F.....", "..........", ".........."]
CITY_TARGET_ROWS = ["..........", "..CC......", "..........", ".........."]
AIRFIELD_ROWS = ["..........", "..A.......", "..........", ".........."]
SEA_ROWS = ["..........", "..p~~~....", "..........", ".........."]
HQ_ROWS = ["..........", "..C.H.....", "..........", ".........."]
A_PORT = [owned_cells("A", [(2, 1)])]


def engaged(attacker, target, rows=SUPPLY_ROWS, regions=A_CITY, **scenario_kwargs):
    """A state with one A unit at column 3 of row 1 and one B unit next to it."""
    return make_state(rows=rows, regions=regions, units=[attacker, target], **scenario_kwargs)


def a_infantry(unit_id="A1", **extra):
    return unit("infantry", "A", 3, 1, unit_id, **extra)


def b_infantry(unit_id="B1", **extra):
    return unit("infantry", "B", 4, 1, unit_id, **extra)


def test_infantry_fighting_infantry_on_open_ground():
    state = engaged(a_infantry(), b_infantry())
    events = do(state, attack("A1", "B1"))

    hit, counter = events[0], events[1]
    assert hit["type"] == "attack" and hit["weapon"] == "rifle" and hit["distance"] == 1
    assert hit["damage"] == 55, "rifle power 55 at 100 % against an infantry class"
    assert hit["defender_hp"] == 45

    assert counter["type"] == "counterattack" and counter["attacker"] == "B1"
    assert counter["damage"] == 12, "55 x 45 % of the defender's remaining HP, at half power"
    assert state.unit("A1").hp == 88

    assert state.unit("A1").acted is True
    assert state.unit("A1").ammo["rifle"] == 9, "one shot, one ammo"


def test_counterattacks_happen_once_at_half_power():
    assert COUNTER_POWER_PCT == 50
    state = engaged(a_infantry(), b_infantry())
    events = do(state, attack("A1", "B1"))
    counters = [event for event in events if event["type"] == "counterattack"]
    assert len(counters) == 1
    assert counters[0]["distance"] == 1


def test_indirect_fire_is_never_answered():
    state = engaged(unit("artillery", "A", 3, 1, "A1"), unit("tank", "B", 5, 1, "B1"))
    events = do(state, attack("A1", "B1"))
    assert events[0]["distance"] == 2
    assert events[0]["damage"] == 85, "howitzer 90 at 95 % against a heavy"
    assert events[0]["defender_hp"] == 15
    assert [event for event in events if event["type"] == "counterattack"] == []
    assert state.unit("A1").hp == 100, "indirect fire is answered by nothing"

    # a bomber is the same kind of answer-less attack, against infantry
    bomber = make_state(
        rows=AIRFIELD_ROWS,
        regions=[owned_cells("A", [(2, 1)])],
        units=[unit("bomber", "A", 3, 1, "A1"), unit("infantry", "B", 5, 1, "B1")],
    )
    events = do(bomber, attack("A1", "B1"))
    assert events[0]["weapon"] == "bombs" and events[0]["damage"] == 76
    assert [event for event in events if event["type"] == "counterattack"] == []


def test_direct_weapons_cannot_shoot_past_their_range():
    state = engaged(a_infantry(), unit("infantry", "B", 5, 1, "B1"))
    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1"))
    assert "at range 2" in str(info.value)
    assert "rifle reaches 1-1" in str(info.value), "the error explains itself so the agent can retry"
    assert attack_options(state, "A1") == []

    state = engaged(unit("artillery", "A", 3, 1, "A1"), unit("tank", "B", 4, 1, "B1"))
    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1"))
    assert "howitzer reaches 2-4" in str(info.value)
    assert "1 away" in str(info.value)
    assert attack_options(state, "A1") == [], "indirect guns cannot cover the hex beside them"


def test_weapons_that_cannot_hit_a_class_are_not_used():
    state = engaged(a_infantry(), unit("helicopter", "B", 4, 1, "B1"))
    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1"))
    assert "cannot hit air" in str(info.value)
    assert attack_options(state, "A1") == []

    # the anti-air missile cannot touch armour; the autocannon only scratches it
    state = engaged(unit("aa_missile", "A", 3, 1, "A1"), unit("tank", "B", 4, 1, "B1"))
    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1", "sam"))
    assert "sam cannot hit a heavy" in str(info.value)

    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "autocannon", "it falls back to the weapon that works"
    assert events[0]["damage"] == 2, "autocannon 45 at 5 % against a heavy"

    # the same unit kills the aircraft with the missile, from two hexes away
    state = engaged(unit("aa_missile", "A", 3, 1, "A1"), unit("helicopter", "B", 5, 1, "B1"))
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "sam" and events[0]["damage"] == 85
    assert events[0]["defender_hp"] == 15


def test_an_attack_with_no_legal_weapon_lists_every_reason():
    state = engaged(unit("aa_missile", "A", 3, 1, "A1"), unit("tank", "B", 5, 1, "B1"))
    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1"))
    message = str(info.value)
    assert "sam cannot hit heavy" in message
    assert "autocannon reaches 1-1" in message, "and that the autocannon is too short for this target"


def test_cover_from_the_targets_terrain_reduces_damage():
    forest = engaged(a_infantry(), b_infantry(), rows=FOREST_ROWS)
    assert tile_at(forest, 4, 1).terrain == "F"
    events = do(forest, attack("A1", "B1"))
    assert events[0]["damage"] == 41, "55 reduced by 25 % forest cover"
    assert events[0]["defender_hp"] == 59
    assert events[1]["damage"] == 16, "the answer is 55 x 59 % HP at half power, into open ground"

    in_town = make_state(
        rows=CITY_TARGET_ROWS,
        regions=A_CITY,
        units=[unit("infantry", "A", 2, 1, "A1"), unit("infantry", "B", 3, 1, "B1")],
    )
    events = do(in_town, attack("A1", "B1"))
    assert events[0]["damage"] == 33, "55 reduced by 40 % city cover"
    assert events[1]["damage"] == 10, "and the counterattack is reduced by the city the attacker holds"


def test_damage_scales_with_the_attackers_condition():
    state = engaged(a_infantry(hp=40), b_infantry())
    events = do(state, attack("A1", "B1"))
    assert events[0]["damage"] == 22, "a rifle at 40 % of its strength"
    assert events[1]["damage"] == 21, "55 x 78 % of the target's HP, at half power"


def test_damage_never_drops_below_one():
    state = engaged(unit("tank", "A", 3, 1, "A1"), unit("tank", "B", 4, 1, "B1"))
    peek = WeaponSpec(id="peek", name="Peek", power=5, min_range=1, max_range=1, ammo=5, multipliers={"heavy": 1})

    assert compute_damage(state, state.unit("A1"), peek, state.unit("B1"), luck_pct=0) == 1
    assert compute_damage(state, state.unit("A1"), peek, state.unit("B1"), power_pct=1, luck_pct=0) == 1
    assert compute_damage(state, state.unit("A1"), peek, state.unit("B1"), luck_pct=-50) == 1


def test_ammo_is_spent_by_shots_and_resupply_lets_the_gun_keep_firing():
    state = engaged(unit("apc", "A", 3, 1, "A1"), b_infantry())
    total = 0
    for _ in range(5):
        preview = attack_options(state, "A1")[0]
        assert preview["weapon"] == "mg" and preview["ammo_left"] == 5, "rearmed at the start of the turn"

        events = do(state, attack("A1", "B1"))
        assert events[0]["weapon"] == "mg"
        assert events[0]["damage"] == preview["preview_damage"], "the preview was exact"
        total += events[0]["damage"]
        assert state.unit("A1").ammo["mg"] == 4, "one shot, one shell out of the magazine"

        # each answer wounds the APC, so its next shot is weaker than the one before
        assert state.unit("A1").hp < 100
        do(state, end_turn())
        do(state, end_turn())

    assert state.unit("B1").hp == 100 - total, "every point of damage came out of the target"
    assert "A1" in state.units, "the counterattacks did not kill it either"
    assert state.unit("A1").hp < 100, "resupply fills magazines and fuel tanks, it does not repair the hull"


def test_attacks_need_ammo_not_a_supply_line():
    state = engaged(
        unit("tank", "A", 3, 1, "A1"), unit("tank", "B", 4, 1, "B1"), regions=[]
    )
    assert is_supplied(state, state.unit("A1")) is False, "the city next to it is neutral ground"
    assert state.unit("A1").ammo == {"cannon": 3, "mg": 4}, "a tank starts with full magazines"
    assert {option["weapon"] for option in attack_options(state, "A1")} == {"cannon", "mg"}
    events = do(state, attack("A1", "B1"))
    assert events[0]["damage"] == 80, "main gun 80 at 100 % against a heavy"
    assert state.unit("A1").ammo["cannon"] == 2, "one shell spent"

    dry = engaged(unit("tank", "A", 3, 1, "A1"), unit("tank", "B", 4, 1, "B1"), regions=[])
    dry.unit("A1").ammo["cannon"] = 0
    dry.unit("A1").ammo["mg"] = 0
    assert attack_options(dry, "A1") == [], "empty magazines stop the attack"
    assert any("out of ammo" in blocker for blocker in explain_blockers(dry, "A1"))
    with pytest.raises(ActionError) as info:
        do(dry, attack("A1", "B1"))
    assert "out of ammo" in str(info.value)


def test_movement_and_capturing_do_not_need_supply():
    state = engaged(unit("tank", "A", 3, 1, "A1"), unit("tank", "B", 5, 1, "B1"), regions=[])
    assert is_supplied(state, state.unit("A1")) is False
    do(state, move("A1", line(3, 1, 1)))
    assert at(state, 4, 1).id == "A1", "a stranded tank can still drive"

    neutral_city = make_state(
        rows=["..........", ".....C....", "..........", ".........."],
        units=[unit("infantry", "A", 5, 1, "A1"), anchor("B")],
    )
    assert is_supplied(neutral_city, neutral_city.unit("A1")) is False
    events = do(neutral_city, {"action": "capture", "unit": "A1"})
    assert events[0]["type"] == "capture_progress", "capturing is never blocked by supply"


def test_a_supply_vehicle_keeps_its_neighbours_attacking():
    state = make_state(
        rows=SUPPLY_ROWS,
        units=[
            unit("tank", "A", 3, 1, "A1"),
            unit("truck", "A", 2, 1, "A2"),
            unit("apc", "B", 4, 1, "B1"),
        ],
    )
    assert is_supplied(state, state.unit("A1")) is True, "a Truck standing next to it resupplies it"
    events = do(state, attack("A1", "B1"))
    assert events[0]["damage"] == 72, "main gun 80 at 90 % against a light vehicle"


def test_aircraft_rearm_at_airfields_or_a_supply_truck_but_can_always_shoot():
    with_city = engaged(
        unit("helicopter", "A", 3, 1, "A1"), unit("tank", "B", 4, 1, "B1"), regions=A_CITY
    )
    assert is_supplied(with_city, with_city.unit("A1")) is False, "a city supplies land units only"
    assert attack_options(with_city, "A1"), "being cut off from rearming does not stop a shot"

    with_airfield = engaged(
        unit("helicopter", "A", 3, 1, "A1"),
        unit("tank", "B", 4, 1, "B1"),
        rows=AIRFIELD_ROWS,
        regions=[owned_cells("A", [(2, 1)])],
    )
    assert is_supplied(with_airfield, with_airfield.unit("A1")) is True
    events = do(with_airfield, attack("A1", "B1"))
    assert events[0]["weapon"] == "atgm" and events[0]["damage"] == 59
    assert events[0]["defender_hp"] == 41

    with_truck = make_state(
        rows=SUPPLY_ROWS,
        units=[unit("helicopter", "A", 3, 1, "A1"), unit("truck", "A", 2, 1, "A2"), unit("tank", "B", 4, 1, "B1")],
    )
    assert is_supplied(with_truck, with_truck.unit("A1")) is True, "a Truck resupplies aircraft as well"


def test_ships_are_supplied_by_ports_and_hunt_each_other_with_antisub_weapons():
    state = engaged(
        unit("destroyer", "A", 3, 1, "A1"), unit("submarine", "B", 4, 1, "B1"), rows=SEA_ROWS, regions=A_PORT
    )
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "main_gun"
    assert events[0]["damage"] == 7, "deck guns barely scratch a submarine: 75 at 10 %"
    assert events[1]["weapon"] == "torpedoes" and events[1]["damage"] == 41
    assert state.unit("A1").hp == 59

    state = engaged(
        unit("submarine", "A", 3, 1, "A1"), unit("destroyer", "B", 4, 1, "B1"), rows=SEA_ROWS, regions=A_PORT
    )
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "torpedoes" and events[0]["damage"] == 90
    assert events[0]["defender_hp"] == 10
    assert events[1]["damage"] == 1, "a deck gun is 75 at 10 %, which a dying ship cannot answer with"
    assert state.unit("A1").hp == 99


def test_a_destroyers_deck_guns_cannot_shoot_the_shore_but_its_bombardment_can():
    shore = engaged(
        unit("destroyer", "A", 3, 1, "A1"), unit("infantry", "B", 3, 2, "B1"), rows=SEA_ROWS, regions=A_PORT
    )
    with pytest.raises(ActionError) as info:
        do(shore, attack("A1", "B1"))
    assert "main_gun cannot hit infantry" in str(info.value)
    assert "howitzer reaches 2-4" in str(info.value)

    offshore = make_state(
        rows=SEA_ROWS,
        regions=A_PORT,
        units=[unit("destroyer", "A", 3, 1, "A1"), unit("infantry", "B", 3, 3, "B1")],
    )
    events = do(offshore, attack("A1", "B1"))
    assert events[0]["weapon"] == "howitzer" and events[0]["distance"] == 2
    assert events[0]["damage"] == 45
    assert [event for event in events if event["type"] == "counterattack"] == []


def test_a_unit_without_weapons_is_destroyed_without_harming_the_attacker():
    state = engaged(unit("tank", "A", 3, 1, "A1"), unit("truck", "B", 4, 1, "B1", hp=50))
    events = do(state, attack("A1", "B1"))
    assert events[0]["damage"] == 72
    assert events[-1]["type"] == "destroyed" and events[-1]["unit"] == "B1"
    assert "B1" not in state.units
    assert [event for event in events if event["type"] == "counterattack"] == [], "a Truck has no weapons"
    assert state.unit("A1").hp == 100


def test_a_unit_dies_when_its_hp_reaches_zero_and_nothing_counters_out_of_it():
    state = engaged(a_infantry(), b_infantry(hp=40))
    events = do(state, attack("A1", "B1"))
    assert events[0]["damage"] == 55 and events[0]["defender_hp"] == 0
    assert events[1] == {"type": "destroyed", "unit": "B1", "owner": "B", "cause": "attacked by A1"}
    assert len(events) == 2
    assert "B1" not in state.units
    assert state.unit("A1").hp == 100


def test_an_attacker_never_hits_its_own_side():
    state = engaged(a_infantry(), unit("infantry", "A", 4, 1, "A2"))
    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "A2"))
    assert "own side" in str(info.value)
    assert attack_options(state, "A1") == [], "a unit next to a friendly unit has no attack to plan"


def test_only_real_units_and_real_weapons_can_be_ordered():
    state = engaged(a_infantry(), b_infantry())

    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "nope"))
    assert "unknown unit id 'nope'" in str(info.value)
    assert "units in play" in str(info.value), "the error lists what does exist"

    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1", "lasermagic"))
    assert "has no weapon 'lasermagic'" in str(info.value)
    assert "it has: rifle" in str(info.value)

    with pytest.raises(ActionError) as info:
        do(state, attack("nope", "B1"))
    assert "unknown unit id 'nope'" in str(info.value)

    loaded = engaged(a_infantry(), b_infantry())
    loaded.units["B1"].pos = None
    with pytest.raises(ActionError) as info:
        do(loaded, attack("A1", "B1"))
    assert "inside a transport" in str(info.value)


def test_defending_an_hq_is_the_hard_part_of_the_job():
    state = engaged(a_infantry(), unit("hq", "B", 4, 1, "B1"), rows=HQ_ROWS)
    events = do(state, attack("A1", "B1"))
    assert tile_at(state, 4, 1).terrain == "H"
    assert events[0]["damage"] == 6, "rifles do 20 % against a headquarters, which also has 40 % cover"
    assert events[0]["defender_hp"] == 94
    assert events[1]["weapon"] == "mg" and events[1]["damage"] == 5, "its guard MG answers at half power"


def test_previews_predict_exactly_what_happens_when_luck_is_off():
    state = engaged(a_infantry(), b_infantry())
    options = attack_options(state, "A1")
    assert options == [
        {
            "target": "B1",
            "target_type": "infantry",
            "weapon": "rifle",
            "distance": 1,
            "preview_damage": 55,
            "ammo_left": 10,
            "counterattack": {"weapon": "rifle", "preview_damage": 12},
        }
    ], "the counter preview assumes the defender has already been hit"

    events = do(state, attack("A1", "B1"))
    assert events[0]["damage"] == options[0]["preview_damage"]
    assert events[1]["damage"] == options[0]["counterattack"]["preview_damage"]


def test_indirect_attacks_are_advertised_without_a_counterattack():
    state = engaged(unit("artillery", "A", 3, 1, "A1"), unit("tank", "B", 5, 1, "B1"))
    options = attack_options(state, "A1")
    assert [option["weapon"] for option in options] == ["howitzer"]
    assert options[0]["distance"] == 2
    assert options[0]["preview_damage"] == 85
    assert "counterattack" not in options[0]
    assert options[0]["ammo_left"] == 6


def test_luck_jitters_damage_only_within_the_published_band():
    damages = set()
    for seed in (1, 2, 3, 4, 5):
        state = engaged(a_infantry(), b_infantry(), damage_luck=True, seed=seed)
        preview = attack_options(state, "A1")[0]["preview_damage"]
        events = do(state, attack("A1", "B1"))
        damage = events[0]["damage"]
        assert preview * (100 - LUCK_RANGE) // 100 <= damage <= preview * (100 + LUCK_RANGE) // 100
        damages.add(damage)

    expected = {
        preview * (100 + jitter) // 100 for jitter in range(-LUCK_RANGE, LUCK_RANGE + 1)
    }
    assert damages <= expected, "every outcome is one of the published jitter values"


def test_a_weapon_can_be_chosen_explicitly_when_several_would_work():
    state = engaged(unit("tank", "A", 3, 1, "A1"), unit("infantry", "B", 4, 1, "B1"))
    options = attack_options(state, "A1")
    assert {option["weapon"] for option in options} == {"cannon", "mg"}
    assert all(option["target"] == "B1" for option in options)
    assert all(option["counterattack"]["weapon"] == "rifle" for option in options)

    # by default the engine uses the first weapon that works, in the unit's own order
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "cannon" and events[0]["damage"] == 48, "80 at 60 % against an infantry"

    support = engaged(unit("tank", "A", 3, 1, "A1"), unit("infantry", "B", 4, 1, "B1"))
    events = do(support, attack("A1", "B1", "mg"))
    assert events[0]["weapon"] == "mg" and events[0]["damage"] == 20, "25 at 80 %"
    assert support.unit("A1").ammo["cannon"] == 3, "the weapon that was not used kept its ammo"


# ---------------------------------------------------------------------------
# movement and indirect fire
# ---------------------------------------------------------------------------


def test_indirect_weapons_cannot_fire_after_moving():
    state = make_state(
        regions=[owned_cells("A", [(2, 1), (3, 1), (4, 1)])],
        units=[unit("artillery", "A", 3, 1, "A1"), unit("tank", "B", 6, 1, "B1"), anchor()],
    )
    assert [option["weapon"] for option in attack_options(state, "A1")] == ["howitzer"]

    do(state, move("A1", line(3, 1, 1, -1)))  # col 2: still in range, still supplied
    assert attack_options(state, "A1") == [], "the howitzer must disappear once the unit has moved"

    with pytest.raises(ActionError) as info:
        do(state, attack("A1", "B1"))
    message = str(info.value)
    assert "moved this turn" in message and "indirect" in message


def test_direct_weapons_still_fire_after_moving():
    state = make_state(
        regions=[owned_cells("A", [(2, 1)])],
        units=[unit("infantry", "A", 2, 1, "A1"), unit("infantry", "B", 4, 1, "B1"), anchor()],
    )
    do(state, move("A1", line(2, 1, 1)))  # col 3, still adjacent to the owned city
    options = attack_options(state, "A1")
    assert [option["weapon"] for option in options] == ["rifle"]
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "rifle"


def test_a_moved_tank_fires_even_when_cut_off_from_supply():
    state = make_state(
        units=[unit("tank", "A", 3, 1, "A1"), unit("infantry", "B", 5, 1, "B1"), anchor()],
    )
    do(state, move("A1", line(3, 1, 1)))  # col 4, adjacent to B1 and far from any base
    assert is_supplied(state, state.unit("A1")) is False
    assert {option["weapon"] for option in attack_options(state, "A1")} == {"cannon", "mg"}
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "cannon" and events[0]["damage"] == 48


def test_a_moved_destroyer_keeps_its_deck_gun_but_loses_shore_bombardment():
    rows = ["..........", "...p~~....", "..........", ".........."]
    state = make_state(
        rows=rows,
        regions=[owned_cells("A", [(3, 1)])],
        units=[
            unit("destroyer", "A", 3, 1, "A1"),
            unit("destroyer", "B", 5, 1, "B1"),
            anchor(),
        ],
    )
    do(state, move("A1", line(3, 1, 1)))  # col 4, next to both the port and B1
    options = attack_options(state, "A1")
    assert [option["weapon"] for option in options] == ["main_gun"]
    assert options[0]["distance"] == 1
    events = do(state, attack("A1", "B1"))
    assert events[0]["weapon"] == "main_gun"


def test_a_moved_unit_is_told_why_its_indirect_weapon_is_unavailable():
    state = make_state(
        regions=[owned_cells("A", [(2, 1), (3, 1), (4, 1)])],
        units=[unit("artillery", "A", 3, 1, "A1"), unit("tank", "B", 6, 1, "B1"), anchor()],
    )
    do(state, move("A1", line(3, 1, 1, -1)))
    blockers = explain_blockers(state, "A1")
    assert any("indirect weapons cannot fire after moving" in blocker for blocker in blockers)
