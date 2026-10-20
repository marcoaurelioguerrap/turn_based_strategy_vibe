"""Supply, fuel and transport: the logistics layer an agent has to plan around.

The rules this file pins down:

* A unit may attack only while it is connected to a supply source: an owned
  city, factory, port, airport or HQ whose domains cover it, or a friendly
  supply vehicle (Truck, APC, Transport Ship) whose domains cover it.  Movement
  and capture never need supply (that half is asserted in ``test_combat``).
* A supplied unit is refuelled and rearmed at the start of your turn; a unit
  cut off keeps whatever it has left.  The very first turn respects exactly what
  the scenario authored.  Air and naval units that end their turn with an empty
  tank *away* from their supply are wrecked; parked beside a base they wait.
  Land units with an empty tank simply stop.
* A transport acts once per turn - it may drive then load, or drive then
  unload, but never two loads - carries only the movement classes its data
  lists, up to its capacity, and its cargo shares its supply situation and
  drowns or burns together with it.

In v1 a Truck is a supply source wherever it stands, even stranded (a deeper
"the truck must be connected to a depot" supply-chain rule is a later
milestone); ``test_a_stranded_supply_vehicle_still_supplies_its_neighbours``
documents that choice deliberately.
"""

from __future__ import annotations

import pytest

from tbs.engine.errors import ActionError
from tbs.engine.rules import (
    _near_supply_source,
    attack_options,
    explain_blockers,
    is_supplied,
    legal_moves,
    load_options,
    unload_options,
)
from tbs.engine.state import OUT_OF_FUEL_DAMAGE
from .helpers import (
    anchor,
    attack,
    at,
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
    unit,
    unload,
)

# Test maps.  All are 10x4; the only non-plains tiles are the ones a test names.
DESERT_ROWS = [
    "..........",
    "..........",
    "..........",
    "..........",
]
LAND_ROWS = ["..........", "..C.......", "..........", ".........."]  # city at column 2
AIRFIELD_ROWS = ["..........", "..A.......", "..........", ".........."]  # airfield at column 2
SHORE_ROWS = ["..........", "..C~......", "..........", ".........."]  # city touching open sea
HARBOUR_ROWS = ["..........", "..p~......", "..........", ".........."]  # harbour touching open sea
OPEN_SEA_ROWS = ["..........", "....~~....", "..........", ".........."]  # open sea at columns 4-5
BAY_ROWS = ["..........", "..~~~~....", "..........", ".........."]  # a bay across row 1

A_CITY = [owned_cells("A", [(2, 1)])]
A_AIRFIELD = [owned_cells("A", [(2, 1)])]
A_HARBOUR = [owned_cells("A", [(2, 1)])]
B_CITY = [owned_cells("B", [(2, 1)])]


# ---------------------------------------------------------------------------
# who supplies whom
# ---------------------------------------------------------------------------


def test_only_your_own_bases_supply_your_units():
    state = make_state(
        rows=LAND_ROWS,
        regions=[owned_cells("B", [(2, 1)])],  # the city belongs to Red
        units=[unit("infantry", "A", 3, 1, "A1"), unit("infantry", "B", 2, 2, "B1")],
    )
    assert is_supplied(state, state.unit("B1")) is True, "it stands next to its own city"
    assert is_supplied(state, state.unit("A1")) is False, "the same city belongs to the enemy"
    assert any("out of supply" in blocker for blocker in explain_blockers(state, "A1"))


def test_a_base_supplies_its_own_tile_and_the_hexes_touching_it():
    units = [
        unit("infantry", "A", 2, 1, "A1"),  # standing on the city
        unit("infantry", "A", 3, 1, "A2"),  # touching it to the east
        unit("infantry", "A", 1, 1, "A3"),  # touching it to the west
        unit("infantry", "A", 2, 2, "A4"),  # touching it to the south
        unit("infantry", "A", 4, 1, "A5"),  # two hexes away, with nothing in between
        anchor("B"),
    ]
    state = make_state(rows=LAND_ROWS, regions=A_CITY, units=units)
    for unit_id in ("A1", "A2", "A3", "A4"):
        assert is_supplied(state, state.unit(unit_id)) is True, f"{unit_id} touches its own city"
    assert is_supplied(state, state.unit("A5")) is False, "supply reaches one hex and no further"


def test_supply_follows_domains_not_just_proximity():
    # An airfield feeds the air; a city feeding the ground feeds nobody in the sky.
    state = make_state(rows=AIRFIELD_ROWS, regions=A_AIRFIELD, units=[unit("helicopter", "A", 3, 1, "A1"), anchor("B")])
    assert is_supplied(state, state.unit("A1")) is True

    state = make_state(rows=LAND_ROWS, regions=A_CITY, units=[unit("helicopter", "A", 3, 1, "A1"), anchor("B")])
    assert is_supplied(state, state.unit("A1")) is False, "a city is not an air base"

    # At sea a harbour is a base and a city on the shoreline is not.
    state = make_state(rows=HARBOUR_ROWS, regions=A_HARBOUR, units=[unit("destroyer", "A", 3, 1, "A1"), anchor("B")])
    assert is_supplied(state, state.unit("A1")) is True, "the port next to it supplies the navy"

    state = make_state(rows=SHORE_ROWS, regions=[owned_cells("A", [(2, 1)])], units=[unit("destroyer", "A", 3, 1, "A1"), anchor("B")])
    assert is_supplied(state, state.unit("A1")) is False, "a city touching the sea is not a harbour"


def test_only_a_supply_vehicle_extends_supply_not_any_friendly_unit():
    # A tank beside the city is supplied, and it supplies nobody: ordinary combat
    # units are not supply sources.
    state = make_state(
        rows=LAND_ROWS,
        regions=A_CITY,
        units=[unit("tank", "A", 3, 1, "A1"), unit("infantry", "A", 4, 1, "A2"), anchor("B")],
    )
    assert is_supplied(state, state.unit("A1")) is True
    assert is_supplied(state, state.unit("A2")) is False, "an ordinary tank does not feed anyone"

    # A Truck in the same hexes does reach the infantry.
    state = make_state(
        rows=LAND_ROWS,
        regions=A_CITY,
        units=[unit("truck", "A", 3, 1, "A1"), unit("infantry", "A", 4, 1, "A2"), anchor("B")],
    )
    assert is_supplied(state, state.unit("A2")) is True, "a Truck is a mobile supply source"


def test_a_stranded_supply_vehicle_still_supplies_its_neighbours():
    # v1 simplification, pinned on purpose: the Truck is a supply source wherever
    # it stands, even with nothing behind it.  Supply *chains* are a later feature.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 4, 1, "A1"), unit("infantry", "A", 5, 1, "A2"), anchor("B")],
    )
    assert _near_supply_source(state, state.unit("A1")) is False, "the Truck itself is on its own in the open"
    assert is_supplied(state, state.unit("A2")) is True, "...and in v1 it still supplies the hex around it"


def test_a_unit_with_no_weapons_is_never_told_to_resupply():
    state = make_state(rows=DESERT_ROWS, units=[unit("truck", "A", 4, 1, "A1"), anchor("B")])
    truck = state.unit("A1")
    assert _near_supply_source(state, truck) is False, "nothing around it is a supply source"
    assert is_supplied(state, truck) is True, "a Truck has no magazines to run dry"
    assert legal_moves(state, "A1"), "and it can still drive"


def test_a_supply_vehicle_only_covers_the_domains_it_is_meant_for():
    # A Truck keeps ground and air units fed, but it cannot feed a ship.
    state = make_state(
        rows=OPEN_SEA_ROWS,
        units=[unit("truck", "A", 3, 1, "A1"), unit("destroyer", "A", 4, 1, "A2"), anchor("B")],
    )
    assert is_supplied(state, state.unit("A2")) is False, "a Truck covers land and air, not the sea"

    # A Transport Ship feeds ships, and not the tanks standing on the shore beside it.
    state = make_state(
        rows=OPEN_SEA_ROWS,
        units=[
            unit("transport_ship", "A", 4, 1, "A1"),
            unit("destroyer", "A", 5, 1, "A2"),
            unit("tank", "A", 3, 1, "A3"),
            anchor("B"),
        ],
    )
    assert is_supplied(state, state.unit("A2")) is True, "the transport next to it covers the navy"
    assert is_supplied(state, state.unit("A3")) is False, "the transport covers naval only"
    # The Transport Ship has no weapons, so orders never nag it (see above); the
    # strict view is that it, too, is stranded out here.
    assert _near_supply_source(state, state.unit("A1")) is False, "it feeds itself no more than anyone else"


# ---------------------------------------------------------------------------
# what a unit is replenished by
# ---------------------------------------------------------------------------


def test_cargo_in_a_supply_vehicle_shares_its_supply():
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 4, 1, "A1"), unit("infantry", "A", 3, 1, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    assert state.unit("A2").loaded is True
    assert is_supplied(state, state.unit("A2")) is True, "riding inside a Truck is like standing next to it"
    assert _near_supply_source(state, state.unit("A2")) is True


def test_cargo_of_a_transport_ship_inherits_the_ship_s_situation():
    # A Transport Ship only promises its own domain; land cargo depends on where
    # the ship itself is connected.  A ship at open water carries a dry landing force.
    stranded = make_state(
        rows=HARBOUR_ROWS,  # harbour at column 2, but nobody owns it
        units=[unit("transport_ship", "A", 3, 1, "A1"), unit("infantry", "A", 3, 2, "A2"), anchor("B")],
    )
    do(stranded, load("A1", "A2"))
    assert is_supplied(stranded, stranded.unit("A2")) is False, "the ship itself is cut off, so its cargo is too"

    # Park the same ship against your own harbour and the landing force is fed.
    at_home = make_state(
        rows=HARBOUR_ROWS,
        regions=A_HARBOUR,
        units=[unit("transport_ship", "A", 3, 1, "A1"), unit("infantry", "A", 3, 2, "A2"), anchor("B")],
    )
    do(at_home, load("A1", "A2"))
    assert is_supplied(at_home, at_home.unit("A2")) is True, "its own harbour feeds the ship and its cargo"


def test_being_in_supply_refuels_and_rearms_at_the_start_of_your_turn():
    state = make_state(
        rows=AIRFIELD_ROWS,
        regions=A_AIRFIELD,
        units=[
            unit("helicopter", "A", 3, 1, "A1", hp=60, fuel=3, ammo={"atgm": 1, "autocannon": 1}),
            anchor("B"),
        ],
    )
    heli = state.unit("A1")
    assert (heli.fuel, heli.hp) == (3, 60), "the first turn respects exactly what the scenario authored"
    assert heli.ammo == {"atgm": 1, "autocannon": 1}

    do(state, end_turn())  # Red's turn
    do(state, end_turn())  # Blue's turn begins: its own airfield re-equip its aircraft
    heli = state.unit("A1")
    assert heli.fuel == 60, "a supplied unit's tanks are topped up at the start of your turn"
    assert heli.ammo == {"atgm": 6, "autocannon": 8}, "and every weapon is rearmed to full"
    assert heli.supplied is True
    assert heli.hp == 60, "supply replenishes equipment; it does not repair the airframe"


def test_nothing_refuels_or_rearms_a_unit_cut_off_from_its_supply():
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("helicopter", "A", 4, 1, "A1", fuel=3, ammo={"atgm": 1}), anchor("B")],
    )
    for _ in range(2):
        do(state, end_turn())
    heli = state.unit("A1")
    assert _near_supply_source(state, heli) is False
    assert heli.fuel == 3, "no base, no refuelling, turn after turn"
    assert heli.ammo["atgm"] == 1, "and nothing rearms a unit that is cut off"


def test_scenarios_can_author_partial_equipment_and_bad_values_are_refused():
    # "This unit has already been fighting" is a thing a scenario must be able to say.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("apc", "A", 4, 1, "A1", hp=30, fuel=10, ammo={"mg": 2}), anchor("B")],
    )
    apc = state.unit("A1")
    assert (apc.hp, apc.fuel, apc.ammo) == (30, 10, {"mg": 2})

    for bad, message in [
        (unit("apc", "A", 4, 1, "A1", hp=0), "hp 0 is not between 1"),
        (unit("apc", "A", 4, 1, "A1", hp=101), "hp 101 is not between 1"),
        (unit("apc", "A", 4, 1, "A1", fuel=41), "fuel 41 is not between 0 and 40"),
        (unit("infantry", "A", 4, 1, "A1", fuel=5), "carries no fuel tank"),
        (unit("apc", "A", 4, 1, "A1", ammo={"rifle": 3}), "has no weapon 'rifle'"),
        (unit("apc", "A", 4, 1, "A1", ammo={"mg": 6}), "mg ammo 6 is not between 0 and 5"),
    ]:
        with pytest.raises(ValueError) as info:
            make_state(rows=DESERT_ROWS, units=[bad, anchor("B")])
        assert message in str(info.value)


def test_a_dry_aircraft_or_ship_far_from_its_supply_is_damaged_and_risky():
    # A dry tank is not yet an accident: a survivable dent first...
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("helicopter", "A", 4, 1, "A1", hp=40, fuel=0), anchor("A"), anchor("B")],
    )
    events = do(state, end_turn())
    assert events_of(events, "out_of_fuel") == [
        {"type": "out_of_fuel", "unit": "A1", "damage": OUT_OF_FUEL_DAMAGE, "hp": 40 - OUT_OF_FUEL_DAMAGE}
    ]
    assert state.unit("A1").hp == 40 - OUT_OF_FUEL_DAMAGE

    # ...and the next dry turn is the one that loses the aircraft.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("helicopter", "A", 4, 1, "A1", hp=OUT_OF_FUEL_DAMAGE, fuel=0), anchor("A"), anchor("B")],
    )
    events = do(state, end_turn())
    wrecked = events_of(events, "destroyed")
    assert [event["unit"] for event in wrecked] == ["A1"]
    assert wrecked[0]["cause"] == "ran out of fuel"
    assert "A1" not in state.units

    # The same rule holds at sea: a ship with nothing alongside runs out of fuel.
    state = make_state(
        rows=OPEN_SEA_ROWS,
        units=[unit("destroyer", "A", 4, 1, "A1", hp=OUT_OF_FUEL_DAMAGE, fuel=0), anchor("A"), anchor("B")],
    )
    events = do(state, end_turn())
    assert events_of(events, "destroyed")[0]["cause"] == "ran out of fuel"
    assert "A1" not in state.units, "the sea has no emergency fuel"


def test_a_dry_craft_parked_beside_its_own_base_only_waits_for_fuel():
    # Advance Wars semantics: running dry next to your airfield is not a crash -
    # the crews top it up, and it flies again next turn.
    state = make_state(
        rows=AIRFIELD_ROWS,
        regions=A_AIRFIELD,
        units=[unit("helicopter", "A", 3, 1, "A1", fuel=0), anchor("B")],
    )
    events = do(state, end_turn())
    assert events_of(events, "out_of_fuel") == [], "no damage when it is near its supply"
    assert state.unit("A1").hp == 100
    do(state, end_turn())  # Red plays, then Blue's turn begins
    assert state.unit("A1").fuel == 60, "and it is refuelled the moment its turn restarts"


def test_land_units_simply_stop_when_their_tank_runs_out():
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("tank", "A", 4, 1, "A1", fuel=0), anchor("B")],
    )
    events: list[dict] = []
    for _ in range(2):
        events.extend(do(state, end_turn()))
    assert events_of(events, "out_of_fuel") == [], "a dry Tank is not an accident at all"
    tank = state.unit("A1")
    assert tank.hp == 100 and tank.fuel == 0, "a dry Land unit is a parked unit, not a casualty"
    assert legal_moves(state, "A1") == [], "it cannot drive again until something refuels it"


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------


def test_a_transport_can_act_only_once_a_turn():
    # Drive up and pick the infantry up: loading is that Truck's whole turn.
    state = make_state(
        rows=DESERT_ROWS,
        units=[
            unit("truck", "A", 4, 1, "A1"),
            unit("infantry", "A", 3, 1, "A2"),
            unit("infantry", "A", 5, 1, "A3"),
            anchor("B"),
        ],
    )
    events = do(state, load("A1", "A2"))
    assert events == [{"type": "load", "transport": "A1", "cargo": "A2", "capacity_used": 1}]

    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(4, 1, 1)))
    assert "already acted this turn" in str(info.value)

    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A3"))
    assert "already acted this turn" in str(info.value), "one transport action per turn, load included"


def test_a_transport_cannot_carry_units_it_is_not_built_for_or_more_than_its_capacity():
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 4, 1, "A1"), unit("tank", "A", 5, 1, "A2"), anchor("B")],
    )
    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A2"))
    assert "carries foot" in str(info.value) and "cannot carry a Tank (tread)" in str(info.value)

    # A Transport Ship lifts ground units; ships and aircraft are not ground units.
    state = make_state(
        rows=OPEN_SEA_ROWS,
        units=[
            unit("transport_ship", "A", 4, 1, "A1"),
            unit("helicopter", "A", 3, 1, "A2"),
            unit("destroyer", "A", 5, 1, "A3"),
            anchor("B"),
        ],
    )
    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A2"))
    assert "cannot carry a Helicopter (air)" in str(info.value)
    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A3"))
    assert "cannot carry a Destroyer" in str(info.value)

    # A Truck fills at one infantry; the next turn it is still full.
    state = make_state(
        rows=DESERT_ROWS,
        units=[
            unit("truck", "A", 4, 1, "A1"),
            unit("infantry", "A", 3, 1, "A2"),
            unit("infantry", "A", 5, 1, "A3"),
            anchor("B"),
        ],
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())
    do(state, end_turn())
    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A3"))
    assert "is full (Truck carries 1)" in str(info.value)

    # An APC's two seats fill over two turns, and only then is it full.
    state = make_state(
        rows=DESERT_ROWS,
        units=[
            unit("apc", "A", 4, 1, "A1"),
            unit("infantry", "A", 3, 1, "A2"),
            unit("infantry", "A", 5, 1, "A3"),
            unit("infantry", "A", 4, 2, "A4"),
            anchor("B"),
        ],
    )
    assert do(state, load("A1", "A2"))[0]["capacity_used"] == 1
    do(state, end_turn())
    do(state, end_turn())
    assert do(state, load("A1", "A3"))[0]["capacity_used"] == 2
    do(state, end_turn())
    do(state, end_turn())
    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A4"))
    assert "is full (APC carries 2)" in str(info.value)

    # Combat units are not transports at all.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("tank", "A", 4, 1, "A1"), unit("infantry", "A", 3, 1, "A2"), anchor("B")],
    )
    with pytest.raises(ActionError) as info:
        do(state, load("A1", "A2"))
    assert "cannot carry units" in str(info.value)


def test_loading_and_unloading_need_the_transport_adjacent_and_the_drop_legal():
    with pytest.raises(ActionError) as info:
        do(
            make_state(
                rows=DESERT_ROWS,
                units=[unit("truck", "A", 4, 1, "A1"), unit("infantry", "A", 2, 1, "A2"), anchor("B")],
            ),
            load("A1", "A2"),
        )
    assert "2 hexes from A1; loading needs the transport to be adjacent" in str(info.value)

    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 4, 1, "A1"), unit("infantry", "A", 3, 1, "A2"), anchor("B")],
    )
    with pytest.raises(ActionError) as info:
        do(state, unload("A1", "A2", cell(4, 2)))
    assert "is not carrying anything" in str(info.value)

    do(state, load("A1", "A2"))
    with pytest.raises(ActionError) as info:
        do(state, end_turn())
        do(state, end_turn())
        do(state, unload("A1", "A2", cell(6, 1)))
    assert "unloading needs an empty hex adjacent to A1" in str(info.value)

    # A drop hex off the map is refused.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 9, 1, "A1"), unit("infantry", "A", 8, 1, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    with pytest.raises(ActionError) as info:
        do(state, end_turn())
        do(state, end_turn())
        do(state, unload("A1", "A2", cell(10, 1)))
    assert "is not part of the map" in str(info.value)

    # A drop hex occupied by a friendly is refused.
    state = make_state(
        rows=DESERT_ROWS,
        units=[
            unit("truck", "A", 4, 1, "A1"),
            unit("infantry", "A", 3, 1, "A2"),
            unit("infantry", "A", 4, 2, "A3"),
            anchor("B"),
        ],
    )
    do(state, load("A1", "A2"))
    with pytest.raises(ActionError) as info:
        do(state, end_turn())
        do(state, end_turn())
        do(state, unload("A1", "A2", cell(4, 2)))
    assert "is occupied by A3" in str(info.value)

    # Open sea is not a drop site, dry shore is.
    state = make_state(
        rows=OPEN_SEA_ROWS,
        units=[unit("truck", "A", 3, 1, "A1"), unit("infantry", "A", 2, 1, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    with pytest.raises(ActionError) as info:
        do(state, end_turn())
        do(state, end_turn())
        do(state, unload("A1", "A2", cell(4, 1)))
    assert "Infantry cannot be dropped on Sea" in str(info.value)

    state = make_state(
        rows=HARBOUR_ROWS,
        units=[unit("transport_ship", "A", 3, 1, "A1"), unit("infantry", "A", 3, 2, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())
    do(state, end_turn())
    do(state, unload("A1", "A2", cell(4, 1)))
    assert at(state, 4, 1).id == "A2", "the beach the Transport Ship's own comment promises"


def test_a_loaded_unit_cannot_be_ordered_and_unloading_it_gives_it_its_turn_back():
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 4, 1, "A1"), unit("infantry", "A", 3, 1, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    assert state.unit("A2").pos is None and state.unit("A1").cargo == ["A2"]

    with pytest.raises(ActionError) as info:
        do(state, move("A2", column_path(3, 1, 1)))
    assert "is loaded inside A1; unload it before it can act" in str(info.value)
    with pytest.raises(ActionError) as info:
        do(state, hold("A2"))
    assert "loaded inside" in str(info.value)

    # A transport only works a turn at a time: the drop happens on the next turn.
    do(state, end_turn())
    do(state, end_turn())
    events = do(state, unload("A1", "A2", cell(4, 2)))
    assert events == [{"type": "unload", "transport": "A1", "cargo": "A2", "at": cell(4, 2)}]
    assert state.unit("A2").loaded is False and state.unit("A2").carried_by is None
    assert at(state, 4, 2).id == "A2"

    do(state, move("A2", column_path(4, 2, 1)))
    assert at(state, 4, 3).id == "A2", "unloaded cargo acts on the same turn it is dropped"


def test_a_transport_may_drive_then_load_or_drive_then_unload():
    # Drive up to the waiting infantry and pick it up: move, then load.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 2, 1, "A1"), unit("infantry", "A", 4, 1, "A2"), anchor("B")],
    )
    do(state, move("A1", line(2, 1, 1)))
    assert at(state, 3, 1).id == "A1"
    do(state, load("A1", "A2"))
    assert state.unit("A2").loaded is True

    # Moving twice in one turn is what is refused, not moving and then working.
    state = make_state(rows=DESERT_ROWS, units=[unit("truck", "A", 2, 1, "A1"), anchor("B")])
    do(state, move("A1", line(2, 1, 1)))
    with pytest.raises(ActionError) as info:
        do(state, move("A1", line(3, 1, 1)))
    assert "has already moved this turn; attack, capture, load, unload or hold instead" in str(info.value)

    # On a later turn a loaded transport drives and then drops its cargo.
    state = make_state(
        rows=DESERT_ROWS,
        units=[unit("truck", "A", 4, 0, "A1"), unit("infantry", "A", 3, 0, "A2"), anchor("B")],
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())
    do(state, end_turn())
    do(state, move("A1", column_path(4, 0, 1)))
    assert at(state, 4, 1).id == "A1"
    do(state, unload("A1", "A2", cell(4, 2)))
    assert at(state, 4, 2).id == "A2", "moving does not spend a transport's working action"


def test_a_ship_lifts_ground_troops_across_water_and_sets_them_down_on_land():
    # Board at one shore, sail along the bay, land at another.
    state = make_state(
        rows=BAY_ROWS,
        units=[unit("transport_ship", "A", 2, 1, "A1"), unit("infantry", "A", 5, 2, "A2"), anchor("B")],
    )
    do(state, move("A1", line(2, 1, 3)))
    assert at(state, 5, 1).id == "A1", "ships cross the sea the way trucks cross roads"
    do(state, load("A1", "A2"))
    assert state.unit("A2").loaded is True, "infantry boards from the shore"

    do(state, end_turn())
    do(state, end_turn())
    do(state, move("A1", line(5, 1, 1, step=-1)))
    do(state, unload("A1", "A2", cell(4, 2)))
    assert at(state, 4, 2).id == "A2", "the landing force comes ashore on open coast"
    do(state, move("A2", column_path(4, 2, 1)))
    assert at(state, 4, 3).id == "A2", "and it walks on the same turn it lands"


def test_cargo_goes_down_with_the_transport():
    # A Truck carrying an infantry is shelled by a tank.
    state = make_state(
        rows=LAND_ROWS,
        regions=B_CITY,
        units=[
            unit("truck", "A", 4, 1, "A1", hp=30),
            unit("infantry", "A", 5, 1, "A2"),
            unit("tank", "B", 3, 1, "B1"),
            anchor("A"),
        ],
    )
    do(state, load("A1", "A2"))
    do(state, end_turn())  # Red's turn, and the Tank is in its own city's supply

    shot = attack_options(state, "B1")[0]
    assert shot["target"] == "A1" and shot["weapon"] == "cannon"
    assert shot["preview_damage"] == 72, "main gun 80 at 90 % against a light"
    events = do(state, attack("B1", "A1"))
    assert events[0]["damage"] == 72, "the preview was exact"

    killed = {event["unit"]: event for event in events_of(events, "destroyed")}
    assert set(killed) == {"A1", "A2"}, "the cargo dies with the transport that carried it"
    assert killed["A2"]["cause"] == "was cargo of A1"
    assert "A1" not in state.units and "A2" not in state.units


# ---------------------------------------------------------------------------
# what the options advertise
# ---------------------------------------------------------------------------


def test_load_and_unload_options_only_offer_legal_choices():
    state = make_state(
        rows=DESERT_ROWS,
        units=[
            unit("truck", "A", 4, 1, "A1"),
            unit("infantry", "A", 3, 1, "A2"),  # adjacent, carriable: the only offer
            unit("tank", "A", 5, 1, "A3"),  # adjacent but not a type a Truck carries
            unit("infantry", "A", 1, 1, "A4"),  # carriable but too far away
            anchor("B"),
        ],
    )
    options = load_options(state, "A1")
    assert options == [{"cargo": "A2", "cargo_type": "infantry", "pos": cell(3, 1)}]

    do(state, load("A1", "A2"))
    assert load_options(state, "A1") == [], "its one seat is taken"
    assert unload_options(state, "A1") == [], "and it has already acted this turn"

    do(state, end_turn())
    do(state, end_turn())
    drops = unload_options(state, "A1")
    assert drops, "the next turn it may land its cargo"
    assert all(option["cargo"] == "A2" for option in drops)
    assert all({"cargo", "at", "terrain"} <= set(option) for option in drops)
    drop_hexes = [option["at"] for option in drops]
    assert cell(4, 2) in drop_hexes, "the hex south of the Truck"
    assert cell(3, 1) in drop_hexes, "the hex it picked the infantry up from is empty again"
    assert cell(5, 1) not in drop_hexes, "not the hex the Tank is standing on"
