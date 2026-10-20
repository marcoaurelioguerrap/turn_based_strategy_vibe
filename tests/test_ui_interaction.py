"""The click resolver, played against the real engine.

Every order the resolver produces here is handed straight to
``rules.perform``; the tests therefore prove the UI can only issue legal
orders, and that a UI-played game still replays from its order log.
"""

from __future__ import annotations

from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.hexgrid import axial_to_offset, hex_distance
from tbs.engine.serialize import replay, state_hash
from tbs.engine.state import new_game
from tbs.ui.interaction import Controller

from .helpers import cell, make_state, owned_cells
from .helpers import unit as placement


def fresh_state():
    scenario = load_scenario(available_scenarios()[0])
    state = new_game(scenario, load_specs())
    rules.start_game(state)
    return scenario, state


def first_attacker(state, player: str):
    for unit in state.units_of(player):
        attacks = rules.attack_options(state, unit.id)
        if attacks:
            return unit, attacks[0]
    raise AssertionError("the opening position offered no attack")


# ---------------------------------------------------------------------------
# selecting and moving
# ---------------------------------------------------------------------------


def test_clicking_an_own_unit_selects_it():
    _, state = fresh_state()
    controller = Controller()
    unit = state.unit("A2")
    intent = controller.click(state, unit.pos)
    assert intent.kind == "select"
    assert controller.selected == "A2"


def test_clicking_empty_ground_with_nothing_selected_does_nothing():
    _, state = fresh_state()
    controller = Controller()
    empty = next(pos for pos, tile in state.map.tiles.items() if state.unit_at(pos) is None and tile.owner is None)
    intent = controller.click(state, empty)
    assert intent.kind == "noop"
    assert controller.selected is None


def test_a_highlighted_destination_becomes_a_legal_move():
    _, state = fresh_state()
    controller = Controller()
    unit = state.unit("A2")
    controller.click(state, unit.pos)
    destination = rules.legal_moves(state, "A2")[0]["pos"]

    intent = controller.click(state, tuple(destination))
    assert intent.kind == "order"
    assert intent.order["action"] == "move"
    events = rules.perform(state, intent.order)
    assert events, "a legal move must produce events"
    assert state.unit("A2").pos == tuple(destination)
    assert controller.selected is None


def test_clicking_an_in_range_enemy_becomes_an_attack():
    _, state = fresh_state()
    controller = Controller()
    attacker, option = first_attacker(state, "A")
    controller.click(state, attacker.pos)
    target = state.units[option["target"]]
    hp_before = target.hp

    intent = controller.click(state, target.pos)
    assert intent.kind == "order"
    assert intent.order["action"] == "attack"
    assert intent.order["target"] == target.id
    rules.perform(state, intent.order)
    assert state.units[target.id].hp < hp_before


def test_an_illegal_click_leaves_the_game_untouched():
    scenario, state = fresh_state()
    controller = Controller()
    before = state_hash(state)
    unit = state.unit("A5")  # artillery: cannot shoot an adjacent hex
    controller.click(state, unit.pos)
    adjacent = (unit.pos[0] + 1, unit.pos[1])
    if adjacent in state.map.tiles and state.unit_at(adjacent) is None:
        intent = controller.click(state, adjacent)
        assert intent.kind in ("noop", "order")
        if intent.kind == "noop":
            assert state_hash(state) == before


# ---------------------------------------------------------------------------
# capture, hold, build, cycle
# ---------------------------------------------------------------------------


def test_hold_is_offered_for_a_selected_unit():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A1").pos)
    order = controller.hold_order(state)
    assert order == {"action": "hold", "unit": "A1"}
    assert rules.perform(state, order)[0]["type"] == "hold"


def test_clicking_an_empty_build_site_opens_the_menu_and_builds():
    _, state = fresh_state()
    controller = Controller()
    site = next(site.pos for site in state.build_sites("A") if state.unit_at(site.pos) is None)

    intent = controller.click(state, site)
    assert intent.kind == "build_menu"
    assert controller.build_site == site

    options = controller.build_options_at(state, site)
    assert options, "an owned factory must offer something to build"
    affordable = next(option for option in options if option["affordable"])
    order = controller.build_order(state, site, affordable["unit"])
    events = rules.perform(state, order)
    assert events[0]["type"] == "build"
    assert controller.build_site is None


def test_cycle_selects_only_units_that_can_still_act():
    _, state = fresh_state()
    controller = Controller()
    selected = controller.cycle_ready(state)
    assert selected is not None
    assert not state.units[selected].acted and not state.units[selected].deployed_this_turn


# ---------------------------------------------------------------------------
# the action menu
# ---------------------------------------------------------------------------


def menu_index_of(controller, state, key: str) -> int:
    for index, item in enumerate(controller.action_items(state)):
        if item.key == key:
            return index
    raise AssertionError(f"no {key!r} item in the action menu")


def test_the_action_menu_offers_move_attack_status_and_hold():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A2").pos)  # infantry, nothing in range
    keys = [item.key for item in controller.action_items(state)]
    assert keys[:2] == ["move", "attack"]
    assert "status" in keys and "hold" in keys and "cancel" in keys
    attack = controller.action_items(state)[menu_index_of(controller, state, "attack")]
    assert attack.enabled is False, "attack must be shown but disabled when nothing is in range"
    assert "0" in attack.label


def test_choosing_move_from_the_menu_then_a_destination_commits():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A2").pos)

    controller.menu_index = menu_index_of(controller, state, "move")
    controller.menu_choose(state)
    assert controller.mode == "move"

    # a click that is not a reachable hex returns to the menu without acting
    far = next(
        pos
        for pos in state.map.tiles
        if pos not in controller.move_region(state) and state.unit_at(pos) is None
    )
    assert controller.click(state, far).kind == "noop"
    assert controller.mode == "menu"

    destination = rules.legal_moves(state, "A2")[0]["pos"]
    intent = controller.click(state, tuple(destination))
    assert intent.kind == "order" and intent.order["action"] == "move"
    rules.perform(state, intent.order)
    assert state.unit("A2").pos == tuple(destination)


def test_choosing_attack_from_the_menu_then_a_target_commits():
    _, state = fresh_state()
    controller = Controller()
    attacker, option = first_attacker(state, "A")
    controller.click(state, attacker.pos)
    controller.menu_index = menu_index_of(controller, state, "attack")
    controller.menu_choose(state)
    assert controller.mode == "attack"

    target = state.units[option["target"]]
    intent = controller.click(state, target.pos)
    assert intent.kind == "order" and intent.order["action"] == "attack"
    assert rules.perform(state, intent.order)


def test_status_mode_opens_and_a_click_closes_it():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A2").pos)
    controller.menu_index = menu_index_of(controller, state, "status")
    controller.menu_choose(state)
    assert controller.mode == "status"
    assert controller.selected == "A2", "status keeps the unit selected"

    controller.click(state, state.unit("A2").pos)
    assert controller.mode == "menu"


def test_back_returns_from_a_target_mode_to_the_menu():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A2").pos)
    controller.menu_index = menu_index_of(controller, state, "move")
    controller.menu_choose(state)
    controller.back()
    assert controller.mode == "menu"
    controller.back()
    assert controller.mode == "idle" and controller.selected is None


# ---------------------------------------------------------------------------
# a UI-played game still replays
# ---------------------------------------------------------------------------


def test_a_ui_played_turn_replays_from_its_orders():
    scenario, state = fresh_state()
    controller = Controller()
    orders: list[dict] = []

    def play(order):
        orders.append(order)
        rules.perform(state, order)

    # Move and attack with whatever the engine offers this side.
    moved = False
    for unit in state.units_of("A"):
        if state.status != "ongoing":
            break
        options = rules.unit_options(state, unit.id)
        if options["moves"]:
            destination = options["moves"][0]["pos"]
            intent = controller.click(state, unit.pos)
            assert intent.kind == "select"
            intent = controller.click(state, tuple(destination))
            if intent.kind == "order":
                play(intent.order)
                moved = True
        attacker = state.units.get(unit.id)
        if attacker is not None and rules.attack_options(state, unit.id):
            target = rules.attack_options(state, unit.id)[0]["target"]
            intent = controller.click(state, state.units[unit.id].pos)
            intent = controller.click(state, state.units[target].pos)
            if intent.kind == "order":
                play(intent.order)
    assert moved, "the opening position must allow at least one UI move"

    rebuilt = replay(scenario, orders)
    assert state_hash(rebuilt) == state_hash(state), "the UI's orders are the whole game, event for event"


# ---------------------------------------------------------------------------
# phase 2: cursor, attack range, roster and weapon picker
# ---------------------------------------------------------------------------


def test_the_cursor_starts_on_the_selected_unit_and_moves_by_offset_steps():
    _, state = fresh_state()
    controller = Controller()
    unit = state.unit("A2")
    controller.click(state, unit.pos)
    assert controller.cursor == axial_to_offset(unit.pos)

    col, row = controller.cursor
    controller.move_cursor(state, 1, 0)
    assert controller.cursor == (col + 1, row)
    for _ in range(50):
        controller.move_cursor(state, -1, 0)
    assert controller.cursor_hex() in state.map.tiles, "the cursor never leaves the map"


def test_the_cursor_can_report_the_engines_move_path():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A2").pos)
    destination = tuple(rules.legal_moves(state, "A2")[0]["pos"])
    path = controller.path_to(state, destination)
    assert path is not None
    assert path[0] == list(state.unit("A2").pos) and tuple(path[-1]) == destination
    unreachable = state.unit("B1").pos
    assert controller.path_to(state, unreachable) is None


def test_the_attack_region_is_the_artillerys_range_annulus():
    _, state = fresh_state()
    controller = Controller()
    controller.click(state, state.unit("A5").pos)
    region = controller.attack_region(state)
    assert region
    origin = state.unit("A5").pos
    assert all(2 <= hex_distance(origin, pos) <= 4 for pos in region)
    state.unit("A5").moved = True
    assert controller.attack_region(state) == set(), "a moved artillery has no indirect range left"


def test_the_roster_lists_only_the_current_players_units():
    _, state = fresh_state()
    controller = Controller()
    items = controller.unit_menu_items(state)
    assert {item.key for item in items} == {unit.id for unit in state.units_of("A")}
    assert all(item.hint for item in items)


def test_a_target_with_two_weapons_opens_the_picker_and_fires_the_choice():
    state = make_state(
        rows=["..........", "..C.......", "..........", ".........."],
        regions=[owned_cells("A", [(2, 1)])],
        units=[
            placement("aa_missile", "A", 2, 1, "A1"),
            placement("attack_plane", "B", 3, 1, "B1"),
            placement("infantry", "B", 9, 3, "B9"),
        ],
    )
    controller = Controller()
    controller.click(state, state.unit("A1").pos)
    intent = controller.click(state, state.unit("B1").pos)
    assert intent.kind == "weapon", "two usable weapons must not be chosen silently"
    choices = controller.weapon_menu_items(state)
    assert len(choices) == 2
    assert choices[0].hint.endswith("no counter") or "counter" in choices[0].hint

    order = controller.choose_weapon(state, 0)
    assert order.kind == "order" and order.order["action"] == "attack"
    events = rules.perform(state, order.order)
    assert events[0]["type"] == "attack" and events[0]["weapon"] == "sam"


def test_a_moved_direct_unit_can_be_reselected_and_attack():
    state = make_state(
        rows=["..........", "..C.......", "..........", ".........."],
        regions=[owned_cells("A", [(2, 1)])],
        units=[
            placement("infantry", "A", 2, 1, "A1"),
            placement("tank", "B", 4, 1, "B1"),
            placement("infantry", "B", 9, 3, "B9"),
        ],
    )
    controller = Controller()
    controller.click(state, state.unit("A1").pos)
    intent = controller.click(state, tuple(cell(3, 1)))
    assert intent.kind == "order" and intent.order["action"] == "move"
    rules.perform(state, intent.order)

    controller.click(state, state.unit("A1").pos)  # reselect the mover
    intent = controller.click(state, state.unit("B1").pos)
    assert intent.kind == "order" and intent.order["action"] == "attack"
    assert rules.perform(state, intent.order)


def test_a_moved_artillery_shows_attack_disabled_in_the_ui():
    state = make_state(
        regions=[owned_cells("A", [(2, 1), (3, 1), (4, 1)])],
        units=[
            placement("artillery", "A", 3, 1, "A1"),
            placement("tank", "B", 6, 1, "B1"),
            placement("infantry", "B", 9, 3, "B9"),
        ],
    )
    controller = Controller()
    controller.click(state, state.unit("A1").pos)
    intent = controller.click(state, tuple(cell(2, 1)))
    rules.perform(state, intent.order)

    controller.click(state, state.unit("A1").pos)
    attack = next(item for item in controller.action_items(state) if item.key == "attack")
    assert attack.enabled is False and "0" in attack.label
    assert rules.attack_options(state, "A1") == []
