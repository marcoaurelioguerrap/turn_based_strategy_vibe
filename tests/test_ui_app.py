"""Smoke tests for the real pygame window, run on SDL's dummy video driver.

These are skipped when the UI extra is not installed, so the fast headless
suite stays green everywhere.  They drive the actual event loop, so a break in
input routing or drawing is caught here rather than by a human.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from tbs.engine import rules
from tbs.engine.hexgrid import axial_to_offset
from tbs.ui import buttons, render
from tbs.ui.app import App, build_state


@pytest.fixture
def app():
    pygame.init()
    pygame.display.set_mode((800, 600))
    state = build_state(None, None, None)
    application = App(state, width=800, height=600)
    yield application
    pygame.quit()


def click(application: App, pos: tuple[int, int], button: int = 1) -> None:
    application.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": button}))


def press(application: App, key: int) -> None:
    application.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": key}))


def test_a_frame_renders_without_a_display_running(app):
    surface = pygame.Surface((800, 600))
    app.draw(surface)
    assert surface.get_size() == (800, 600)


def test_clicking_selects_a_unit_and_a_destination_moves_it(app):
    unit = app.state.unit("A2")
    click(app, app.camera.to_screen(unit.pos))
    assert app.controller.selected == "A2"

    destination = rules.legal_moves(app.state, "A2")[0]["pos"]
    click(app, app.camera.to_screen(tuple(destination)))
    assert app.state.unit("A2").pos == tuple(destination)
    assert app.controller.selected == "A2", "the mover stays selected so a direct attack can follow"


def test_space_ends_the_turn_and_hands_it_to_the_other_player(app):
    assert app.state.current_player == "A"
    press(app, pygame.K_SPACE)
    assert app.state.current_player == "B"


def test_b_opens_a_build_menu_on_an_empty_owned_site(app):
    press(app, pygame.K_b)
    assert app.controller.build_site is not None
    assert app.controller.build_options_at(app.state, app.controller.build_site)


def test_zoom_wheel_does_not_crash(app):
    before = app.camera.size
    click(app, (400, 300), button=4)
    assert app.camera.size > before
    click(app, (400, 300), button=5)
    assert app.camera.size == pytest.approx(before)


def test_escape_cancels_a_selection(app):
    unit = app.state.unit("A2")
    click(app, app.camera.to_screen(unit.pos))
    assert app.controller.selected == "A2"
    press(app, pygame.K_ESCAPE)
    assert app.controller.selected is None


# ---------------------------------------------------------------------------
# menus
# ---------------------------------------------------------------------------


def test_main_menu_starts_a_new_game_through_the_scenario_picker(app):
    app.show_main_menu()
    assert app.screen == "main"
    app.draw(pygame.Surface((800, 600)))
    assert app.main_menu.current().key == "new"
    press(app, pygame.K_RETURN)
    assert app.screen == "scenario"
    app.draw(pygame.Surface((800, 600)))
    press(app, pygame.K_RETURN)  # first scenario
    assert app.screen == "play"
    assert app.state.turn_number == 1
    assert app.state.current_player in app.state.players


def test_main_menu_load_is_disabled_without_a_save(app):
    if not app.save_path.exists():
        load = next(item for item in app.main_menu.items if item.key == "load")
        assert load.enabled is False


def test_the_action_menu_move_option_moves_the_unit(app):
    unit = app.state.unit("A2")
    click(app, app.camera.to_screen(unit.pos))
    assert app.controller.mode == "menu"
    rows = render.action_menu_row_rects((800, 600), app.state, app.controller)
    click(app, rows[0].center)  # click "Move" in the side panel
    assert app.controller.mode == "move"
    destination = rules.legal_moves(app.state, "A2")[0]["pos"]
    click(app, app.camera.to_screen(tuple(destination)))
    assert app.state.unit("A2").pos == tuple(destination)


def test_status_opens_with_i_and_draws_and_closes_on_a_click(app):
    unit = app.state.unit("A2")
    click(app, app.camera.to_screen(unit.pos))
    press(app, pygame.K_i)
    assert app.controller.mode == "status"
    app.draw(pygame.Surface((800, 600)))  # must not crash
    click(app, (200, 200))
    assert app.controller.mode == "menu"


def test_escape_opens_and_closes_the_pause_menu(app):
    press(app, pygame.K_ESCAPE)
    assert app.pause_menu is not None
    app.draw(pygame.Surface((800, 600)))
    press(app, pygame.K_ESCAPE)
    assert app.pause_menu is None


def test_the_movement_boundary_renders_for_a_selected_unit(app):
    click(app, app.camera.to_screen(app.state.unit("A2").pos))
    surface = pygame.Surface((800, 600))
    app.draw(surface)  # boundary strokes are drawn without raising


# ---------------------------------------------------------------------------
# phase 2: toolbar, hover, roster, keyboard cursor, attack range
# ---------------------------------------------------------------------------


def test_the_toolbar_end_turn_button_ends_the_turn(app):
    button = next(b for b in buttons.toolbar_layout(render.HUD_HEIGHT, 600) if b.id == "end_turn")
    click(app, (button.x + button.w // 2, button.y + button.h // 2))
    assert app.state.current_player == "B"


def test_the_toolbar_units_button_opens_the_roster_and_choosing_selects(app):
    button = next(b for b in buttons.toolbar_layout(render.HUD_HEIGHT, 600) if b.id == "units")
    click(app, (button.x + button.w // 2, button.y + button.h // 2))
    assert app.unit_menu is not None
    press(app, pygame.K_RETURN)  # first unit in the roster
    assert app.unit_menu is None
    assert app.controller.selected is not None


def test_hovering_a_unit_fills_the_tooltip_and_draws(app):
    unit = app.state.unit("A2")
    pos = app.camera.to_screen(unit.pos)
    app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (int(pos[0]), int(pos[1]))}))
    assert app.hover_hex == unit.pos
    app.draw(pygame.Surface((800, 600)))  # tooltip must render without raising


def test_arrow_keys_move_the_cursor_and_enter_confirms_a_move(app):
    unit = app.state.unit("A2")
    click(app, app.camera.to_screen(unit.pos))
    start = app.controller.cursor
    press(app, pygame.K_RIGHT)
    assert app.controller.cursor != start

    destination = rules.legal_moves(app.state, "A2")[0]["pos"]
    app.controller.cursor = axial_to_offset(tuple(destination))
    press(app, pygame.K_RETURN)
    assert app.state.unit("A2").pos == tuple(destination)


def test_the_attack_range_boundary_renders(app):
    click(app, app.camera.to_screen(app.state.unit("A5").pos))
    app.controller.mode = "attack"
    assert app.controller.attack_region(app.state)
    app.draw(pygame.Surface((800, 600)))


def test_the_build_menu_is_clickable(app):
    site = next(site for site in app.state.build_sites("A") if app.state.unit_at(site.pos) is None)
    click(app, app.camera.to_screen(site.pos))
    assert app.controller.build_site == site.pos
    before = len(app.state.units_of("A"))
    rows = render.build_menu_row_rects((800, 600), app.state, app.controller)
    click(app, rows[0].center)
    assert len(app.state.units_of("A")) == before + 1


