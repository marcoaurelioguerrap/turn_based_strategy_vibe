"""pygame window and event loop: ``py -m tbs.ui.app``.

The window is a thin shell.  Screen points become hexes
(:mod:`tbs.ui.layout`), hexes become orders (:mod:`tbs.ui.interaction`), and
orders run through :func:`tbs.engine.rules.perform`.  It also owns the menu
screens: a main menu (new game / load / quit), a scenario picker, an in-game
pause menu, and the per-unit order menu that :mod:`tbs.ui.interaction` builds.
Nothing in here knows a rule; it only routes input and draws.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # a clear message beats a bare ImportError when the UI extra is absent
    import pygame
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "tbs.ui needs pygame. Install the UI extra first:\n"
        '  py -m pip install -e ".[ui]"\n'
        f"(import error: {exc})"
    ) from exc

from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.errors import ActionError
from tbs.engine.serialize import load_game, save_game
from tbs.engine.state import new_game
from tbs.ui import buttons, info, layout, render
from tbs.ui.interaction import (
    MODE_ATTACK,
    MODE_BUILD,
    MODE_LOAD,
    MODE_MENU,
    MODE_MOVE,
    MODE_STATUS,
    MODE_UNLOAD,
    MODE_WEAPON,
    Controller,
)
from tbs.ui.layout import Camera
from tbs.ui.menu import Menu, MenuItem

DEFAULT_WIDTH = 1100
DEFAULT_HEIGHT = 720
DEFAULT_SAVE = Path("saves") / "quicksave.json"
_TARGET_MODES = (MODE_MOVE, MODE_ATTACK, MODE_LOAD, MODE_UNLOAD)
_ARROW_STEPS = {
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
}


class App:
    """One running window: menus, state, camera, selection and the pygame loop."""

    def __init__(self, state, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> None:
        self.state = state
        self.width = width
        self.height = height
        self.camera = Camera.centered(state, width, height)
        self.controller = Controller()
        self.show_coords = False
        self.show_log = True
        self.message = ""
        self.log_lines: list[str] = []
        self.running = True
        self.save_path = DEFAULT_SAVE
        self.screen = "play"
        self.main_menu = self._make_main_menu()
        self.scenario_menu: Menu | None = None
        self.pause_menu: Menu | None = None
        self.unit_menu: Menu | None = None
        self.hover_hex = None
        self.mouse_pos = (0, 0)
        self._panning = False
        self._pan_last = (0, 0)
        pygame.font.init()
        self.fonts = render.Fonts()

    # -- menus --------------------------------------------------------------

    def _make_main_menu(self) -> Menu:
        return Menu(
            "tbs",
            [
                MenuItem("new", "New Game"),
                MenuItem("load", "Load Game", self.save_path.exists(), str(self.save_path)),
                MenuItem("quit", "Quit"),
            ],
        )

    def show_main_menu(self) -> None:
        self.screen = "main"
        self.main_menu = self._make_main_menu()
        self.pause_menu = None

    def _make_scenario_menu(self) -> Menu:
        items = [MenuItem(str(path), path.stem) for path in available_scenarios()]
        items.append(MenuItem("__back__", "Back"))
        return Menu("New Game", items)

    def _menu_event(self, menu: Menu, event) -> str | None:
        """Route one event through a menu; return the chosen item's key."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                menu.move(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                menu.move(1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                item = menu.choose()
                return item.key if item else None
            elif event.key == pygame.K_ESCAPE:
                return "__back__"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in enumerate(render.menu_item_rects((self.width, self.height), menu)):
                if rect.collidepoint(event.pos):
                    menu.set_index(index)
                    item = menu.choose()
                    return item.key if item else None
        elif event.type == pygame.MOUSEMOTION:
            for index, rect in enumerate(render.menu_item_rects((self.width, self.height), menu)):
                if rect.collidepoint(event.pos) and menu.items[index].enabled:
                    menu.index = index
        return None

    # -- input --------------------------------------------------------------

    def handle_event(self, event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.screen == "main":
            self._handle_main(event)
        elif self.screen == "scenario":
            self._handle_scenario(event)
        else:
            self._handle_play(event)

    def _handle_main(self, event) -> None:
        key = self._menu_event(self.main_menu, event)
        if key == "new":
            self.scenario_menu = self._make_scenario_menu()
            self.screen = "scenario"
        elif key == "load":
            if self._load():
                self.screen = "play"
        elif key == "quit":
            self.running = False

    def _handle_scenario(self, event) -> None:
        assert self.scenario_menu is not None
        key = self._menu_event(self.scenario_menu, event)
        if key == "__back__":
            self.show_main_menu()
        elif key is not None:
            self.start_scenario(key)

    def _handle_play(self, event) -> None:
        if self.unit_menu is not None:
            key = self._menu_event(self.unit_menu, event)
            if key == "__back__":
                self.unit_menu = None
            elif key is not None:
                self._choose_unit_from_roster(key)
            return

        if self.pause_menu is not None:
            key = self._menu_event(self.pause_menu, event)
            if key in ("resume", "__back__"):
                self.pause_menu = None
            elif key == "save":
                self._save()
                self.pause_menu = None
            elif key == "load":
                self._load()
                self.pause_menu = None
            elif key == "menu":
                self.show_main_menu()
            elif key == "quit":
                self.running = False
            return

        if event.type == pygame.KEYDOWN:
            self._on_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._on_left_click(event.pos)
            elif event.button == 2:
                self._panning = True
                self._pan_last = event.pos
            elif event.button == 3:
                self._on_right_click()
            elif event.button == 4:
                self.camera.zoom_at(1.12, *event.pos)
            elif event.button == 5:
                self.camera.zoom_at(1 / 1.12, *event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                self._panning = False
        elif event.type == pygame.MOUSEMOTION:
            self.mouse_pos = event.pos
            if self._panning:
                dx = event.pos[0] - self._pan_last[0]
                dy = event.pos[1] - self._pan_last[1]
                self.camera.pan(dx, dy)
                self._pan_last = event.pos
            else:
                self._update_hover(event.pos)

    def _update_hover(self, pos: tuple[int, int]) -> None:
        if self.pause_menu is not None or self.unit_menu is not None:
            self.hover_hex = None
            return
        if pos[0] < buttons.TOOLBAR_WIDTH + 6 or pos[0] >= self.width - render.PANEL_WIDTH:
            self.hover_hex = None
            return
        self.hover_hex = layout.hex_at(self.state, pos[0], pos[1], self.camera)

    def _on_left_click(self, pos: tuple[int, int]) -> None:
        toolbar = buttons.button_at(
            buttons.toolbar_layout(render.HUD_HEIGHT, self.height), pos[0], pos[1]
        )
        if toolbar is not None:
            self._dispatch_toolbar(toolbar.id)
            return
        if pos[0] < buttons.TOOLBAR_WIDTH:
            return

        if self.controller.mode == MODE_STATUS:
            self.controller.back()
            return

        if pos[0] >= self.width - render.PANEL_WIDTH:
            self._on_panel_click(pos)
            return

        hex_pos = layout.hex_at(self.state, pos[0], pos[1], self.camera)
        if hex_pos is None:
            self.controller.clear()
            self.message = ""
            return
        self.controller.set_cursor(self.state, hex_pos)
        intent = self.controller.click(self.state, hex_pos)
        self.message = intent.note
        if intent.kind == "order":
            self.perform(intent.order)

    def _on_panel_click(self, pos: tuple[int, int]) -> None:
        size = (self.width, self.height)
        if self.controller.build_site is not None:
            for index, rect in enumerate(render.build_menu_row_rects(size, self.state, self.controller)):
                if rect.collidepoint(pos):
                    self._build(index)
                    return
            return
        if self.controller.mode == MODE_WEAPON:
            for index, rect in enumerate(render.weapon_menu_row_rects(size, self.state, self.controller)):
                if rect.collidepoint(pos):
                    self._choose_weapon(index)
                    return
            return
        if self.controller.selected is not None:
            for index, rect in enumerate(render.action_menu_row_rects(size, self.state, self.controller)):
                if rect.collidepoint(pos):
                    self.controller.menu_index = index
                    self._menu_choose()
                    return

    def _dispatch_toolbar(self, button_id: str) -> None:
        if button_id == "menu":
            self._open_pause()
        elif button_id == "end_turn":
            self._end_turn()
        elif button_id == "units":
            self._open_unit_menu()
        elif button_id == "build":
            site = self.controller.open_build_menu(self.state)
            self.message = f"build at {list(site)}" if site is not None else "no empty build site"
        elif button_id == "save":
            self._save()
        elif button_id == "load":
            self._load()
        elif button_id == "log":
            self.show_log = not self.show_log
        elif button_id == "coords":
            self.show_coords = not self.show_coords
        elif button_id == "zoom_in":
            self.camera.zoom_at(1.15, self.width / 2, self.height / 2)
        elif button_id == "zoom_out":
            self.camera.zoom_at(1 / 1.15, self.width / 2, self.height / 2)

    def _on_right_click(self) -> None:
        if self.controller.mode in _TARGET_MODES or self.controller.mode in (
            MODE_STATUS,
            MODE_BUILD,
            MODE_WEAPON,
        ):
            self.controller.back()
        else:
            self.controller.clear()
        self.message = ""

    def _on_key(self, event) -> None:
        key = event.key
        shift = bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)

        if self.controller.mode == MODE_STATUS:
            if key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.controller.back()
            return
        if self.controller.mode == MODE_WEAPON:
            if key in (pygame.K_UP, pygame.K_w):
                self.controller.weapon_move(-1)
            elif key in (pygame.K_DOWN, pygame.K_s):
                self.controller.weapon_move(1)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_f):
                self._choose_weapon(self.controller.weapon_index)
            elif key == pygame.K_ESCAPE:
                self.controller.back()
            return
        if self.controller.build_site is not None:
            if key in (pygame.K_ESCAPE, pygame.K_p):
                self.controller.back()
            elif pygame.K_1 <= key <= pygame.K_9:
                self._build(key - pygame.K_1)
            return

        if key in _ARROW_STEPS:
            dcol, drow = _ARROW_STEPS[key]
            if shift:
                self.camera.pan(-dcol * 60, -drow * 60)
            else:
                self.controller.move_cursor(self.state, dcol, drow)
            return

        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._confirm_cursor()
            return

        if self.controller.mode == MODE_MENU:
            if key in (pygame.K_w, pygame.K_PAGEUP):
                self.controller.menu_move(self.state, -1)
                return
            if key in (pygame.K_s, pygame.K_PAGEDOWN):
                self.controller.menu_move(self.state, 1)
                return
            if key == pygame.K_f:
                self._menu_choose()
                return
            if key == pygame.K_ESCAPE:
                self.controller.clear()
                self.message = ""
                return
        elif self.controller.mode in _TARGET_MODES:
            if key == pygame.K_ESCAPE:
                self.controller.back()
                self.message = ""
                return

        # keys that work from anywhere in play
        if key == pygame.K_h:
            self.perform(self.controller.hold_order(self.state))
        elif key in (pygame.K_e, pygame.K_SPACE):
            self._end_turn()
        elif key == pygame.K_TAB:
            self.controller.cycle_ready(self.state)
        elif key == pygame.K_i:
            self._open_status()
        elif key == pygame.K_u:
            self._open_unit_menu()
        elif key == pygame.K_b:
            site = self.controller.open_build_menu(self.state)
            self.message = f"build at {list(site)}" if site is not None else "no empty build site"
        elif key == pygame.K_c:
            self.show_coords = not self.show_coords
        elif key == pygame.K_l:
            self.show_log = not self.show_log
        elif key == pygame.K_m:
            self._open_pause()
        elif key == pygame.K_HOME:
            self.camera = Camera.centered(self.state, self.width, self.height)
        elif key == pygame.K_F5:
            self._save()
        elif key == pygame.K_F9:
            self._load()
        elif key == pygame.K_ESCAPE:
            if self.controller.selected is not None:
                self.controller.clear()
                self.message = ""
            else:
                self._open_pause()

    def _confirm_cursor(self) -> None:
        pos = self.controller.cursor_hex()
        if pos is None:
            return
        intent = self.controller.click(self.state, pos)
        self.message = intent.note
        if intent.kind == "order":
            self.perform(intent.order)

    def _choose_weapon(self, index: int) -> None:
        intent = self.controller.choose_weapon(self.state, index)
        self.message = intent.note
        if intent.kind == "order":
            self.perform(intent.order)

    def _open_status(self) -> None:
        if self.controller.selected is not None:
            self.controller.mode = MODE_STATUS
            self.message = f"{self.controller.selected} status"

    def _open_unit_menu(self) -> None:
        items = self.controller.unit_menu_items(self.state)
        if not items:
            self.message = "no units"
            return
        self.unit_menu = Menu(f"{self.state.current_player} units", items)
        self.hover_hex = None
        self.message = ""

    def _choose_unit_from_roster(self, unit_id: str) -> None:
        self.unit_menu = None
        if unit_id not in self.state.units:
            return
        self.controller.select_unit(self.state, unit_id)
        unit = self.state.units[unit_id]
        if unit.pos is not None:
            self.camera.focus_on(unit.pos, self.width, self.height)
        self.message = f"selected {unit_id}"

    def _open_pause(self) -> None:
        self.pause_menu = Menu(
            "paused",
            [
                MenuItem("resume", "Resume"),
                MenuItem("save", "Save"),
                MenuItem("load", "Load", self.save_path.exists()),
                MenuItem("menu", "Main Menu"),
                MenuItem("quit", "Quit"),
            ],
        )

    # -- orders -------------------------------------------------------------

    def _menu_choose(self) -> None:
        intent = self.controller.menu_choose(self.state)
        self.message = intent.note
        if intent.kind == "order":
            self.perform(intent.order)

    def perform(self, order: dict | None) -> list[dict]:
        if order is None or self.state.status != "ongoing":
            return []
        moved_id = order.get("unit") if order.get("action") == "move" else None
        try:
            events = rules.perform(self.state, order)
        except ActionError as error:
            self.message = str(error)
            return []
        self.controller.clear()
        self.message = ""
        for event in events:
            line = render.describe_event(event)
            if line:
                self.log_lines.append(line)
        self.log_lines = self.log_lines[-60:]
        if moved_id is not None and self.state.status == "ongoing":
            unit = self.state.units.get(moved_id)
            if unit is not None and not unit.acted:
                # keep the mover selected so a direct attack or capture can follow
                self.controller.select_unit(self.state, moved_id)
                self.message = f"{moved_id} moved - attack or act again"
        return events

    def _end_turn(self) -> None:
        self.perform(self.controller.end_turn_order())

    def _build(self, index: int) -> None:
        site = self.controller.build_site
        if site is None:
            return
        options = self.controller.build_options_at(self.state, site)
        if not 0 <= index < len(options):
            self.message = "no such build option"
            return
        self.perform(self.controller.build_order(self.state, site, options[index]["unit"]))

    def start_scenario(self, path: str) -> None:
        scenario = load_scenario(Path(path))
        state = new_game(scenario, load_specs())
        rules.start_game(state)
        self._adopt(state)
        self.screen = "play"
        self.message = scenario.get("name", "")

    def _adopt(self, state) -> None:
        self.state = state
        self.camera = Camera.centered(state, self.width, self.height)
        self.controller.clear()
        self.log_lines = []
        self.pause_menu = None
        self.unit_menu = None
        self.hover_hex = None

    def _save(self) -> None:
        save_game(self.state, self.save_path)
        self.message = f"saved to {self.save_path}"

    def _load(self) -> bool:
        try:
            state = load_game(self.save_path)
        except OSError:
            self.message = f"no save at {self.save_path}"
            return False
        self._adopt(state)
        self.message = f"loaded {self.save_path}"
        return True

    # -- drawing and the loop ----------------------------------------------

    def draw(self, surface) -> None:
        if self.screen == "main":
            render.draw_menu_screen(surface, self.main_menu, self.fonts, subtitle="a turn-based hex wargame")
        elif self.screen == "scenario":
            assert self.scenario_menu is not None
            render.draw_menu_screen(surface, self.scenario_menu, self.fonts, subtitle="new game - choose a scenario")
        else:
            tooltip_lines = None
            if (
                self.hover_hex is not None
                and self.pause_menu is None
                and self.unit_menu is None
            ):
                tooltip_lines = info.hover_lines(self.state, self.hover_hex)
            render.draw_frame(
                surface,
                self.state,
                self.camera,
                self.controller,
                self.fonts,
                message=self.message,
                log_lines=self.log_lines,
                show_log=self.show_log,
                show_coords=self.show_coords,
                pause_menu=self.pause_menu,
                tooltip_lines=tooltip_lines,
                tooltip_pos=self.mouse_pos,
            )
            if self.unit_menu is not None:
                render.draw_modal_menu(
                    surface, self.unit_menu, self.fonts, f"{self.state.current_player} units"
                )

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("tbs")
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self.draw(screen)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def build_state(scenario_path: str | None, seed: int | None, session: str | None):
    if session:
        return load_game(Path(session))
    path = Path(scenario_path) if scenario_path else available_scenarios()[0]
    scenario = load_scenario(path)
    if seed is not None:
        scenario["seed"] = seed
    state = new_game(scenario, load_specs())
    rules.start_game(state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbs.ui.app", description="pygame hot-seat window over the tbs engine")
    parser.add_argument("--scenario", help="scenario JSON path (default: the bundled border skirmish)")
    parser.add_argument("--seed", type=int, help="override the scenario seed")
    parser.add_argument("--session", help="resume from a save file instead of starting a scenario")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--no-menu", action="store_true", help="skip the main menu and start playing at once")
    args = parser.parse_args(argv)
    state = build_state(args.scenario, args.seed, args.session)
    app = App(state, width=args.width, height=args.height)
    if not args.no_menu and not args.session:
        app.show_main_menu()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
