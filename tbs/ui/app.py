"""pygame window and event loop: ``py -m tbs.ui.app``.

The window is a thin shell.  Screen points become hexes
(:mod:`tbs.ui.layout`), hexes become orders (:mod:`tbs.ui.interaction`), and
orders run through :func:`tbs.engine.rules.perform`.  It also owns the menu
screens: a main menu (new game / new game vs LLM / load / quit), a scenario
picker, a side picker, an in-game pause menu, and the per-unit order menu that
:mod:`tbs.ui.interaction` builds.

An optional local LLM plays one side through the same agent protocol.  Its HTTP
call runs in a worker thread; the main thread builds the request, applies every
order, and owns all state mutation.  Nothing in here knows a rule; the engine
remains authoritative.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

try:  # a clear message beats a bare ImportError when the UI extra is absent
    import pygame
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        "tbs.ui needs pygame. Install the UI extra first:\n"
        '  py -m pip install -e ".[ui]"\n'
        f"(import error: {exc})"
    ) from exc

from tbs.agents.council import CouncilAgent
from tbs.agents.generals import DEFAULT_GENERALS, load_generals, select_generals
from tbs.agents.llm import (
    DEFAULT_BASE_URL,
    RETRYABLE_LLM_ERRORS,
    JsonlLogger,
    LlmClient,
    LlmConfig,
    LlmTransportError,
    SingleLlmAgent,
)
from tbs.agents.protocol import Agent
from tbs.agents.turn import StepwiseSession, TurnSession
from tbs.engine import rules
from tbs.engine.data import available_scenarios, default_scenario, load_scenario, load_specs
from tbs.engine.errors import ActionError
from tbs.engine.serialize import load_game, save_game
from tbs.engine.state import new_game
from tbs.ui import buttons, info, layout, picker, render
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

    def __init__(
        self,
        state,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        llm_config: LlmConfig | None = None,
        llm_attempts: int = 2,
        llm_log: str | None = None,
        agent: Agent | None = None,
        stepwise: bool = False,
        agent_max_steps: int = 64,
        agent_kind: str = "llm",
        council_generals=None,
        council_use_commander: bool = True,
        council_parallel: bool = False,
    ) -> None:
        self.state = state
        self.width = width
        self.height = height
        self.camera = layout.Camera.fit(state, width, height)
        self.controller = Controller()
        self.show_coords = False
        self.show_log = True
        self.message = ""
        self.log_lines: list[str] = []
        self.running = True
        self.save_path = DEFAULT_SAVE
        self.screen = "play"
        self.hover_hex = None
        self.mouse_pos = (0, 0)
        self._panning = False
        self._pan_last = (0, 0)

        # local LLM opponent, or an injected agent (the NCM player)
        self.llm_config = llm_config
        self.llm_attempts = max(1, int(llm_attempts))
        self.llm_logger = JsonlLogger(llm_log) if llm_log else None
        self.external_agent = agent
        self.stepwise = bool(stepwise)
        self.agent_max_steps = max(1, int(agent_max_steps))
        self.agent_kind = agent_kind if agent_kind in ("llm", "council") else "llm"
        self.council_generals = tuple(council_generals or DEFAULT_GENERALS)
        self.council_use_commander = bool(council_use_commander)
        self.council_parallel = bool(council_parallel)
        self.agent: Agent | None = agent
        self.agent_label = (
            "NCM" if stepwise else ("Council" if self.agent_kind == "council" else "LLM")
        )
        self.agent_side: str | None = None
        self.agent_enabled = False
        self.agent_session: TurnSession | StepwiseSession | None = None
        self.agent_queue: queue.Queue = queue.Queue()
        self.agent_thread: threading.Thread | None = None
        self.agent_token = 0
        self.agent_busy = False
        self.agent_started_at = 0.0
        self.agent_error_menu: Menu | None = None

        # menu state
        self.pending_vs_llm = False
        self.pending_scenario: str | None = None
        self.side_menu: Menu | None = None
        self.scenario_infos: dict[str, picker.ScenarioInfo] = {}

        self.main_menu = self._make_main_menu()
        self.scenario_menu: Menu | None = None
        self.pause_menu: Menu | None = None
        self.unit_menu: Menu | None = None
        pygame.font.init()
        self.fonts = render.Fonts()
        if self.agent is None and self.llm_config is not None and self.llm_config.model:
            self.agent = SingleLlmAgent(LlmClient(self.llm_config), logger=self.llm_logger)

    # -- menus --------------------------------------------------------------

    def _make_main_menu(self) -> Menu:
        llm_ready = self.llm_config is not None and bool(self.llm_config.model)
        label = "New Game vs Council" if self.agent_kind == "council" else "New Game vs LLM"
        return Menu(
            "tbs",
            [
                MenuItem("new", "New Game (hot-seat)"),
                MenuItem(
                    "new_llm",
                    label,
                    llm_ready,
                    "" if llm_ready else "set --llm-model or TBS_LLM_MODEL",
                ),
                MenuItem("load", "Load Game", self.save_path.exists(), str(self.save_path)),
                MenuItem("quit", "Quit"),
            ],
        )

    def show_main_menu(self) -> None:
        self._cancel_agent_run()
        self.screen = "main"
        self.main_menu = self._make_main_menu()
        self.pause_menu = None
        self.side_menu = None

    def _make_scenario_menu(self) -> Menu:
        infos = picker.scenario_infos(available_scenarios())
        self.scenario_infos = {str(info.path): info for info in infos}
        items = [
            MenuItem(
                str(info.path),
                info.title,
                enabled=bool(info.rows),
                hint=info.summary if info.rows else info.description,
            )
            for info in infos
        ]
        items.append(MenuItem("__back__", "Back"))
        title = f"New Game vs {self.agent_label}" if self.pending_vs_llm else "New Game"
        menu = Menu(title, items, visible_rows=self._menu_visible_rows())
        # Start on the explicit default map, not whichever filename sorts first.
        default_key = str(default_scenario())
        for index, item in enumerate(items):
            if item.key == default_key:
                menu.set_index(index)
                break
        return menu

    def _menu_visible_rows(self) -> int:
        """How many menu rows fit between the title and the footer."""
        return max(3, (self.height - 260) // render.MENU_ITEM_HEIGHT)

    def _selected_scenario(self) -> picker.ScenarioInfo | None:
        if self.scenario_menu is None:
            return None
        item = self.scenario_menu.current()
        if item is None:
            return None
        return self.scenario_infos.get(item.key)

    def _make_side_menu(self) -> Menu:
        return Menu(
            "Choose your side",
            [
                MenuItem("A", f"Play as A — {self.agent_label} plays B"),
                MenuItem("B", f"Play as B — {self.agent_label} plays A"),
                MenuItem("__back__", "Back"),
            ],
        )

    def _menu_event(self, menu: Menu, event) -> str | None:
        """Route one event through a menu; return the chosen item's key."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                menu.move(-1)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                menu.move(1)
            elif event.key == pygame.K_PAGEUP:
                menu.page(-1)
            elif event.key == pygame.K_PAGEDOWN:
                menu.page(1)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                item = menu.choose()
                return item.key if item else None
            elif event.key == pygame.K_ESCAPE:
                return "__back__"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in render.menu_item_rects((self.width, self.height), menu):
                if rect.collidepoint(event.pos):
                    menu.set_index(index)
                    item = menu.choose()
                    return item.key if item else None
        elif event.type == pygame.MOUSEMOTION:
            for index, rect in render.menu_item_rects((self.width, self.height), menu):
                if rect.collidepoint(event.pos) and menu.items[index].enabled:
                    menu.set_index(index)
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
        elif self.screen == "side":
            self._handle_side(event)
        else:
            self._handle_play(event)

    def _handle_main(self, event) -> None:
        key = self._menu_event(self.main_menu, event)
        if key in ("new", "new_llm"):
            self.pending_vs_llm = key == "new_llm"
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
        elif key is not None and self.pending_vs_llm:
            self.pending_scenario = key
            self.side_menu = self._make_side_menu()
            self.screen = "side"
        elif key is not None:
            self.start_scenario(key, agent_side=None)

    def _handle_side(self, event) -> None:
        assert self.side_menu is not None
        key = self._menu_event(self.side_menu, event)
        if key == "__back__":
            self.screen = "scenario"
        elif key in ("A", "B") and self.pending_scenario is not None:
            self.start_scenario(self.pending_scenario, agent_side="B" if key == "A" else "A")

    def _handle_play(self, event) -> None:
        if self.agent_error_menu is not None:
            key = self._menu_event(self.agent_error_menu, event)
            if key == "retry":
                self.agent_error_menu = None
                self.message = f"retrying the {self.agent_label.lower()}..."
                self._start_agent_turn()
            elif key == "hotseat":
                self.agent_error_menu = None
                self.agent_enabled = False
                self.message = f"{self.agent_label} disabled - you also control {self.agent_side}"
            elif key == "quit":
                self.running = False
            return

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
            elif key == "takeover":
                self._cancel_agent_run()
                self.agent_enabled = False
                self.pause_menu = None
                self.message = f"{self.agent_label} disabled - you also control {self.agent_side}"
            elif key == "resume_llm":
                self.agent_enabled = True
                self.pause_menu = None
                self._make_agent()
                self._maybe_start_agent_turn()
            elif key == "menu":
                self.show_main_menu()
            elif key == "quit":
                self.running = False
            return

        if self.agent_busy or self._agent_controls_current_player():
            self._handle_agent_wait(event)
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

    def _handle_agent_wait(self, event) -> None:
        """During the LLM's turn only menu/save/log input is accepted."""
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_m, pygame.K_ESCAPE):
                self._open_pause()
            elif event.key == pygame.K_l:
                self.show_log = not self.show_log
            elif event.key == pygame.K_F5:
                self._save()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            toolbar = buttons.button_at(
                buttons.toolbar_layout(render.HUD_HEIGHT, self.height), event.pos[0], event.pos[1]
            )
            if toolbar is not None and toolbar.id == "menu":
                self._open_pause()

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
            self.camera = layout.Camera.fit(self.state, self.width, self.height)
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
        self.unit_menu = Menu(
            f"{self.state.current_player} units",
            items,
            visible_rows=self._menu_visible_rows(),
        )
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
        items = [
            MenuItem("resume", "Resume"),
            MenuItem("save", "Save"),
            MenuItem("load", "Load", self.save_path.exists()),
        ]
        if self.agent_side is not None and (
            self.external_agent is not None or self.llm_config is not None
        ):
            if self.agent_enabled:
                items.append(
                    MenuItem(
                        "takeover",
                        f"Take over for {self.agent_side} (disable {self.agent_label})",
                    )
                )
            else:
                items.append(MenuItem("resume_llm", f"Resume {self.agent_label}"))
        items += [MenuItem("menu", "Main Menu"), MenuItem("quit", "Quit")]
        self.pause_menu = Menu("paused", items)

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
        self._maybe_start_agent_turn()

    def _build(self, index: int) -> None:
        site = self.controller.build_site
        if site is None:
            return
        options = self.controller.build_options_at(self.state, site)
        if not 0 <= index < len(options):
            self.message = "no such build option"
            return
        self.perform(self.controller.build_order(self.state, site, options[index]["unit"]))

    def start_scenario(self, path: str, agent_side: str | None = None) -> None:
        scenario = load_scenario(Path(path))
        state = new_game(scenario, load_specs())
        rules.start_game(state)
        self._adopt(state)
        self.screen = "play"
        self.agent_side = agent_side
        self.agent_enabled = agent_side is not None and (
            self.external_agent is not None or self.llm_config is not None
        )
        if self.agent_enabled:
            self._make_agent()
        human = "hot-seat" if agent_side is None else f"you are {self._human_side()}"
        title = scenario.get("title") or scenario.get("name", "")
        self.message = f"{title} - {human}"
        self._maybe_start_agent_turn()

    def _adopt(self, state) -> None:
        self._cancel_agent_run()
        self.state = state
        self.camera = layout.Camera.fit(state, self.width, self.height)
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
        if self.agent_enabled:
            self._make_agent()
            self._maybe_start_agent_turn()
        self.message = f"loaded {self.save_path}"
        return True

    # -- the local LLM / NCM opponent ---------------------------------------

    def enable_agent(self, side: str) -> None:
        """Start a game already configured to let an agent play ``side``."""
        self.agent_side = side
        self.agent_enabled = self.external_agent is not None or self.llm_config is not None
        if not self.agent_enabled:
            self.message = "no agent configured: set --llm-model or --ncm-checkpoint"
            return
        self._make_agent()
        self._maybe_start_agent_turn()

    def _make_agent(self) -> None:
        if self.external_agent is not None:
            self.agent = self.external_agent
            return
        if self.llm_config is None or not self.llm_config.model:
            self.agent = None
            return
        if self.agent_kind == "council":
            self.agent = CouncilAgent(
                LlmClient(self.llm_config),
                generals=self.council_generals,
                logger=self.llm_logger,
                use_commander=self.council_use_commander,
                parallel=self.council_parallel,
            )
        else:
            self.agent = SingleLlmAgent(LlmClient(self.llm_config), logger=self.llm_logger)

    def _human_side(self) -> str:
        return "B" if self.agent_side == "A" else "A"

    def _agent_controls_current_player(self) -> bool:
        return (
            self.agent_enabled
            and self.agent_side is not None
            and self.state.status == "ongoing"
            and self.state.current_player == self.agent_side
        )

    def _maybe_start_agent_turn(self) -> None:
        if not self._agent_controls_current_player():
            return
        if self.agent_busy or self.agent_session is not None:
            return
        if self.agent is None:
            self._make_agent()
        if self.agent is None:
            self.agent_enabled = False
            self.message = "no agent configured: set --llm-model or --ncm-checkpoint"
            return
        if self.stepwise:
            self.agent_session = StepwiseSession(
                self.state,
                self.agent,
                max_steps=self.agent_max_steps,
                logger=self.llm_logger,
            )
        else:
            self.agent_session = TurnSession(
                self.state,
                self.agent,
                attempts=self.llm_attempts,
                logger=self.llm_logger,
            )
        self.message = f"{self.agent_side} ({self.agent_label}) is planning..."
        self._request_next_plan()

    def _start_agent_turn(self) -> None:
        """Retry entry point after a transport error aborted the old session."""
        self.agent_session = None
        self.agent_busy = False
        self._maybe_start_agent_turn()

    def _request_next_plan(self) -> None:
        session = self.agent_session
        if session is None:
            return
        request = session.next_request()
        if request is None:
            self._finish_agent_turn()
            return
        self.agent_busy = True
        self.agent_started_at = time.monotonic()
        token = self.agent_token
        self.agent_thread = threading.Thread(
            target=self._agent_worker,
            args=(request, token),
            name="tbs-llm",
            daemon=True,
        )
        self.agent_thread.start()

    def _agent_worker(self, request, token: int) -> None:
        """Worker thread: HTTP + parsing only, never touches GameState."""
        try:
            plan = self.agent.plan(request) if self.agent is not None else None
        except Exception as exc:  # handed to the main thread for classification
            self.agent_queue.put((token, "error", exc))
        else:
            self.agent_queue.put((token, "plan", plan))

    def _poll_agent(self) -> None:
        while True:
            try:
                token, kind, payload = self.agent_queue.get_nowait()
            except queue.Empty:
                break
            if token != self.agent_token:
                continue  # a result from a cancelled or replaced game
            self.agent_busy = False
            self.agent_started_at = 0.0
            session = self.agent_session
            if session is None:
                continue
            if kind == "plan":
                session.record_plan(payload)
                if session.ready():
                    self._request_next_plan()
                else:
                    self._finish_agent_turn()
                continue
            error = payload
            if isinstance(error, LlmTransportError):
                session.abort()
                self.agent_session = None
                self._show_agent_error(str(error))
            elif isinstance(error, RETRYABLE_LLM_ERRORS):
                session.record_failure(str(error))
                if session.ready():
                    self._request_next_plan()
                else:
                    self._finish_agent_turn()
            else:
                session.abort()
                self.agent_session = None
                self._show_agent_error(f"agent error: {error}")

    def _finish_agent_turn(self) -> None:
        session = self.agent_session
        self.agent_session = None
        self.agent_busy = False
        self.agent_started_at = 0.0
        if session is None:
            return
        result = session.finish()
        self.controller.clear()
        self.hover_hex = None
        for event in result.events:
            line = render.describe_event(event)
            if line:
                self.log_lines.append(line)
        for comment in result.commentary[-5:]:
            parts = comment.split("; ") if self.agent_kind == "council" else [comment]
            for part in parts:
                self.log_lines.append(f"{self.agent_label}: {part[:180]}")
        if result.errors:
            self.log_lines.append(f"{self.agent_label} orders rejected: {result.errors[-1]}")
        if result.ignored:
            self.log_lines.append(
                f"{self.agent_label} ignored {len(result.ignored)} duplicate order(s) for units that already acted"
            )
        self.log_lines = self.log_lines[-60:]
        if result.applied:
            self.message = f"{result.player} ({self.agent_label}) applied {len(result.applied)} order(s)"
        elif result.errors:
            self.message = f"{result.player} ({self.agent_label}) produced no usable orders"
        else:
            self.message = f"{result.player} ({self.agent_label}) passed"
        if result.ended_turn:
            self.message = f"{result.player} ({self.agent_label}) ends its turn"
        self._maybe_start_agent_turn()

    def _show_agent_error(self, message: str) -> None:
        self.message = f"{self.agent_label} unavailable - Retry, switch to hot-seat, or quit"
        self.agent_error_menu = Menu(
            f"{self.agent_label} unavailable",
            [
                MenuItem("retry", "Retry"),
                MenuItem("hotseat", "Switch to hot-seat"),
                MenuItem("quit", "Quit"),
            ],
        )
        self.log_lines.append(f"{self.agent_label} error: {message}")
        self.log_lines = self.log_lines[-60:]

    def _cancel_agent_run(self) -> None:
        self.agent_token += 1
        self.agent_busy = False
        self.agent_started_at = 0.0
        self.agent_session = None
        self.agent_error_menu = None
        while True:
            try:
                self.agent_queue.get_nowait()
            except queue.Empty:
                break

    # -- drawing and the loop ----------------------------------------------

    def draw(self, surface) -> None:
        if self.screen == "main":
            render.draw_menu_screen(surface, self.main_menu, self.fonts, subtitle="a turn-based hex wargame")
        elif self.screen == "scenario":
            assert self.scenario_menu is not None
            scenario_info = self._selected_scenario()
            subtitle = (
                f"new game vs {self.agent_label} - choose a scenario"
                if self.pending_vs_llm
                else "new game - choose a scenario"
            )
            detail_lines = ()
            preview: tuple[str, ...] = ()
            if scenario_info is not None and scenario_info.rows:
                detail_lines = (scenario_info.detail,)
                preview = picker.preview_grid(scenario_info.rows)
            render.draw_menu_screen(
                surface,
                self.scenario_menu,
                self.fonts,
                subtitle=subtitle,
                detail_lines=detail_lines,
                preview=preview,
            )
        elif self.screen == "side":
            assert self.side_menu is not None
            render.draw_menu_screen(surface, self.side_menu, self.fonts, subtitle="choose your side")
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
            if self.agent_busy:
                elapsed = max(0.0, time.monotonic() - self.agent_started_at)
                model = self.llm_config.model if self.llm_config is not None else self.agent_label
                verb = "is deliberating..." if self.agent_kind == "council" else "is thinking..."
                detail = f"{model}  -  {elapsed:.0f}s"
                if self.agent_kind == "council":
                    detail = f"{model}  -  {len(self.council_generals)} generals  -  {elapsed:.0f}s"
                render.draw_thinking_overlay(
                    surface,
                    self.fonts,
                    f"{self.agent_side} ({self.agent_label}) {verb}",
                    detail,
                )
            if self.agent_error_menu is not None:
                render.draw_modal_menu(
                    surface, self.agent_error_menu, self.fonts, f"{self.agent_label} unavailable"
                )

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("tbs")
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self._poll_agent()
            self.draw(screen)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()


def build_state(scenario_path: str | None, seed: int | None, session: str | None):
    if session:
        return load_game(Path(session))
    path = Path(scenario_path) if scenario_path else default_scenario()
    scenario = load_scenario(path)
    if seed is not None:
        scenario["seed"] = seed
    state = new_game(scenario, load_specs())
    rules.start_game(state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbs.ui.app", description="pygame hot-seat window over the tbs engine")
    parser.add_argument("--scenario", help="scenario JSON path (default: data/maps/border_skirmish.json)")
    parser.add_argument("--seed", type=int, help="override the scenario seed")
    parser.add_argument("--session", help="resume from a save file instead of starting a scenario")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--no-menu", action="store_true", help="skip the main menu and start playing at once")
    parser.add_argument("--agent-side", choices=("A", "B"), help="let an agent play this side")
    parser.add_argument(
        "--agent-kind",
        choices=("llm", "council"),
        default="llm",
        help="one LLM commander, or the M10 council of generals",
    )
    parser.add_argument(
        "--council-members",
        help="comma-separated general ids for --agent-kind council (default: all)",
    )
    parser.add_argument(
        "--council-config",
        help="JSON roster override for --agent-kind council",
    )
    parser.add_argument(
        "--council-no-commander",
        action="store_true",
        help="skip the supreme commander's pre-brief (one fewer LLM call per turn)",
    )
    parser.add_argument(
        "--council-parallel",
        action="store_true",
        help="ask generals concurrently (the local server may still serialize them)",
    )
    parser.add_argument(
        "--ncm-checkpoint",
        help="directory with tokenizer.pt/generator.pt; plays the NCM model (needs the NCM environment)",
    )
    parser.add_argument("--ncm-device", help="torch device for the NCM model (default: cuda when available)")
    parser.add_argument("--ncm-max-steps", type=int, default=64, help="NCM order batches per turn")
    parser.add_argument(
        "--llm-base-url", default=os.environ.get("TBS_LLM_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--llm-model", default=os.environ.get("TBS_LLM_MODEL"))
    parser.add_argument("--llm-api-key", default=os.environ.get("TBS_LLM_API_KEY"))
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1500)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-retries", type=int, default=2, help="transport retries")
    parser.add_argument("--llm-attempts", type=int, default=2, help="whole-turn attempts after rejections")
    parser.add_argument("--llm-json-mode", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--llm-no-thinking",
        action="store_true",
        help="disable model reasoning (recommended for the 8k-context Nemotron server)",
    )
    thinking.add_argument("--llm-thinking", action="store_true", help="force model reasoning on")
    parser.add_argument(
        "--llm-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "max", "xhigh"),
        help="server-side reasoning effort, when supported",
    )
    parser.add_argument("--llm-log", help="append prompt/response JSONL here")
    args = parser.parse_args(argv)

    llm_config = None
    if args.llm_model:
        llm_config = LlmConfig(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
            json_mode=args.llm_json_mode,
            enable_thinking=True if args.llm_thinking else False,
            reasoning_effort=args.llm_reasoning_effort,
        )
    council_generals = None
    if args.agent_kind == "council":
        if llm_config is None:
            parser.error("--agent-kind council needs --llm-model or TBS_LLM_MODEL")
        try:
            council_generals = (
                load_generals(args.council_config) if args.council_config else DEFAULT_GENERALS
            )
            if args.council_members:
                council_generals = select_generals(
                    council_generals,
                    [item.strip() for item in args.council_members.split(",") if item.strip()],
                )
        except ValueError as exc:
            parser.error(str(exc))
    external_agent: Agent | None = None
    if args.ncm_checkpoint:
        try:
            from tbs_ncm.ncm_agent import NcmAgent, NcmSampler
        except ImportError as exc:  # pragma: no cover - depends on the environment
            parser.error(f"--ncm-checkpoint needs the NCM environment ({exc})")
        external_agent = NcmAgent(
            NcmSampler(args.ncm_checkpoint, device=args.ncm_device)
        )
    if args.agent_side and llm_config is None and external_agent is None:
        parser.error("--agent-side needs --llm-model or --ncm-checkpoint")
    if external_agent is not None and not args.agent_side:
        parser.error("--ncm-checkpoint needs --agent-side A or B")

    state = build_state(args.scenario, args.seed, args.session)
    app = App(
        state,
        width=args.width,
        height=args.height,
        llm_config=llm_config,
        llm_attempts=args.llm_attempts,
        llm_log=args.llm_log,
        agent=external_agent,
        stepwise=external_agent is not None,
        agent_max_steps=args.ncm_max_steps,
        agent_kind=args.agent_kind,
        council_generals=council_generals,
        council_use_commander=not args.council_no_commander,
        council_parallel=args.council_parallel,
    )
    menu_shown = not args.no_menu and not args.session
    if menu_shown:
        app.show_main_menu()
    if not menu_shown and args.agent_side:
        app.enable_agent(args.agent_side)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
