"""pygame drawing for the tbs window.  The only module here that needs pygame.

It draws exactly what the engine reports: terrain and ownership from
``state.map``, units from ``state.specs`` (glyphs, names, HP), and highlights
from the very same option lists the agent API serves.  The movement range is
stroked as a *boundary* around every hex the selected unit can reach, and the
selected unit's orders appear as a menu.  Nothing here decides what is legal;
that is :mod:`tbs.ui.interaction`.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import pygame

from tbs.engine import rules, view
from tbs.engine.hexgrid import Hex, axial_to_offset
from tbs.engine.state import GameState
from tbs.ui import buttons, palette
from tbs.ui.interaction import (
    MODE_ATTACK,
    MODE_IDLE,
    MODE_LOAD,
    MODE_MENU,
    MODE_MOVE,
    MODE_STATUS,
    MODE_UNLOAD,
    MODE_WEAPON,
    Controller,
)
from tbs.ui.layout import Camera, hex_corners, reachable_boundary
from tbs.ui.menu import Menu

HUD_HEIGHT = 42
PANEL_WIDTH = 320
LOG_LINES = 9
MENU_ITEM_HEIGHT = 44
MENU_WIDTH = 380


class Fonts:
    """The handful of fonts the window uses, all from pygame's built-in font."""

    def __init__(self) -> None:
        self.tiny = pygame.font.Font(None, 15)
        self.small = pygame.font.Font(None, 18)
        self.body = pygame.font.Font(None, 21)
        self.glyph = pygame.font.Font(None, 24)
        self.hud = pygame.font.Font(None, 24)
        self.big = pygame.font.Font(None, 46)


def _text(surface: pygame.Surface, font: pygame.font.Font, text: str, center: tuple[float, float], color) -> None:
    image = font.render(text, True, color)
    surface.blit(image, image.get_rect(center=(int(center[0]), int(center[1]))))


def _text_left(surface: pygame.Surface, font: pygame.font.Font, text: str, x: float, y: float, color) -> None:
    image = font.render(text, True, color)
    surface.blit(image, (int(x), int(y)))


def describe_event(event: dict[str, Any]) -> str | None:
    """One readable line for an engine event, or ``None`` for bookkeeping noise."""
    kind = event.get("type")
    if kind == "attack":
        return f"{event['attacker']} attacks {event['defender']} for {event['damage']} (hp {event['defender_hp']})"
    if kind == "counterattack":
        return f"{event['attacker']} counterattacks {event['defender']} for {event['damage']}"
    if kind == "destroyed":
        return f"{event['unit']} destroyed ({event.get('cause')})"
    if kind == "captured":
        return f"site {event.get('site')} captured by {event.get('owner')}"
    if kind == "objective_captured":
        return f"OBJECTIVE {event.get('site')} taken - {event.get('winner')} wins"
    if kind == "build":
        return f"{event.get('unit')} built at {event.get('site')} for {event.get('cost')}"
    if kind == "out_of_fuel":
        return f"{event.get('unit')} out of fuel (-{event.get('damage')} hp)"
    if kind == "eliminated":
        return f"player {event.get('player')} eliminated"
    return None


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------


def draw_frame(
    surface: pygame.Surface,
    state: GameState,
    camera: Camera,
    controller: Controller,
    fonts: Fonts,
    *,
    message: str = "",
    log_lines: Iterable[str] = (),
    show_log: bool = True,
    show_coords: bool = False,
    pause_menu: Menu | None = None,
    tooltip_lines: list[str] | None = None,
    tooltip_pos: tuple[int, int] | None = None,
) -> None:
    surface.fill(palette.BACKGROUND)
    _draw_terrain(surface, state, camera, fonts, show_coords)
    _draw_capture_progress(surface, state, camera)
    _draw_build_sites(surface, state, camera)
    _draw_action_fills(surface, state, camera, controller)
    _draw_move_boundary(surface, state, camera, controller)
    _draw_attack_boundary(surface, state, camera, controller)
    _draw_units(surface, state, camera, controller, fonts)
    _draw_highlights(surface, state, camera, controller)
    _draw_cursor_and_path(surface, state, camera, controller)
    panel = _draw_side_panel(surface, state, controller, fonts)
    _draw_hud(surface, state, fonts)
    _draw_toolbar(surface, fonts)
    if show_log:
        _draw_log(surface, log_lines, fonts, panel)
    if message:
        _draw_message(surface, message, fonts)
    if controller.mode == MODE_STATUS and controller.selected:
        _draw_status(surface, state, controller, fonts)
    if state.status != "ongoing":
        _draw_game_over(surface, state, fonts)
    if pause_menu is not None:
        draw_modal_menu(surface, pause_menu, fonts, "paused")
    if tooltip_lines:
        _draw_tooltip(surface, tooltip_lines, tooltip_pos or (0, 0), fonts)


def _draw_terrain(
    surface: pygame.Surface,
    state: GameState,
    camera: Camera,
    fonts: Fonts,
    show_coords: bool,
) -> None:
    ids = state.player_ids()
    for pos, tile in sorted(state.map.tiles.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        center = camera.to_screen(pos)
        corners = hex_corners(center, camera.size)
        color = palette.terrain_color(tile.terrain)
        if tile.owner:
            color = palette.mix(color, palette.player_color(tile.owner, ids), 0.38)
        pygame.draw.polygon(surface, color, corners)
        pygame.draw.polygon(surface, palette.GRID, corners, 1)
        if show_coords:
            col, row = axial_to_offset(pos)
            _text(surface, fonts.tiny, f"{col},{row}", center, (20, 20, 20))


def _draw_capture_progress(surface: pygame.Surface, state: GameState, camera: Camera) -> None:
    for tile in state.map.tiles.values():
        if not tile.capture_points or not tile.capture_progress:
            continue
        center = camera.to_screen(tile.pos)
        ratio = min(1.0, tile.capture_progress / tile.capture_points)
        width = int(camera.size * 1.3)
        bar = pygame.Rect(int(center[0] - width / 2), int(center[1] + camera.size * 0.72), width, 5)
        pygame.draw.rect(surface, (20, 20, 20), bar)
        filled = pygame.Rect(bar.x, bar.y, int(width * ratio), bar.height)
        pygame.draw.rect(surface, palette.CAPTURE_HIGHLIGHT, filled)


def _draw_build_sites(surface: pygame.Surface, state: GameState, camera: Camera) -> None:
    buildable = {tuple(option["site"]) for option in rules.build_options(state, state.current_player)}
    for site in state.build_sites(state.current_player):
        if tuple(site.pos) not in buildable:
            continue
        center = camera.to_screen(site.pos)
        size = camera.size * 0.16
        rect = pygame.Rect(int(center[0] - size), int(center[1] - size), int(size * 2), int(size * 2))
        pygame.draw.rect(surface, palette.LOAD_HIGHLIGHT, rect, 2)


def _draw_action_fills(surface: pygame.Surface, state: GameState, camera: Camera, controller: Controller) -> None:
    options = controller.selection_options(state)
    if options is None:
        return
    mode = controller.mode
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    if mode in (MODE_MENU, MODE_MOVE):
        for move in options["moves"]:
            corners = hex_corners(camera.to_screen(tuple(move["pos"])), camera.size * 0.96)
            pygame.draw.polygon(overlay, (*palette.MOVE_HIGHLIGHT, 70), corners)
    if mode in (MODE_MENU, MODE_LOAD):
        for load in options["loads"]:
            corners = hex_corners(camera.to_screen(tuple(load["pos"])), camera.size * 0.96)
            pygame.draw.polygon(overlay, (*palette.LOAD_HIGHLIGHT, 80), corners)
    if mode in (MODE_MENU, MODE_UNLOAD):
        for unload in options["unloads"]:
            corners = hex_corners(camera.to_screen(tuple(unload["at"])), camera.size * 0.96)
            pygame.draw.polygon(overlay, (*palette.LOAD_HIGHLIGHT, 80), corners)
    if mode in (MODE_MENU, MODE_ATTACK) and options["captures"]:
        selected = state.units.get(controller.selected)
        if selected is not None and selected.pos is not None:
            corners = hex_corners(camera.to_screen(selected.pos), camera.size * 0.96)
            pygame.draw.polygon(overlay, (*palette.CAPTURE_HIGHLIGHT, 60), corners)
    surface.blit(overlay, (0, 0))


def _draw_move_boundary(surface: pygame.Surface, state: GameState, camera: Camera, controller: Controller) -> None:
    """Stroke the perimeter of the selected unit's whole movement range."""
    if controller.mode not in (MODE_MENU, MODE_MOVE):
        return
    options = controller.selection_options(state)
    if options is None or not options["moves"]:
        return
    region = controller.move_region(state)
    if not region:
        return
    segments = reachable_boundary(region, camera.size, offset=(camera.pan_x, camera.pan_y))
    for start, end in segments:
        pygame.draw.line(surface, palette.MOVE_HIGHLIGHT, start, end, 4)


def _draw_attack_boundary(surface: pygame.Surface, state: GameState, camera: Camera, controller: Controller) -> None:
    """Stroke the perimeter of every hex the selected unit's weapons can reach."""
    if controller.mode == MODE_ATTACK:
        pass
    elif controller.mode == MODE_MENU:
        item = controller.menu_current(state)
        if item is None or item.key != "attack" or not item.enabled:
            return
    else:
        return
    region = controller.attack_region(state)
    if not region:
        return
    segments = reachable_boundary(region, camera.size, offset=(camera.pan_x, camera.pan_y))
    for start, end in segments:
        pygame.draw.line(surface, palette.ATTACK_HIGHLIGHT, start, end, 3)


def _draw_cursor_and_path(
    surface: pygame.Surface, state: GameState, camera: Camera, controller: Controller
) -> None:
    """The keyboard cursor as a white hex, with the engine's move path to it."""
    if controller.mode not in (MODE_IDLE, MODE_MENU, MODE_MOVE, MODE_ATTACK, MODE_LOAD, MODE_UNLOAD):
        return
    pos = controller.cursor_hex()
    if pos is None or not state.map.in_bounds(pos):
        return
    center = camera.to_screen(pos)
    pygame.draw.polygon(surface, palette.SELECT_HIGHLIGHT, hex_corners(center, camera.size * 0.9), 3)

    if controller.mode not in (MODE_MENU, MODE_MOVE):
        return
    path = controller.path_to(state, pos)
    if not path or len(path) < 2:
        return
    points = [camera.to_screen(tuple(step)) for step in path]
    for start, end in zip(points, points[1:]):
        pygame.draw.line(surface, palette.SELECT_HIGHLIGHT, start, end, 3)
    for point in points[1:-1]:
        pygame.draw.circle(surface, palette.SELECT_HIGHLIGHT, (int(point[0]), int(point[1])), 3)
    arrow_head(surface, points[-2], points[-1], palette.SELECT_HIGHLIGHT)


def arrow_head(surface: pygame.Surface, start, end, color) -> None:
    """A small filled triangle at ``end`` pointing away from ``start``."""
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 10
    left = (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45))
    right = (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45))
    pygame.draw.polygon(surface, color, [end, left, right])


def _unit_centers(state: GameState, pos: Hex, camera: Camera) -> dict[str, tuple[float, float]]:
    """Screen centres for every unit on a hex.

    A single unit sits in the middle.  When an aircraft shares a hex with a
    surface unit, the surface unit is drawn slightly low and the aircraft above
    it, so both remain visible.
    """
    center = camera.to_screen(pos)
    units = state.units_at(pos)
    if len(units) <= 1:
        return {unit.id: center for unit in units}

    air = [unit for unit in units if state.spec_of(unit).domain == "air"]
    surface = [unit for unit in units if state.spec_of(unit).domain != "air"]
    if len(air) == 1 and len(surface) == 1:
        offset = camera.size * 0.20
        return {
            surface[0].id: (center[0], center[1] + offset),
            air[0].id: (center[0], center[1] - offset),
        }

    # Defensive fallback for an impossible stack authored outside the rules:
    # fan the units out so none is completely hidden.
    radius = camera.size * 0.30
    return {
        unit.id: (
            center[0] + radius * math.cos(2 * math.pi * index / len(units)),
            center[1] + radius * math.sin(2 * math.pi * index / len(units)),
        )
        for index, unit in enumerate(units)
    }


def _draw_units(
    surface: pygame.Surface,
    state: GameState,
    camera: Camera,
    controller: Controller,
    fonts: Fonts,
) -> None:
    ids = state.player_ids()
    # Draw surface units first and aircraft last, so the air layer stays on top.
    for unit in sorted(
        state.units.values(),
        key=lambda u: (state.spec_of(u).domain == "air", u.id),
    ):
        if unit.pos is None:
            continue
        centers = _unit_centers(state, unit.pos, camera)
        center = centers.get(unit.id) or camera.to_screen(unit.pos)
        color = palette.player_color(unit.owner, ids)
        if unit.acted or unit.loaded:
            color = palette.shade(color, 0.55)
        stacked = len(state.units_at(unit.pos)) > 1
        radius = max(6, int(camera.size * (0.36 if stacked else 0.46)))
        spec = state.spec_of(unit)
        pygame.draw.circle(surface, color, (int(center[0]), int(center[1])), radius)
        pygame.draw.circle(surface, palette.GRID, (int(center[0]), int(center[1])), radius, 2)
        _text(surface, fonts.glyph, spec.glyph, center, (245, 245, 245))
        _draw_hp_bar(surface, center, camera, unit.hp, spec.max_hp)
        if spec.weapons and not any(unit.ammo.get(weapon.id, 0) > 0 for weapon in spec.weapons):
            pygame.draw.circle(surface, palette.ATTACK_HIGHLIGHT, (int(center[0] - radius), int(center[1] - radius)), 4)
        if unit.cargo:
            _text(surface, fonts.tiny, str(len(unit.cargo)), (center[0] + radius, center[1] + radius), palette.TEXT)
        if unit.deployed_this_turn:
            pygame.draw.circle(surface, palette.TEXT, (int(center[0]), int(center[1])), radius + 2, 1)


def _draw_hp_bar(surface: pygame.Surface, center: tuple[float, float], camera: Camera, hp: int, max_hp: int) -> None:
    width = max(12, int(camera.size * 1.2))
    bar = pygame.Rect(int(center[0] - width / 2), int(center[1] + camera.size * 0.5), width, 4)
    pygame.draw.rect(surface, (16, 16, 16), bar)
    filled = pygame.Rect(bar.x, bar.y, int(width * max(0, min(1.0, hp / max_hp))), bar.height)
    pygame.draw.rect(surface, palette.hp_color(hp, max_hp), filled)


def _draw_highlights(surface: pygame.Surface, state: GameState, camera: Camera, controller: Controller) -> None:
    options = controller.selection_options(state)
    if options is not None and controller.mode in (MODE_MENU, MODE_ATTACK):
        targets = {option["target"] for option in options["attacks"]}
        for target in targets:
            unit = state.units.get(target)
            if unit is None or unit.pos is None:
                continue
            center = _unit_centers(state, unit.pos, camera).get(unit.id) or camera.to_screen(unit.pos)
            radius = int(camera.size * 0.5)
            pygame.draw.circle(surface, palette.ATTACK_HIGHLIGHT, (int(center[0]), int(center[1])), radius, 3)
    if controller.selected is not None and controller.selected in state.units:
        unit = state.units[controller.selected]
        if unit.pos is not None:
            center = _unit_centers(state, unit.pos, camera).get(unit.id) or camera.to_screen(unit.pos)
            radius = int(camera.size * 0.5)
            pygame.draw.circle(surface, palette.SELECT_HIGHLIGHT, (int(center[0]), int(center[1])), radius, 2)


# ---------------------------------------------------------------------------
# HUD, panels, overlays
# ---------------------------------------------------------------------------


def _draw_hud(surface: pygame.Surface, state: GameState, fonts: Fonts) -> None:
    width = surface.get_width()
    bar = pygame.Rect(0, 0, width, HUD_HEIGHT)
    pygame.draw.rect(surface, palette.PANEL, bar)
    pygame.draw.line(surface, palette.PANEL_EDGE, (0, HUD_HEIGHT), (width, HUD_HEIGHT))
    player = state.players.get(state.current_player)
    name = player.name if player else state.current_player
    funds = player.funds if player else 0
    income = rules.turn_income(state, state.current_player) if player else 0
    text = f"Turn {state.turn_number}  |  {state.current_player} {name}  |  funds {funds}  |  income {income}/turn"
    if state.status != "ongoing":
        text += f"  |  {state.status}" + (f" winner {state.winner}" if state.winner else "")
    _text_left(surface, fonts.hud, text, 12, 10, palette.TEXT)


def _draw_side_panel(surface: pygame.Surface, state: GameState, controller: Controller, fonts: Fonts) -> int:
    """Draw the selection panel (unit + action menu, or build menu)."""
    if controller.selected is None and controller.build_site is None:
        return surface.get_width()
    width = surface.get_width()
    panel = pygame.Rect(width - PANEL_WIDTH, HUD_HEIGHT, PANEL_WIDTH, surface.get_height() - HUD_HEIGHT)
    pygame.draw.rect(surface, palette.PANEL, panel)
    pygame.draw.line(surface, palette.PANEL_EDGE, (panel.x, panel.y), (panel.x, panel.bottom))
    x = panel.x + 12
    y = panel.y + 10

    unit = state.units.get(controller.selected) if controller.selected else None
    if unit is not None:
        spec = state.spec_of(unit)
        _text_left(surface, fonts.body, f"{unit.id}  {spec.name}", x, y, palette.TEXT)
        y += 26
        for line in _unit_description_lines(state, unit):
            _text_left(surface, fonts.tiny, line, x, y, palette.TEXT_DIM)
            y += 17
        y += 10

    if controller.build_site is not None:
        _draw_build_menu(surface, state, controller, fonts, panel, y)
        return panel.x

    if controller.mode == MODE_STATUS:
        return panel.x

    if controller.mode == MODE_WEAPON:
        _draw_weapon_menu(surface, state, controller, fonts, panel, y)
        return panel.x

    _draw_action_menu(surface, state, controller, fonts, panel, y)
    return panel.x


def _draw_build_menu(surface, state, controller, fonts, panel, y) -> None:
    x = panel.x + 12
    site = tuple(controller.build_site)
    options = controller.build_options_at(state, site)
    _text_left(surface, fonts.body, f"Build at {list(site)}", x, y, palette.TEXT)
    y += 26
    if not options:
        _text_left(surface, fonts.small, "nothing can be built here", x, y, palette.TEXT_DIM)
        return
    for index, option in enumerate(options, start=1):
        color = palette.TEXT if option["affordable"] else palette.TEXT_DIM
        suffix = "" if option["affordable"] else "  (no funds)"
        _text_left(surface, fonts.small, f"{index}. {option['unit']:<16} {option['cost']}{suffix}", x, y, color)
        y += 22
    _text_left(surface, fonts.tiny, "press a number to build, Esc to close", x, y + 6, palette.TEXT_DIM)


def _draw_action_menu(surface, state, controller, fonts, panel, y) -> None:
    items = controller.action_items(state)
    if not items:
        return
    x = panel.x + 12
    _text_left(surface, fonts.small, "orders", x, y, palette.TEXT)
    y += 22
    for index, item in enumerate(items):
        row = pygame.Rect(panel.x + 6, y - 2, PANEL_WIDTH - 12, 24)
        if index == controller.menu_index:
            pygame.draw.rect(surface, palette.mix(palette.PANEL, palette.MOVE_HIGHLIGHT, 0.5), row)
        if item.enabled:
            color = palette.SELECT_HIGHLIGHT if index == controller.menu_index else palette.TEXT
        else:
            color = palette.TEXT_DIM
        prefix = "> " if index == controller.menu_index else "  "
        _text_left(surface, fonts.small, prefix + item.label, x, y, color)
        y += 24
    _text_left(surface, fonts.tiny, "W/S choose, F use, or click a row", x, y + 4, palette.TEXT_DIM)


def _draw_weapon_menu(surface, state, controller, fonts, panel, y) -> None:
    target = state.units.get(controller.weapon_target) if controller.weapon_target else None
    title = f"Attack {target.id}" if target is not None else "Attack"
    x = panel.x + 12
    _text_left(surface, fonts.small, title, x, y, palette.TEXT)
    y += 26
    for index, item in enumerate(controller.weapon_menu_items(state)):
        row = pygame.Rect(panel.x + 6, y - 2, PANEL_WIDTH - 12, 24)
        if index == controller.weapon_index:
            pygame.draw.rect(surface, palette.mix(palette.PANEL, palette.ATTACK_HIGHLIGHT, 0.45), row)
        color = palette.SELECT_HIGHLIGHT if index == controller.weapon_index else palette.TEXT
        _text_left(surface, fonts.small, item.label, x, y, color)
        _text_left(surface, fonts.tiny, item.hint, x + 110, y + 2, palette.TEXT_DIM)
        y += 24
    _text_left(surface, fonts.tiny, "W/S choose, Enter/F fire, Esc back", x, y + 4, palette.TEXT_DIM)


def _unit_header_height(state: GameState, unit) -> int:
    lines = _unit_description_lines(state, unit) or [""]
    return 10 + 26 + 17 * len(lines) + 10


def _unit_description_lines(state: GameState, unit) -> list[str]:
    lines: list[str] = []
    for line in view.describe_unit(state, unit.id).splitlines():
        lines.extend(_wrap(line, 44))
    return lines


def action_menu_row_rects(surface_size: tuple[int, int], state: GameState, controller: Controller) -> list[pygame.Rect]:
    """Client rectangles of the order rows, for mouse hit-testing."""
    if controller.selected is None or controller.build_site is not None:
        return []
    unit = state.units.get(controller.selected)
    if unit is None:
        return []
    width, _ = surface_size
    top = HUD_HEIGHT + _unit_header_height(state, unit) + 22
    return [
        pygame.Rect(width - PANEL_WIDTH + 6, top + index * 24 - 2, PANEL_WIDTH - 12, 24)
        for index in range(len(controller.action_items(state)))
    ]


def weapon_menu_row_rects(surface_size: tuple[int, int], state: GameState, controller: Controller) -> list[pygame.Rect]:
    """Client rectangles of the weapon-picker rows, for mouse hit-testing."""
    if controller.selected is None or controller.mode != MODE_WEAPON:
        return []
    unit = state.units.get(controller.selected)
    if unit is None:
        return []
    width, _ = surface_size
    top = HUD_HEIGHT + _unit_header_height(state, unit) + 26
    return [
        pygame.Rect(width - PANEL_WIDTH + 6, top + index * 24 - 2, PANEL_WIDTH - 12, 24)
        for index in range(len(controller.weapon_menu_items(state)))
    ]


def build_menu_row_rects(surface_size: tuple[int, int], state: GameState, controller: Controller) -> list[pygame.Rect]:
    """Client rectangles of the build-menu rows, for mouse hit-testing."""
    if controller.build_site is None:
        return []
    unit = state.units.get(controller.selected) if controller.selected else None
    header = _unit_header_height(state, unit) if unit is not None else 10
    width, _ = surface_size
    top = HUD_HEIGHT + header + 26
    options = controller.build_options_at(state, controller.build_site)
    return [
        pygame.Rect(width - PANEL_WIDTH + 6, top + index * 22 - 2, PANEL_WIDTH - 12, 22)
        for index in range(len(options))
    ]


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_log(surface: pygame.Surface, log_lines: Iterable[str], fonts: Fonts, panel_x: int) -> None:
    lines = list(log_lines)[-LOG_LINES:]
    if not lines:
        return
    height = HUD_HEIGHT + 8 + LOG_LINES * 18 + 8
    left = 8 + buttons.TOOLBAR_WIDTH
    available = max(140, panel_x - left - 8)
    box = pygame.Rect(left, surface.get_height() - height, min(available, 520), height - HUD_HEIGHT)
    overlay = pygame.Surface(box.size, pygame.SRCALPHA)
    overlay.fill((12, 12, 14, 170))
    surface.blit(overlay, box.topleft)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 1)
    y = box.y + 6
    for line in lines:
        _text_left(surface, fonts.tiny, line, box.x + 8, y, palette.TEXT_DIM)
        y += 18


def _draw_message(surface: pygame.Surface, message: str, fonts: Fonts) -> None:
    width = surface.get_width()
    image = fonts.body.render(message, True, palette.TEXT)
    box = pygame.Rect(0, HUD_HEIGHT + 6, min(width - 24, image.get_width() + 24), 30)
    box.centerx = width // 2
    pygame.draw.rect(surface, palette.PANEL, box)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 1)
    surface.blit(image, image.get_rect(center=box.center))


# ---------------------------------------------------------------------------
# icon toolbar and hover tooltip
# ---------------------------------------------------------------------------


def _draw_toolbar(surface: pygame.Surface, fonts: Fonts) -> None:
    for button in buttons.toolbar_layout(HUD_HEIGHT, surface.get_height()):
        rect = pygame.Rect(button.x, button.y, button.w, button.h)
        pygame.draw.rect(surface, palette.PANEL, rect)
        pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)
        _draw_icon(surface, button.id, rect, palette.TEXT)


def _draw_icon(surface: pygame.Surface, icon: str, rect: pygame.Rect, color) -> None:
    """Simple vector icons: no image assets, so the UI stays dependency-light."""
    cx, cy = rect.center
    s = rect.width // 2 - 9
    if icon == "menu":
        for offset in (-6, 0, 6):
            pygame.draw.line(surface, color, (cx - s, cy + offset), (cx + s, cy + offset), 2)
    elif icon == "end_turn":
        pygame.draw.polygon(surface, color, [(cx - 7, cy - 8), (cx + 5, cy), (cx - 7, cy + 8)])
        pygame.draw.line(surface, color, (cx + 8, cy - 8), (cx + 8, cy + 8), 3)
    elif icon == "units":
        pygame.draw.circle(surface, color, (cx - 5, cy - 5), 4, 1)
        pygame.draw.circle(surface, color, (cx + 5, cy - 3), 3, 1)
        pygame.draw.arc(surface, color, pygame.Rect(cx - 11, cy - 1, 13, 12), 3.5, 6.0, 2)
        pygame.draw.arc(surface, color, pygame.Rect(cx - 1, cy, 12, 11), 3.5, 6.0, 2)
    elif icon == "build":
        pygame.draw.rect(surface, color, pygame.Rect(cx - 9, cy - 2, 18, 11), 2)
        pygame.draw.line(surface, color, (cx, cy - 10), (cx, cy + 4), 2)
        pygame.draw.line(surface, color, (cx - 5, cy - 7), (cx + 5, cy - 7), 2)
    elif icon == "save":
        pygame.draw.rect(surface, color, pygame.Rect(cx - 9, cy - 9, 18, 18), 2)
        pygame.draw.rect(surface, color, pygame.Rect(cx - 4, cy - 9, 8, 6))
        pygame.draw.rect(surface, color, pygame.Rect(cx - 5, cy + 1, 10, 8), 1)
    elif icon == "load":
        pygame.draw.polygon(
            surface,
            color,
            [(cx - 10, cy - 5), (cx - 3, cy - 5), (cx - 1, cy - 8), (cx + 10, cy - 8), (cx + 10, cy + 8), (cx - 10, cy + 8)],
            2,
        )
        pygame.draw.line(surface, color, (cx - 10, cy - 1), (cx + 10, cy - 1), 2)
    elif icon == "log":
        for offset in (-6, 0, 6):
            pygame.draw.line(surface, color, (cx - 9, cy + offset), (cx + 9, cy + offset), 2)
    elif icon == "coords":
        for offset in (-6, 0, 6):
            pygame.draw.line(surface, color, (cx - 9, cy + offset), (cx + 9, cy + offset), 1)
            pygame.draw.line(surface, color, (cx + offset, cy - 9), (cx + offset, cy + 9), 1)
    elif icon in ("zoom_in", "zoom_out"):
        pygame.draw.circle(surface, color, (cx - 2, cy - 2), 7, 2)
        pygame.draw.line(surface, color, (cx + 3, cy + 3), (cx + 9, cy + 9), 2)
        pygame.draw.line(surface, color, (cx - 6, cy - 2), (cx + 2, cy - 2), 2)
        if icon == "zoom_in":
            pygame.draw.line(surface, color, (cx - 2, cy - 6), (cx - 2, cy + 2), 2)
    else:  # pragma: no cover - every toolbar id is drawn above
        _text(surface, pygame.font.Font(None, 14), icon[:2], (cx, cy), color)


def _draw_tooltip(surface: pygame.Surface, lines: list[str], pos: tuple[int, int], fonts: Fonts) -> None:
    rendered = [fonts.tiny.render(line, True, palette.TEXT) if line else None for line in lines]
    width = max([image.get_width() for image in rendered if image is not None] or [0]) + 16
    height = len(rendered) * 16 + 12
    x, y = pos[0] + 18, pos[1] + 18
    if x + width > surface.get_width() - 8:
        x = pos[0] - width - 12
    if y + height > surface.get_height() - 8:
        y = surface.get_height() - height - 8
    x = max(8, x)
    y = max(HUD_HEIGHT + 8, y)
    box = pygame.Rect(x, y, width, height)
    pygame.draw.rect(surface, palette.PANEL, box)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 1)
    yy = y + 6
    for image in rendered:
        if image is not None:
            surface.blit(image, (x + 8, yy))
        yy += 16


def _draw_status(surface: pygame.Surface, state: GameState, controller: Controller, fonts: Fonts) -> None:
    """A modal status sheet for the selected unit: stats plus every legal order."""
    unit = state.units.get(controller.selected)
    if unit is None:
        return
    lines: list[str] = [view.describe_unit(state, unit.id), "", "legal orders:"]
    lines.extend(view.unit_options_text(state, unit.id).splitlines()[1:])
    rendered = [line for entry in lines for line in (entry.splitlines() or [""])]
    if not rendered:
        rendered = ["(no status)"]
    width = min(surface.get_width() - 80, 720)
    height = min(surface.get_height() - 120, 40 + len(rendered) * 18)
    box = pygame.Rect(0, 0, width, height)
    box.center = (surface.get_width() // 2, surface.get_height() // 2)
    pygame.draw.rect(surface, palette.PANEL, box)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 2)
    _text_left(surface, fonts.body, f"STATUS  {unit.id}", box.x + 14, box.y + 10, palette.TEXT)
    y = box.y + 40
    for line in rendered:
        lines_out = _wrap(line, 86)
        for piece in lines_out:
            if y + 16 > box.bottom:
                return
            _text_left(surface, fonts.tiny, piece, box.x + 14, y, palette.TEXT_DIM)
            y += 18
    _text_left(surface, fonts.tiny, "click or Esc to close", box.x + 14, box.bottom - 20, palette.TEXT_DIM)


def _draw_game_over(surface: pygame.Surface, state: GameState, fonts: Fonts) -> None:
    if state.status == "victory" and state.winner:
        text = f"Player {state.winner} wins!"
    else:
        text = state.status.capitalize()
    center = (surface.get_width() // 2, surface.get_height() // 2)
    image = fonts.big.render(text, True, palette.TEXT)
    box = image.get_rect(center=center).inflate(60, 40)
    pygame.draw.rect(surface, palette.PANEL, box)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 2)
    surface.blit(image, image.get_rect(center=center))


# ---------------------------------------------------------------------------
# menu screens
# ---------------------------------------------------------------------------


def menu_item_rects(surface_size: tuple[int, int], menu: Menu) -> list[tuple[int, pygame.Rect]]:
    """``(item_index, clickable rect)`` for the menu's visible window.

    Only the rows of ``menu.window()`` are returned, so a menu longer than the
    window scrolls instead of drawing past the screen edge.  Carrying the real
    item index keeps click routing correct while scrolled.
    """
    width, height = surface_size
    first, stop = menu.window()
    count = stop - first
    total = count * MENU_ITEM_HEIGHT
    top = height // 2 - total // 2
    rects: list[tuple[int, pygame.Rect]] = []
    for row, index in enumerate(range(first, stop)):
        rect = pygame.Rect(0, 0, MENU_WIDTH, MENU_ITEM_HEIGHT - 8)
        rect.centerx = width // 2
        rect.y = top + row * MENU_ITEM_HEIGHT
        rects.append((index, rect))
    return rects


def _draw_menu_row(surface: pygame.Surface, menu: Menu, fonts: "Fonts", index: int, rect: pygame.Rect) -> None:
    item = menu.items[index]
    selected = index == menu.index
    background = palette.PANEL if not selected else palette.mix(palette.PANEL, palette.MOVE_HIGHLIGHT, 0.5)
    pygame.draw.rect(surface, background, rect)
    pygame.draw.rect(surface, palette.PANEL_EDGE, rect, 1)
    if not item.enabled:
        color = palette.TEXT_DIM
    elif selected:
        color = palette.SELECT_HIGHLIGHT
    else:
        color = palette.TEXT
    if item.hint:
        _text(surface, fonts.body, item.label, (rect.centerx, rect.centery - 8), color)
        _text(surface, fonts.tiny, item.hint, (rect.centerx, rect.bottom - 8), palette.TEXT_DIM)
    else:
        _text(surface, fonts.body, item.label, rect.center, color)


def _draw_preview(
    surface: pygame.Surface,
    menu: Menu,
    preview: Iterable[str],
    *,
    cell: int = 8,
) -> None:
    """Draw a terrain-coloured thumbnail to the right of a centred menu."""
    rows = list(preview)
    if not rows or not rows[0]:
        return
    width, height = surface.get_size()
    columns = len(rows[0])
    rects = menu_item_rects((width, height), menu)
    if not rects:
        return
    menu_right = rects[0][1].right
    menu_top = rects[0][1].top
    menu_bottom = rects[-1][1].bottom
    x0 = menu_right + 40
    available = width - 20 - x0
    if available < columns * 3:
        return
    cell = max(3, min(cell, available // columns, max(3, (menu_bottom - menu_top) // max(1, len(rows)))))
    y0 = (menu_top + menu_bottom) // 2 - len(rows) * cell // 2
    for row_index, row in enumerate(rows):
        for col_index, code in enumerate(row):
            color = palette.terrain_color(code)
            pygame.draw.rect(
                surface,
                color,
                (x0 + col_index * cell, y0 + row_index * cell, cell - 1, cell - 1),
            )
    pygame.draw.rect(
        surface,
        palette.PANEL_EDGE,
        (x0 - 2, y0 - 2, columns * cell + 2, len(rows) * cell + 2),
        1,
    )


def _wrap_text(text: str, max_chars: int, max_lines: int = 3) -> list[str]:
    """Split ``text`` into at most ``max_lines`` lines, ellipsising the tail."""
    lines: list[str] = []
    remaining = " ".join(text.split())
    while remaining and len(lines) < max_lines:
        if len(remaining) <= max_chars:
            lines.append(remaining)
            remaining = ""
            break
        cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        lines.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].strip()
    if remaining and lines:
        lines[-1] = lines[-1][: max(0, max_chars - 3)].rstrip() + "..."
    return lines


def draw_menu_screen(
    surface: pygame.Surface,
    menu: Menu,
    fonts: "Fonts",
    *,
    subtitle: str = "",
    detail_lines: Iterable[str] = (),
    preview: Iterable[str] = (),
) -> None:
    surface.fill(palette.BACKGROUND)
    _text(surface, fonts.big, menu.title, (surface.get_width() // 2, surface.get_height() // 2 - 150), palette.TEXT)
    if subtitle:
        _text(surface, fonts.small, subtitle, (surface.get_width() // 2, surface.get_height() // 2 - 105), palette.TEXT_DIM)
    rects = menu_item_rects(surface.get_size(), menu)
    for index, rect in rects:
        _draw_menu_row(surface, menu, fonts, index, rect)
    _draw_preview(surface, menu, preview)

    detail: list[str] = []
    for line in detail_lines:
        detail.extend(_wrap_text(line, max(40, int(surface.get_width() / 7.0))))
    menu_bottom = rects[-1][1].bottom if rects else surface.get_height() // 2
    for line_index, line in enumerate(detail[:3]):
        _text(
            surface,
            fonts.tiny,
            line,
            (surface.get_width() // 2, menu_bottom + 18 + line_index * 16),
            palette.TEXT_DIM,
        )

    first, stop = menu.window()
    instruction = "arrows move, enter chooses, Esc back"
    if stop - first < len(menu.items):
        instruction += f"   [{first + 1}-{stop} of {len(menu.items)}]"
    _text(
        surface,
        fonts.tiny,
        instruction,
        (surface.get_width() // 2, surface.get_height() - 40),
        palette.TEXT_DIM,
    )


def draw_thinking_overlay(
    surface: pygame.Surface, fonts: Fonts, title: str, subtitle: str = ""
) -> None:
    """Dim the board while the local LLM is planning."""
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))
    box = pygame.Rect(0, 0, MENU_WIDTH + 140, 120)
    box.center = (surface.get_width() // 2, surface.get_height() // 2)
    pygame.draw.rect(surface, palette.PANEL, box)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 2)
    dots = "." * (1 + (pygame.time.get_ticks() // 500) % 3)
    _text(surface, fonts.body, title + dots, (box.centerx, box.y + 38), palette.TEXT)
    if subtitle:
        _text(surface, fonts.tiny, subtitle, (box.centerx, box.y + 76), palette.TEXT_DIM)
    _text(
        surface,
        fonts.tiny,
        "Esc or M opens the menu (take over for the LLM there)",
        (surface.get_width() // 2, box.bottom + 24),
        palette.TEXT_DIM,
    )


def draw_modal_menu(surface: pygame.Surface, menu: Menu, fonts: Fonts, title: str) -> None:
    """Draw a menu on top of the board, dimming it first."""
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))
    rects = menu_item_rects(surface.get_size(), menu)
    box = pygame.Rect(0, 0, MENU_WIDTH + 60, len(rects) * MENU_ITEM_HEIGHT + 90)
    box.center = (surface.get_width() // 2, surface.get_height() // 2)
    pygame.draw.rect(surface, palette.PANEL, box)
    pygame.draw.rect(surface, palette.PANEL_EDGE, box, 2)
    _text(surface, fonts.body, title, (box.centerx, box.y + 24), palette.TEXT)
    for index, rect in rects:
        _draw_menu_row(surface, menu, fonts, index, rect)
