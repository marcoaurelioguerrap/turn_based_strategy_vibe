"""The click/keyboard mode machine.  Pure: no pygame import.

Everything the player *decides* happens here, against the engine's own option
lists, so a click can only ever produce an order the engine already considers
legal.  The pygame layer merely turns screen points into hexes and hands them
over, and performs whatever order comes back.

Modes
-----
``idle``    nothing selected.
``menu``    a unit is selected and its action menu is open (Move / Attack /
            Capture / Load / Unload / Status / Hold / Cancel).  Clicking a
            highlighted tile still commits directly as a shortcut.
``move``/``attack``/``load``/``unload``
            a specific action was chosen from the menu; the next click must be
            one of that action's targets (or the mode returns to ``menu``).
``status``  the unit's full status sheet is shown; any click closes it.
``build``   a build-site menu is open.

``move_region`` returns the union of the unit's hex and every hex it can reach,
which the window strokes as the boundary of its movement range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tbs.engine import rules
from tbs.engine.hexgrid import Hex, axial_to_offset, hex_distance, hex_range, offset_to_axial
from tbs.engine.state import GameState
from tbs.ui.menu import MenuItem

MODE_IDLE = "idle"
MODE_MENU = "menu"
MODE_MOVE = "move"
MODE_ATTACK = "attack"
MODE_LOAD = "load"
MODE_UNLOAD = "unload"
MODE_STATUS = "status"
MODE_BUILD = "build"
MODE_WEAPON = "weapon"

_TARGET_MODES = (MODE_MOVE, MODE_ATTACK, MODE_LOAD, MODE_UNLOAD)
_MENU_MODES = (MODE_IDLE, MODE_MENU)


@dataclass(frozen=True)
class Intent:
    """What a click or key resolved to.

    ``kind`` is one of ``"select"`` (selection changed), ``"order"`` (run
    ``order`` through ``rules.perform``), ``"build_menu"`` (open the build menu
    for ``site``), ``"cancel"``, ``"menu"`` or ``"noop"``.  ``note`` is
    display text.
    """

    kind: str
    order: dict[str, Any] | None = None
    site: Hex | None = None
    note: str = ""


def load_order(state: GameState, selected_id: str, other_id: str) -> dict[str, Any] | None:
    """A load order linking two friendly units, in whichever direction applies."""
    selected = state.units[selected_id]
    other = state.units[other_id]
    if state.spec_of(selected).is_transport:
        for option in rules.load_options(state, selected_id):
            if option["cargo"] == other_id:
                return {"action": "load", "unit": selected_id, "cargo": other_id}
    if state.spec_of(other).is_transport:
        for option in rules.load_options(state, other_id):
            if option["cargo"] == selected_id:
                return {"action": "load", "unit": other_id, "cargo": selected_id}
    return None


class Controller:
    """Selection state, the active mode and the pure click resolver."""

    def __init__(self) -> None:
        self.selected: str | None = None
        self.mode: str = MODE_IDLE
        self.build_site: Hex | None = None
        self.menu_index: int = 0
        self.cursor: tuple[int, int] | None = None  # authored offset (col, row)
        self.weapon_target: str | None = None
        self.weapon_choices: list[dict[str, Any]] = []
        self.weapon_index: int = 0

    # -- selection helpers --------------------------------------------------

    def clear(self) -> None:
        self.selected = None
        self.mode = MODE_IDLE
        self.build_site = None
        self.menu_index = 0
        self.weapon_target = None
        self.weapon_choices = []
        self.weapon_index = 0

    def select_unit(self, state: GameState, unit_id: str) -> None:
        self.selected = unit_id
        self.mode = MODE_MENU
        self.build_site = None
        self.menu_index = 0
        unit = state.units.get(unit_id)
        if unit is not None and unit.pos is not None:
            self.cursor = axial_to_offset(unit.pos)

    def back(self) -> None:
        """Leave a sub-mode: a target mode returns to the menu, else deselect."""
        if self.build_site is not None:
            self.build_site = None
            self.mode = MODE_IDLE
        elif self.mode == MODE_WEAPON:
            self.mode = MODE_MENU
            self.weapon_target = None
            self.weapon_choices = []
        elif self.mode in _TARGET_MODES or self.mode == MODE_STATUS:
            self.mode = MODE_MENU
        else:
            self.clear()

    def selection_options(self, state: GameState) -> dict[str, Any] | None:
        """The engine's option dict for the selected unit, or ``None``."""
        if self.selected is None or self.selected not in state.units:
            return None
        return rules.unit_options(state, self.selected)

    def move_region(self, state: GameState) -> set[Hex]:
        """The unit's hex plus every hex within its movement range."""
        if self.selected is None or self.selected not in state.units:
            return set()
        unit = state.units[self.selected]
        region: set[Hex] = set()
        if unit.pos is not None:
            region.add(tuple(unit.pos))
        for move in rules.legal_moves(state, self.selected):
            region.add(tuple(move["pos"]))
        return region

    def attack_region(self, state: GameState) -> set[Hex]:
        """Every hex the selected unit's *current* weapons could fire into.

        A weapon with no ammo contributes nothing, and a unit that has moved
        this turn only contributes its direct weapons (the engine's rule).  The
        result is what the window strokes as the attack-range boundary.
        """
        if self.selected is None or self.selected not in state.units:
            return set()
        unit = state.units[self.selected]
        if unit.pos is None:
            return set()
        region: set[Hex] = set()
        for weapon in state.spec_of(unit).weapons:
            if unit.ammo.get(weapon.id, 0) <= 0:
                continue
            if unit.moved and not weapon.is_direct:
                continue
            for pos in hex_range(unit.pos, weapon.max_range):
                if pos not in state.map.tiles:
                    continue
                distance = hex_distance(unit.pos, pos)
                if weapon.min_range <= distance <= weapon.max_range:
                    region.add(tuple(pos))
        return region

    # -- the keyboard cursor ------------------------------------------------

    def set_cursor(self, state: GameState, pos: Hex) -> None:
        if tuple(pos) in state.map.tiles:
            self.cursor = axial_to_offset(pos)

    def cursor_hex(self) -> Hex | None:
        if self.cursor is None:
            return None
        return offset_to_axial(*self.cursor)

    def move_cursor(self, state: GameState, dcol: int, drow: int) -> Hex | None:
        """Move the keyboard cursor one authored cell; it never leaves the map."""
        if self.cursor is None:
            if self.selected is not None and self.selected in state.units:
                unit = state.units[self.selected]
                if unit.pos is not None:
                    self.cursor = axial_to_offset(unit.pos)
            if self.cursor is None and state.map.tiles:
                tiles = sorted(state.map.tiles)
                self.cursor = axial_to_offset(tiles[len(tiles) // 2])
        if self.cursor is None:
            return None
        col, row = self.cursor
        candidate = (col + dcol, row + drow)
        if offset_to_axial(*candidate) in state.map.tiles:
            self.cursor = candidate
        return self.cursor_hex()

    def path_to(self, state: GameState, pos: Hex) -> list[list[int]] | None:
        """The engine's exact move path to ``pos``, if it is a legal destination."""
        if self.selected is None or self.selected not in state.units:
            return None
        for move in rules.legal_moves(state, self.selected):
            if tuple(move["pos"]) == tuple(pos):
                return [list(step) for step in move["path"]]
        return None

    # -- the unit roster ----------------------------------------------------

    def unit_menu_items(self, state: GameState) -> list[MenuItem]:
        """Every unit of the current player, for the roster menu."""
        items: list[MenuItem] = []
        for unit in state.units_of(state.current_player):
            spec = state.spec_of(unit)
            if unit.acted:
                status = "acted"
            elif unit.deployed_this_turn:
                status = "new"
            elif unit.loaded:
                status = f"in {unit.carried_by}"
            elif unit.moved:
                status = "moved"
            else:
                status = "ready"
            items.append(MenuItem(unit.id, f"{unit.id}  {spec.name}  {unit.hp} hp", True, status))
        return items

    # -- the weapon picker --------------------------------------------------

    def weapon_menu_items(self, state: GameState) -> list[MenuItem]:
        if self.mode != MODE_WEAPON or not self.weapon_choices:
            return []
        items: list[MenuItem] = []
        for index, option in enumerate(self.weapon_choices):
            counter = option.get("counterattack")
            hint = f"{option['preview_damage']} dmg" + (
                f", counter {counter['preview_damage']}" if counter else ", no counter"
            )
            items.append(MenuItem(str(index), option["weapon"], True, hint))
        return items

    def weapon_move(self, delta: int) -> None:
        if not self.weapon_choices:
            return
        self.weapon_index = (self.weapon_index + delta) % len(self.weapon_choices)

    def choose_weapon(self, state: GameState, index: int) -> Intent:
        if self.mode != MODE_WEAPON or not 0 <= index < len(self.weapon_choices):
            return Intent("noop")
        option = self.weapon_choices[index]
        unit_id = self.selected
        target = self.weapon_target
        self.clear()
        if unit_id is None or target is None:
            return Intent("noop")
        return Intent(
            "order",
            order={"action": "attack", "unit": unit_id, "target": target, "weapon": option["weapon"]},
            note=f"{unit_id} attacks {target} with {option['weapon']}",
        )

    # -- the action menu ----------------------------------------------------

    def action_items(self, state: GameState) -> list[MenuItem]:
        """The selected unit's orders, as menu entries (disabled when illegal)."""
        if self.selected is None or self.selected not in state.units:
            return []
        options = self.selection_options(state)
        if options is None:
            return []
        items = [
            MenuItem("move", f"Move ({len(options['moves'])})", bool(options["moves"])),
            MenuItem("attack", f"Attack ({len(options['attacks'])})", bool(options["attacks"])),
        ]
        if options["captures"]:
            items.append(MenuItem("capture", "Capture", True))
        if options["loads"]:
            items.append(MenuItem("load", f"Load ({len(options['loads'])})", True))
        if options["unloads"]:
            items.append(MenuItem("unload", f"Unload ({len(options['unloads'])})", True))
        items.append(MenuItem("status", "Status", True))
        items.append(MenuItem("hold", "Hold", bool(options["can_hold"])))
        items.append(MenuItem("cancel", "Cancel", True))
        return items

    def menu_move(self, state: GameState, delta: int) -> None:
        items = self.action_items(state)
        if not items:
            return
        size = len(items)
        for step in range(1, size + 1):
            index = (self.menu_index + delta * step) % size
            if items[index].enabled:
                self.menu_index = index
                return

    def menu_current(self, state: GameState) -> MenuItem | None:
        items = self.action_items(state)
        if not items:
            return None
        return items[min(self.menu_index, len(items) - 1)]

    def menu_choose(self, state: GameState) -> Intent:
        item = self.menu_current(state)
        if item is None or not item.enabled:
            return Intent("noop")
        if item.key == "move":
            self.mode = MODE_MOVE
            return Intent("noop", note="choose a destination")
        if item.key == "attack":
            self.mode = MODE_ATTACK
            return Intent("noop", note="choose a target")
        if item.key == "load":
            self.mode = MODE_LOAD
            return Intent("noop", note="choose a unit to load")
        if item.key == "unload":
            self.mode = MODE_UNLOAD
            return Intent("noop", note="choose a drop hex")
        if item.key == "status":
            self.mode = MODE_STATUS
            return Intent("noop", note=f"{self.selected} status")
        if item.key == "capture":
            unit_id = self.selected
            self.clear()
            return Intent("order", order={"action": "capture", "unit": unit_id}, note=f"{unit_id} captures")
        if item.key == "hold":
            order = self.hold_order(state)
            if order is None:
                return Intent("noop")
            return Intent("order", order=order, note=f"{order['unit']} holds")
        if item.key == "cancel":
            self.clear()
        return Intent("noop")

    # -- the click resolver -------------------------------------------------

    def click(self, state: GameState, pos: Hex) -> Intent:
        """Resolve one left-click on a map hex into an :class:`Intent`."""
        pos = tuple(pos)
        if state.status != "ongoing":
            return Intent("noop")
        if pos in state.map.tiles:
            self.cursor = axial_to_offset(pos)

        if self.mode == MODE_BUILD:
            if self._is_empty_build_site(state, pos, state.current_player):
                self.build_site = pos
                return Intent("build_menu", site=pos, note="choose a unit to build")
            self.back()
            return Intent("noop")

        if self.mode == MODE_STATUS:
            self.mode = MODE_MENU
            return Intent("noop")

        if self.mode == MODE_WEAPON:
            self.mode = MODE_MENU
            self.weapon_target = None
            self.weapon_choices = []
            return Intent("noop", note="weapon choice cancelled")

        self.build_site = None
        unit = state.unit_at(pos)
        player = state.current_player

        if self.selected is None:
            if unit is not None and unit.owner == player and not unit.loaded:
                self.select_unit(state, unit.id)
                return Intent("select", note=f"selected {unit.id}")
            if self.mode == MODE_IDLE and self._is_empty_build_site(state, pos, player):
                self.mode = MODE_BUILD
                self.build_site = pos
                return Intent("build_menu", site=pos, note="choose a unit to build")
            return Intent("noop")

        selected = state.units.get(self.selected)
        if selected is None or selected.loaded:
            self.clear()
            return Intent("noop")

        # A different friendly unit: load into/out of a transport if possible, else reselect.
        if unit is not None and unit.owner == player and unit.id != selected.id:
            if self.mode in (MODE_MENU, MODE_LOAD):
                order = load_order(state, selected.id, unit.id)
                if order is not None:
                    self.clear()
                    return Intent("order", order=order, note=f"{selected.id} loads {unit.id}")
            self.select_unit(state, unit.id)
            return Intent("select", note=f"selected {unit.id}")

        intent = self._commit(state, selected, pos)
        if intent is not None:
            return intent

        if self.mode in _TARGET_MODES:
            self.mode = MODE_MENU
            return Intent("noop", note="no legal target there")
        return Intent("noop")

    # -- keyboard helpers ---------------------------------------------------

    def hold_order(self, state: GameState) -> dict[str, Any] | None:
        if self.selected is None or self.selected not in state.units:
            return None
        unit = state.units[self.selected]
        self.clear()
        if unit.acted or unit.loaded or unit.deployed_this_turn:
            return None
        return {"action": "hold", "unit": unit.id}

    def cycle_ready(self, state: GameState) -> str | None:
        ready = [
            unit
            for unit in state.units_of(state.current_player)
            if not unit.acted and not unit.loaded and not unit.deployed_this_turn and unit.pos is not None
        ]
        if not ready:
            self.clear()
            return None
        ids = [unit.id for unit in ready]
        if self.selected in ids:
            next_id = ids[(ids.index(self.selected) + 1) % len(ids)]
        else:
            next_id = ids[0]
        self.select_unit(state, next_id)
        return next_id

    def end_turn_order(self) -> dict[str, Any]:
        self.clear()
        return {"action": "end_turn"}

    # -- build menu ---------------------------------------------------------

    def open_build_menu(self, state: GameState, pos: Hex | None = None) -> Hex | None:
        if pos is not None and self._is_empty_build_site(state, pos, state.current_player):
            self.mode = MODE_BUILD
            self.build_site = tuple(pos)
            return self.build_site
        for site in state.build_sites(state.current_player):
            if state.unit_at(site.pos) is None:
                self.mode = MODE_BUILD
                self.build_site = site.pos
                return self.build_site
        self.build_site = None
        return None

    def build_options_at(self, state: GameState, site: Hex) -> list[dict[str, Any]]:
        return [o for o in rules.build_options(state, state.current_player) if tuple(o["site"]) == tuple(site)]

    def build_order(self, state: GameState, site: Hex, unit_type: str) -> dict[str, Any]:
        self.clear()
        return {"action": "build", "site": list(site), "unit": unit_type}

    # -- internals ----------------------------------------------------------

    def _commit(self, state: GameState, selected, pos: Hex) -> Intent | None:
        options = rules.unit_options(state, selected.id)
        mode = self.mode

        if mode in (MODE_IDLE, MODE_MENU, MODE_MOVE):
            for move in options["moves"]:
                if tuple(move["pos"]) == pos:
                    self.clear()
                    return Intent(
                        "order",
                        order={"action": "move", "unit": selected.id, "path": move["path"]},
                        note=f"{selected.id} moves",
                    )

        if mode in (MODE_IDLE, MODE_MENU, MODE_ATTACK):
            unit = state.unit_at(pos)
            if unit is not None and unit.owner != selected.owner:
                attacks = [a for a in options["attacks"] if a["target"] == unit.id]
                if attacks:
                    if len(attacks) > 1:
                        self.mode = MODE_WEAPON
                        self.weapon_target = unit.id
                        self.weapon_choices = sorted(
                            attacks, key=lambda a: (-a["preview_damage"], a["weapon"])
                        )
                        self.weapon_index = 0
                        return Intent("weapon", note=f"choose a weapon against {unit.id}")
                    best = attacks[0]
                    self.clear()
                    return Intent(
                        "order",
                        order={
                            "action": "attack",
                            "unit": selected.id,
                            "target": unit.id,
                            "weapon": best["weapon"],
                        },
                        note=f"{selected.id} attacks {unit.id} ({best['preview_damage']} dmg)",
                    )

        if mode in (MODE_IDLE, MODE_MENU) and selected.pos == pos and options["captures"]:
            self.clear()
            return Intent("order", order={"action": "capture", "unit": selected.id}, note=f"{selected.id} captures")

        if mode in (MODE_IDLE, MODE_MENU, MODE_UNLOAD):
            for unload in options["unloads"]:
                if tuple(unload["at"]) == pos:
                    self.clear()
                    return Intent(
                        "order",
                        order={
                            "action": "unload",
                            "unit": selected.id,
                            "cargo": unload["cargo"],
                            "target": list(unload["at"]),
                        },
                        note=f"{selected.id} unloads {unload['cargo']}",
                    )

        return None

    def _is_empty_build_site(self, state: GameState, pos: Hex, player: str) -> bool:
        tile = state.map.tiles.get(tuple(pos))
        if tile is None or tile.owner != player:
            return False
        if not state.terrain_of(tile).build:
            return False
        return state.unit_at(pos) is None
