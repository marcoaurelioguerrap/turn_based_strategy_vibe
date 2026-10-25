"""Menu model.  Pure: no pygame import, so navigation is tested headlessly.

A menu is a title plus a list of :class:`MenuItem`s and a highlighted index.
Navigation skips disabled entries, which is how "Attack" can be shown but
greyed out when a unit has nothing in range.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    enabled: bool = True
    hint: str = ""


class Menu:
    def __init__(self, title: str, items: list[MenuItem], visible_rows: int | None = None):
        self.title = title
        self.items = list(items)
        self.index = 0
        #: When set and smaller than ``len(items)``, only this many rows are
        #: drawn at a time; the window scrolls to keep the highlight visible.
        self.visible_rows = max(1, visible_rows) if visible_rows else None
        self.first_visible = 0
        self._snap_to_enabled()

    def _snap_to_enabled(self, step: int = 1) -> None:
        if not self.items:
            self.index = 0
            self._sync_window()
            return
        for offset in range(len(self.items)):
            index = (self.index + step * offset) % len(self.items)
            if self.items[index].enabled:
                self.index = index
                break
        else:
            self.index = 0
        self._sync_window()

    def _sync_window(self) -> None:
        """Scroll the visible window just enough to contain ``index``."""
        total = len(self.items)
        if not self.visible_rows or self.visible_rows >= total:
            self.first_visible = 0
            return
        if self.index < self.first_visible:
            self.first_visible = self.index
        elif self.index >= self.first_visible + self.visible_rows:
            self.first_visible = self.index - self.visible_rows + 1
        self.first_visible = max(0, min(self.first_visible, total - self.visible_rows))

    def window(self) -> tuple[int, int]:
        """The half-open ``(first, stop)`` range of item indexes to draw."""
        self._sync_window()
        total = len(self.items)
        if not self.visible_rows or self.visible_rows >= total:
            return (0, total)
        return (self.first_visible, self.first_visible + self.visible_rows)

    def visible_items(self) -> list[tuple[int, "MenuItem"]]:
        first, stop = self.window()
        return list(enumerate(self.items[first:stop], start=first))

    def move(self, delta: int) -> None:
        """Highlight the next enabled item, wrapping around."""
        if not self.items:
            return
        self.index = (self.index + delta) % len(self.items)
        self._snap_to_enabled(step=delta or 1)

    def page(self, delta: int) -> None:
        """Jump by one visible page (PageUp / PageDown)."""
        self.move(delta * (self.visible_rows or max(1, len(self.items))))

    def current(self) -> MenuItem | None:
        if not self.items:
            return None
        return self.items[self.index]

    def choose(self) -> MenuItem | None:
        item = self.current()
        if item is None or not item.enabled:
            return None
        return item

    def set_index(self, index: int) -> None:
        if not self.items:
            return
        self.index = max(0, min(index, len(self.items) - 1))
        self._snap_to_enabled()
