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
    def __init__(self, title: str, items: list[MenuItem]):
        self.title = title
        self.items = list(items)
        self.index = 0
        self._snap_to_enabled()

    def _snap_to_enabled(self, step: int = 1) -> None:
        if not self.items:
            self.index = 0
            return
        for offset in range(len(self.items)):
            index = (self.index + step * offset) % len(self.items)
            if self.items[index].enabled:
                self.index = index
                return
        self.index = 0

    def move(self, delta: int) -> None:
        """Highlight the next enabled item, wrapping around."""
        if not self.items:
            return
        self.index = (self.index + delta) % len(self.items)
        self._snap_to_enabled(step=delta or 1)

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
