"""Icon-toolbar layout and hit-testing.  Pure: no pygame import.

The window draws one button per entry along the left border, below the HUD.
The same layout powers click handling, so a drawn button and a clickable
button can never drift apart.  Icons themselves are vector-drawn in
:mod:`tbs.ui.render`; this module only owns geometry and identities.
"""

from __future__ import annotations

from dataclasses import dataclass

TOOLBAR_X = 6
TOOLBAR_TOP_GAP = 6
BUTTON_SIZE = 40
BUTTON_GAP = 5
TOOLBAR_WIDTH = TOOLBAR_X * 2 + BUTTON_SIZE

#: Toolbar buttons, in top-to-bottom order: ``(id, human label)``.
TOOLBAR_ITEMS: tuple[tuple[str, str], ...] = (
    ("menu", "Menu"),
    ("end_turn", "End turn"),
    ("units", "Units"),
    ("build", "Build"),
    ("save", "Save"),
    ("load", "Load"),
    ("log", "Log"),
    ("coords", "Coordinates"),
    ("zoom_out", "Zoom out"),
    ("zoom_in", "Zoom in"),
)


@dataclass(frozen=True)
class Button:
    id: str
    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


def toolbar_layout(top: int, height: int) -> list[Button]:
    """Lay the toolbar out below ``top``, dropping buttons that would overflow."""
    buttons: list[Button] = []
    y = top + TOOLBAR_TOP_GAP
    for button_id, label in TOOLBAR_ITEMS:
        if y + BUTTON_SIZE > height - TOOLBAR_TOP_GAP:
            break
        buttons.append(Button(button_id, label, TOOLBAR_X, y, BUTTON_SIZE, BUTTON_SIZE))
        y += BUTTON_SIZE + BUTTON_GAP
    return buttons


def button_at(buttons: list[Button], x: float, y: float) -> Button | None:
    for button in buttons:
        if button.contains(x, y):
            return button
    return None
