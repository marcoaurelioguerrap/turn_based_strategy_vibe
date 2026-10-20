"""UI colours.  Pure data: no pygame import, so this module is testable anywhere.

Colours are a presentation concern and deliberately live in the UI rather than
in ``data/``: the engine and its JSON stay free of rendering details.  Every
rule-facing value a cell shows (terrain code, owner, unit glyph) still comes
from the engine, so recolouring the window can never change the game.
"""

from __future__ import annotations

Rgb = tuple[int, int, int]

#: Fallback fill for a terrain code the palette has not been taught.
DEFAULT_TERRAIN: Rgb = (140, 140, 140)

#: Terrain code -> base fill colour.  Codes match ``data/terrain.json``.
TERRAIN_COLORS: dict[str, Rgb] = {
    ".": (120, 168, 100),  # Plains
    "r": (172, 170, 158),  # Road
    "F": (60, 118, 70),    # Forest
    "#": (150, 120, 90),   # Rough
    "M": (118, 118, 130),  # Mountain
    "C": (198, 192, 172),  # City
    "f": (150, 145, 135),  # Factory
    "p": (92, 140, 172),   # Port
    "A": (200, 185, 150),  # Airport
    "H": (208, 178, 108),  # Headquarters
    "~": (70, 110, 172),   # Sea
}

#: Player colours, handed out in the scenario's own player order.
PLAYER_COLORS: tuple[Rgb, ...] = (
    (74, 118, 208),   # blue
    (206, 76, 72),    # red
    (86, 170, 96),    # green
    (208, 164, 62),   # amber
    (154, 96, 184),   # violet
    (78, 178, 178),   # teal
)

BACKGROUND: Rgb = (22, 24, 28)
PANEL: Rgb = (36, 38, 44)
PANEL_EDGE: Rgb = (70, 74, 84)
TEXT: Rgb = (232, 232, 232)
TEXT_DIM: Rgb = (164, 166, 172)
GRID: Rgb = (12, 12, 14)

MOVE_HIGHLIGHT: Rgb = (86, 148, 230)
ATTACK_HIGHLIGHT: Rgb = (232, 78, 78)
CAPTURE_HIGHLIGHT: Rgb = (232, 200, 78)
LOAD_HIGHLIGHT: Rgb = (86, 198, 124)
SELECT_HIGHLIGHT: Rgb = (255, 255, 255)

HP_FULL: Rgb = (92, 196, 96)
HP_HURT: Rgb = (224, 196, 72)
HP_LOW: Rgb = (216, 78, 72)


def terrain_color(code: str) -> Rgb:
    return TERRAIN_COLORS.get(code, DEFAULT_TERRAIN)


def player_color(player_id: str, player_ids: list[str] | tuple[str, ...]) -> Rgb:
    """A stable colour for a player, by position in the scenario's player list."""
    order = list(player_ids)
    index = order.index(player_id) if player_id in order else len(order)
    return PLAYER_COLORS[index % len(PLAYER_COLORS)]


def shade(color: Rgb, factor: float) -> Rgb:
    """Darken (factor < 1) or lighten (factor > 1) a colour, clamped to bytes."""
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)  # type: ignore[return-value]


def mix(a: Rgb, b: Rgb, t: float) -> Rgb:
    """Linear blend of two colours; ``t=0`` is ``a`` and ``t=1`` is ``b``."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def hp_color(hp: int, max_hp: int = 100) -> Rgb:
    ratio = hp / max_hp if max_hp else 0.0
    if ratio > 0.6:
        return HP_FULL
    if ratio > 0.3:
        return HP_HURT
    return HP_LOW
