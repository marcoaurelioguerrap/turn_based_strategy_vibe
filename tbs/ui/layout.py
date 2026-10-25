"""Hex/pixel geometry and the viewport camera.  Pure: no pygame import.

The engine already owns the authoritative coordinate maths
(:func:`tbs.engine.hexgrid.axial_to_pixel` / ``pixel_to_axial``); this module
only adds what a window needs on top: a hex polygon, the bounding box of the
whole board, a camera that maps board pixels to screen pixels, and hit-testing.
Keeping it pygame-free means every geometry claim is pinned by a fast test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tbs.engine.hexgrid import AXIAL_DIRECTIONS, Hex, axial_to_pixel, pixel_to_axial
from tbs.engine.state import GameState

#: Default circumradius of a hex, in pixels.
DEFAULT_HEX_SIZE = 34.0
MIN_HEX_SIZE = 14.0
MAX_HEX_SIZE = 90.0


def hex_corners(center: tuple[float, float], size: float) -> list[tuple[float, float]]:
    """The six corners of a pointy-top hex, clockwise from the upper-right.

    The same pointy-top layout the engine's ``axial_to_pixel`` assumes, so the
    polygon always lines up with the hit-testing maths.
    """
    cx, cy = center
    return [
        (
            cx + size * math.cos(math.radians(60 * i + 30)),
            cy + size * math.sin(math.radians(60 * i + 30)),
        )
        for i in range(6)
    ]


def board_bounds(state: GameState, size: float = DEFAULT_HEX_SIZE) -> tuple[float, float, float, float]:
    """Pixel bounding box ``(min_x, min_y, max_x, max_y)`` of every hex centre."""
    xs: list[float] = []
    ys: list[float] = []
    for pos in state.map.tiles:
        x, y = axial_to_pixel(pos, size)
        xs.append(x)
        ys.append(y)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Camera:
    """Maps board pixels to screen pixels: ``screen = board - (pan_x, pan_y)``."""

    size: float = DEFAULT_HEX_SIZE
    pan_x: float = 0.0
    pan_y: float = 0.0

    @classmethod
    def centered(
        cls,
        state: GameState,
        viewport_width: float,
        viewport_height: float,
        size: float = DEFAULT_HEX_SIZE,
    ) -> "Camera":
        """A camera with the whole board centred in the viewport."""
        min_x, min_y, max_x, max_y = board_bounds(state, size)
        camera = cls(size=size)
        camera.pan_x = (min_x + max_x) / 2.0 - viewport_width / 2.0
        camera.pan_y = (min_y + max_y) / 2.0 - viewport_height / 2.0
        return camera

    @classmethod
    def fit(
        cls,
        state: GameState,
        viewport_width: float,
        viewport_height: float,
        *,
        padding: float = 40.0,
    ) -> "Camera":
        """A camera whose hex size makes the whole board fit the viewport.

        ``board_bounds`` scales linearly with the hex size, so one measurement
        at size 1 gives the aspect ratio; the result is clamped to the zoom
        range so a tiny map is not blown up past ``MAX_HEX_SIZE`` and a huge
        one stays legible.
        """
        min_x, min_y, max_x, max_y = board_bounds(state, 1.0)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        available_x = max(1.0, viewport_width - 2.0 * padding)
        available_y = max(1.0, viewport_height - 2.0 * padding)
        size = min(available_x / span_x, available_y / span_y)
        size = max(MIN_HEX_SIZE, min(MAX_HEX_SIZE, size))
        return cls.centered(state, viewport_width, viewport_height, size=size)

    def to_screen(self, pos: Hex) -> tuple[float, float]:
        x, y = axial_to_pixel(tuple(pos), self.size)
        return (x - self.pan_x, y - self.pan_y)

    def to_hex(self, x: float, y: float) -> Hex:
        return pixel_to_axial(x + self.pan_x, y + self.pan_y, self.size)

    def pan(self, dx: float, dy: float) -> None:
        """Move the view by a screen-pixel delta (the board follows the drag)."""
        self.pan_x -= dx
        self.pan_y -= dy

    def focus_on(self, pos: Hex, viewport_width: float, viewport_height: float) -> None:
        """Centre the viewport on one hex."""
        x, y = axial_to_pixel(tuple(pos), self.size)
        self.pan_x = x - viewport_width / 2.0
        self.pan_y = y - viewport_height / 2.0

    def zoom_at(self, factor: float, x: float, y: float) -> None:
        """Zoom by ``factor`` keeping the hex under the screen point ``(x, y)``."""
        anchor = self.to_hex(x, y)
        new_size = max(MIN_HEX_SIZE, min(MAX_HEX_SIZE, self.size * factor))
        if new_size == self.size:
            return
        self.size = new_size
        px, py = axial_to_pixel(anchor, self.size)
        self.pan_x = px - x
        self.pan_y = py - y


def hex_at(state: GameState, x: float, y: float, camera: Camera) -> Hex | None:
    """The map hex under a screen point, or ``None`` where the map ends."""
    pos = camera.to_hex(x, y)
    return pos if state.map.in_bounds(pos) else None


def hex_edge_towards(
    center: tuple[float, float],
    neighbor_center: tuple[float, float],
    size: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two corners of a hex that face ``neighbor_center`` (its shared edge)."""
    corners = hex_corners((0.0, 0.0), size)
    vx = neighbor_center[0] - center[0]
    vy = neighbor_center[1] - center[1]
    length = math.hypot(vx, vy) or 1.0
    vx, vy = vx / length, vy / length
    facing = sorted(range(6), key=lambda i: -(corners[i][0] * vx + corners[i][1] * vy))
    first, second = corners[facing[0]], corners[facing[1]]
    return (
        (center[0] + first[0], center[1] + first[1]),
        (center[0] + second[0], center[1] + second[1]),
    )


def reachable_boundary(
    region: set[Hex] | frozenset[Hex],
    size: float = DEFAULT_HEX_SIZE,
    offset: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The outline of a connected region of hexes, as screen-space line segments.

    For every hex in ``region`` the edges that do *not* border another member
    are emitted, so the caller can stroke the whole reachable area's perimeter
    rather than every tile inside it.  ``offset`` maps board pixels to screen.
    """
    region = {tuple(pos) for pos in region}
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for pos in sorted(region):
        cx, cy = axial_to_pixel(pos, size)
        center = (cx - offset[0], cy - offset[1])
        for direction in AXIAL_DIRECTIONS:
            neighbor = (pos[0] + direction[0], pos[1] + direction[1])
            if neighbor in region:
                continue
            nx, ny = axial_to_pixel(neighbor, size)
            neighbor_center = (nx - offset[0], ny - offset[1])
            segments.append(hex_edge_towards(center, neighbor_center, size))
    return segments
