"""Hex-grid maths.

Positions are stored as *axial* coordinates ``(q, r)``; the implied cube
coordinate is ``s = -q - r``.  Maps are authored as text rows in ``odd-r``
offset coordinates (one string per row, alternating rows shifted by half a hex)
and converted to axial coordinates when they are loaded, so map authors never
have to think in axial terms.

The neighbour ordering in :data:`AXIAL_DIRECTIONS` is part of the game
contract: pathfinding tie-breaks and generated action orders depend on it, so
do not reorder it casually.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable

Hex = tuple[int, int]

#: The six axial neighbour offsets, in deterministic search order.
AXIAL_DIRECTIONS: tuple[Hex, ...] = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def hex_add(a: Hex, b: Hex) -> Hex:
    """``a`` shifted by the axial offset ``b``."""
    return (a[0] + b[0], a[1] + b[1])


def hex_neighbors(a: Hex) -> list[Hex]:
    """The six hexes touching ``a``, in :data:`AXIAL_DIRECTIONS` order."""
    return [hex_add(a, d) for d in AXIAL_DIRECTIONS]


def cube(a: Hex) -> tuple[int, int, int]:
    """Axial to cube coordinates ``(q, r, s)`` with ``q + r + s == 0``."""
    q, r = a
    return (q, r, -q - r)


def hex_distance(a: Hex, b: Hex) -> int:
    """Number of hex steps from ``a`` to ``b``."""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def hex_range(center: Hex, dist: int) -> list[Hex]:
    """Every hex within ``dist`` of ``center`` (including the center), in a
    stable order so generated target lists are reproducible."""
    if dist < 0:
        return []
    out: list[Hex] = []
    for dq in range(-dist, dist + 1):
        low = max(-dist, -dq - dist)
        high = min(dist, -dq + dist)
        for ds in range(low, high + 1):
            out.append((center[0] + dq, center[1] - dq - ds))
    return out


def hex_line(a: Hex, b: Hex) -> list[Hex]:
    """Hexes on the straight line between ``a`` and ``b``, endpoints excluded.

    Used by line-of-sight rules (fog of war, M9).
    """
    steps = hex_distance(a, b)
    if steps <= 1:
        return []
    a_cube = cube(a)
    b_cube = cube(b)
    out: list[Hex] = []
    for i in range(1, steps):
        t = i / steps
        lerped = tuple(a_cube[k] + (b_cube[k] - a_cube[k]) * t for k in range(3))
        out.append(_cube_to_axial(_cube_round(lerped)))
    return out


def offset_to_axial(col: int, row: int) -> Hex:
    """``odd-r`` offset coordinates (text authoring) to axial.

    Rows are numbered from the top; odd rows are shifted right by half a hex.
    Note that the axial ``r`` equals the authored row, so a map row label in a
    text view is also an axial coordinate.
    """
    return (col - (row - (row & 1)) // 2, row)


def axial_to_offset(a: Hex) -> tuple[int, int]:
    """Axial to ``odd-r`` offset coordinates ``(col, row)``, used for text and
    graphical rendering."""
    q, r = a
    return (q + (r - (r & 1)) // 2, r)


def hex_cells(rows: Iterable[str]) -> dict[Hex, str]:
    """Turn a text map (one string per row, one character per hex) into
    ``{axial_position: terrain_code}``."""
    cells: dict[Hex, str] = {}
    for row, text in enumerate(rows):
        for col, code in enumerate(text):
            cells[offset_to_axial(col, row)] = code
    return cells


def axial_to_pixel(a: Hex, size: float) -> tuple[float, float]:
    """Axial to pixel position for a pointy-top layout (used by the pygame UI)."""
    q, r = a
    x = size * math.sqrt(3.0) * (q + r / 2.0)
    y = size * 1.5 * r
    return (x, y)


def pixel_to_axial(x: float, y: float, size: float) -> Hex:
    """Pixel position to the nearest axial hex (pointy-top layout)."""
    q = (math.sqrt(3.0) / 3.0 * x - y / 3.0) / size
    r = (2.0 / 3.0) * y / size
    return _cube_to_axial(_cube_round((q, r, -q - r)))


def _cube_to_axial(c: tuple[float, float, float]) -> Hex:
    return (int(round(c[0])), int(round(c[1])))


def _cube_round(c: tuple[float, float, float]) -> tuple[int, int, int]:
    q, r, s = (int(round(c[0])), int(round(c[1])), int(round(c[2])))
    dq, dr, ds = abs(q - c[0]), abs(r - c[1]), abs(s - c[2])
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    else:
        s = -q - r
    return (q, r, s)


#: Returns the cost of *entering* a hex, or ``None`` when it is impassable.
MoveCost = Callable[[Hex], "int | None"]


def reachable_hexes(
    start: Hex,
    budget: int,
    cost: MoveCost,
    blocked: frozenset[Hex] | set[Hex] = frozenset(),
) -> dict[Hex, tuple[int, list[Hex]]]:
    """Dijkstra over hexes, limited by ``budget`` movement points.

    Hexes in ``blocked`` may not be entered.  Returns ``{hex: (cost, path)}``
    where ``path`` starts at ``start``.  Equal-cost ties always resolve the same
    way for the same input, which keeps generated move lists deterministic.
    """
    best: dict[Hex, tuple[int, list[Hex]]] = {start: (0, [start])}
    queue: list[tuple[int, Hex]] = [(0, start)]
    while queue:
        cost_so_far, here = heapq.heappop(queue)
        if cost_so_far != best[here][0]:
            continue
        for nxt in hex_neighbors(here):
            if nxt in blocked:
                continue
            step = cost(nxt)
            if step is None:
                continue
            total = cost_so_far + step
            if total > budget:
                continue
            current = best.get(nxt)
            if current is not None and current[0] <= total:
                continue
            best[nxt] = (total, best[here][1] + [nxt])
            heapq.heappush(queue, (total, nxt))
    return best


def find_path(
    start: Hex,
    goal: Hex,
    budget: int,
    cost: MoveCost,
    blocked: frozenset[Hex] | set[Hex] = frozenset(),
) -> list[Hex] | None:
    """Cheapest legal path from ``start`` to ``goal``, or ``None``."""
    found = reachable_hexes(start, budget, cost, blocked).get(goal)
    return found[1] if found else None
