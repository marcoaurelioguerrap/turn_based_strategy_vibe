"""Geometry and guardrails for the pygame UI, all without importing pygame.

Every hex pixel claim is pinned here, so the window cannot quietly drift from
the engine's own coordinate maths, and a guard test keeps ``engine/`` free of
any UI or pygame dependency.
"""

from __future__ import annotations

from pathlib import Path

import tbs.engine
from tbs.engine.data import default_scenario, load_scenario, load_specs
from tbs.engine.state import new_game
from tbs.ui import layout


def fresh_state():
    return new_game(load_scenario(default_scenario()), load_specs())


# ---------------------------------------------------------------------------
# the engine stays pure
# ---------------------------------------------------------------------------


def test_engine_never_imports_pygame_the_ui_or_the_agent_layer():
    engine_dir = Path(tbs.engine.__file__).resolve().parent
    offenders = []
    for path in engine_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import pygame" in text or "tbs.ui" in text or "tbs.agents" in text:
            offenders.append(path.name)
    assert offenders == [], f"engine modules must stay client-free, but these import one: {offenders}"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_every_hex_round_trips_through_the_camera():
    state = fresh_state()
    camera = layout.Camera(size=layout.DEFAULT_HEX_SIZE)
    for pos in state.map.tiles:
        screen = camera.to_screen(pos)
        assert camera.to_hex(*screen) == pos, f"hex {pos} did not survive the screen round trip"


def test_hex_corners_are_a_pointy_top_six_gon():
    corners = layout.hex_corners((0.0, 0.0), 10.0)
    assert len(corners) == 6
    assert len(set(corners)) == 6
    # A pointy-top hex reaches its full size top and bottom, and sqrt(3)/2 of it sideways.
    ys = [y for _, y in corners]
    xs = [x for x, _ in corners]
    assert max(ys) - min(ys) == 20.0
    assert abs((max(xs) - min(xs)) - 10.0 * 3**0.5) < 1e-9


def test_the_board_is_centred_in_the_viewport():
    state = fresh_state()
    width, height = 1000, 700
    camera = layout.Camera.centered(state, width, height)
    min_x, min_y, max_x, max_y = layout.board_bounds(state, camera.size)
    board_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    screen_center = (board_center[0] - camera.pan_x, board_center[1] - camera.pan_y)
    assert abs(screen_center[0] - width / 2) < 1e-6
    assert abs(screen_center[1] - height / 2) < 1e-6


def test_hit_testing_finds_the_tile_under_a_point_and_rejects_off_board():
    state = fresh_state()
    camera = layout.Camera.centered(state, 1000, 700)
    for pos in list(state.map.tiles)[:10]:
        x, y = camera.to_screen(pos)
        assert layout.hex_at(state, x, y, camera) == pos
    assert layout.hex_at(state, -10_000, -10_000, camera) is None


def test_fit_keeps_every_bundled_map_on_screen():
    from tbs.engine.data import available_scenarios

    for path in available_scenarios():
        state = new_game(load_scenario(path), load_specs())
        camera = layout.Camera.fit(state, 800, 600)
        assert layout.MIN_HEX_SIZE <= camera.size <= layout.MAX_HEX_SIZE
        for pos in state.map.tiles:
            x, y = camera.to_screen(pos)
            assert -1 <= x <= 801, f"{path.stem}: hex {pos} is off-screen horizontally"
            assert -1 <= y <= 601, f"{path.stem}: hex {pos} is off-screen vertically"


def test_zoom_keeps_the_hex_under_the_cursor():
    state = fresh_state()
    camera = layout.Camera.centered(state, 1000, 700)
    anchor = (3, 2)
    x, y = camera.to_screen(anchor)
    camera.zoom_at(1.5, x, y)
    assert layout.hex_at(state, x, y, camera) == anchor
    camera.zoom_at(1 / 1.5, x, y)
    assert layout.hex_at(state, x, y, camera) == anchor


def test_zoom_is_clamped():
    state = fresh_state()
    camera = layout.Camera.centered(state, 1000, 700)
    for _ in range(50):
        camera.zoom_at(2.0, 500, 350)
    assert camera.size == layout.MAX_HEX_SIZE
    for _ in range(100):
        camera.zoom_at(0.5, 500, 350)
    assert camera.size == layout.MIN_HEX_SIZE


# ---------------------------------------------------------------------------
# the reachable-area boundary
# ---------------------------------------------------------------------------


def test_a_single_hex_has_a_six_edge_boundary():
    segments = layout.reachable_boundary({(0, 0)}, size=10.0)
    assert len(segments) == 6


def test_two_adjacent_hexes_share_an_edge_so_the_boundary_has_ten():
    from tbs.engine.hexgrid import AXIAL_DIRECTIONS

    neighbor = AXIAL_DIRECTIONS[0]
    segments = layout.reachable_boundary({(0, 0), neighbor}, size=10.0)
    assert len(segments) == 6 + 6 - 2, "the shared edge is interior and must not be stroked"


def test_a_solid_flower_of_seven_hexes_has_an_eighteen_edge_boundary():
    from tbs.engine.hexgrid import hex_neighbors

    region = {(0, 0), *hex_neighbors((0, 0))}
    # Each of the six outer hexes keeps three of its edges; the centre keeps none.
    assert len(layout.reachable_boundary(region, size=10.0)) == 18


def test_boundary_edges_meet_at_shared_corners():
    from tbs.engine.hexgrid import AXIAL_DIRECTIONS

    neighbor = AXIAL_DIRECTIONS[0]
    segments = layout.reachable_boundary({(0, 0), neighbor}, size=10.0)
    endpoints = {
        (round(point[0], 4), round(point[1], 4)) for start, end in segments for point in (start, end)
    }
    # twelve edges of a two-hex domino meet at ten distinct corners
    assert len(endpoints) == 10
