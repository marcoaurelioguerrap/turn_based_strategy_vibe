"""Toolbar geometry: layout, overflow and hit-testing, all without pygame."""

from __future__ import annotations

from tbs.ui import buttons


def test_toolbar_lays_out_every_button_inside_the_window():
    layout = buttons.toolbar_layout(top=42, height=720)
    assert [button.id for button in layout] == [item[0] for item in buttons.TOOLBAR_ITEMS]
    assert all(button.y >= 42 and button.y + button.h <= 720 for button in layout)
    assert all(button.x == buttons.TOOLBAR_X for button in layout)


def test_button_at_finds_the_drawn_button():
    layout = buttons.toolbar_layout(top=42, height=720)
    for button in layout:
        assert buttons.button_at(layout, button.x + 1, button.y + 1) == button
    assert buttons.button_at(layout, 700, 700) is None


def test_a_short_window_drops_buttons_that_would_overflow():
    layout = buttons.toolbar_layout(top=42, height=150)
    assert 0 < len(layout) < len(buttons.TOOLBAR_ITEMS)
    assert all(button.y + button.h <= 150 - buttons.TOOLBAR_TOP_GAP for button in layout)
