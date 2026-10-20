"""The pure menu model: navigation skips disabled entries and wraps around."""

from __future__ import annotations

from tbs.ui.menu import Menu, MenuItem


def make_menu() -> Menu:
    return Menu(
        "test",
        [
            MenuItem("a", "A", enabled=False),
            MenuItem("b", "B"),
            MenuItem("c", "C", enabled=False),
            MenuItem("d", "D"),
        ],
    )


def test_starting_index_snaps_to_the_first_enabled_item():
    menu = make_menu()
    assert menu.current().key == "b"


def test_move_skips_disabled_items():
    menu = make_menu()
    menu.move(1)
    assert menu.current().key == "d", "C is disabled and must be skipped"
    menu.move(1)
    assert menu.current().key == "b", "navigation wraps back to the first enabled item"


def test_move_backwards_skips_disabled_items():
    menu = make_menu()
    menu.move(-1)
    assert menu.current().key == "d"


def test_choose_never_returns_a_disabled_item():
    menu = make_menu()
    menu.index = 0  # A is disabled
    assert menu.choose() is None
    menu.index = 1
    assert menu.choose().key == "b"


def test_set_index_clamps_and_snaps():
    menu = make_menu()
    menu.set_index(99)
    assert menu.current().key == "d"
    menu.set_index(0)
    assert menu.current().key == "b"
