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


def make_long_menu() -> Menu:
    return Menu(
        "long",
        [MenuItem(f"item{i}", f"Item {i}") for i in range(20)],
        visible_rows=5,
    )


def test_a_long_menu_scrolls_to_keep_the_highlight_visible():
    menu = make_long_menu()
    assert menu.window() == (0, 5)

    for _ in range(5):
        menu.move(1)
    assert menu.index == 5
    assert menu.window() == (1, 6), "one row off the top, one new at the bottom"

    menu.set_index(19)
    assert menu.window() == (15, 20)
    menu.set_index(0)
    assert menu.window() == (0, 5)


def test_page_moves_by_a_whole_window_and_wraps():
    menu = make_long_menu()
    menu.page(1)
    assert menu.index == 5
    menu.page(1)
    assert menu.index == 10
    menu.page(-2)
    assert menu.index == 0


def test_visible_items_carry_their_real_indexes():
    menu = make_long_menu()
    menu.set_index(7)
    visible = menu.visible_items()
    assert [index for index, _ in visible] == [3, 4, 5, 6, 7]
    assert [item.key for _, item in visible][-1] == "item7"
