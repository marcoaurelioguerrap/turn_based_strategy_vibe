"""The all-AI league fixtures and score table, without loading any model."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from scripts.all_ai_league import (
    _board_winner,
    _head_to_head,
    _resolve_maps,
    _standings,
    _turn_cap,
    build_fixtures,
)

ROSTER = ["llm", "ncm-vanguard", "ncm-siege", "conventional"]


def test_the_league_plays_every_pair_once_per_map_and_balances_sides():
    maps = _resolve_maps("all")
    fixtures = build_fixtures(ROSTER, maps, seed_base=880000)

    assert len(maps) == 10
    assert len(fixtures) == 60
    assert len({fixture.id for fixture in fixtures}) == 60
    assert len({fixture.seed for fixture in fixtures}) == 60

    per_map = Counter(fixture.map_name for fixture in fixtures)
    assert set(per_map.values()) == {6}, "six pairings per map"

    pairs = Counter(
        (fixture.map_name, tuple(sorted((fixture.agent_a, fixture.agent_b))))
        for fixture in fixtures
    )
    assert set(pairs.values()) == {1}, "each pair appears exactly once on a map"

    sides = Counter()
    for fixture in fixtures:
        sides[(fixture.agent_a, "A")] += 1
        sides[(fixture.agent_b, "B")] += 1
    assert set(sides.values()) == {15}, "every representative has fifteen A and fifteen B games"


def test_board_decision_and_authored_turn_caps():
    board = {"A": {"score": 12.0, "units": 4}, "B": {"score": 10.0, "units": 3}}
    assert _board_winner(board) == "A"
    assert _board_winner(board, margin=2.0) is None
    assert _board_winner({"A": {"score": 10.0}, "B": {"score": 10.0}}) is None

    assert _turn_cap(SimpleNamespace(max_turns=0), {"max_turns": 120}) == 120
    assert _turn_cap(SimpleNamespace(max_turns=45), {"max_turns": 120}) == 45


def test_unfinished_games_score_half_a_point_and_head_to_head_is_symmetric():
    results = [
        {
            "agents": {"A": "llm", "B": "ncm-vanguard"},
            "board": {"A": {"score": 11.0, "units": 5}, "B": {"score": 9.0, "units": 4}},
            "winner": None,
            "unfinished": True,
            "leader": "llm",
            "error": None,
        },
        {
            "agents": {"A": "conventional", "B": "llm"},
            "board": {"A": {"score": 4.0, "units": 2}, "B": {"score": 12.0, "units": 6}},
            "winner": "llm",
            "unfinished": False,
            "leader": None,
            "error": None,
        },
    ]
    rows = _standings(results, ROSTER)
    assert rows["llm"]["points"] == 1.5
    assert rows["ncm-vanguard"]["points"] == 0.5
    assert rows["conventional"]["points"] == 0.0

    table = _head_to_head(results, ROSTER)
    assert table["llm"]["ncm-vanguard"]["draws"] == 1
    assert table["ncm-vanguard"]["llm"]["draws"] == 1
    assert table["llm"]["conventional"]["wins"] == 1
    assert table["conventional"]["llm"]["losses"] == 1
