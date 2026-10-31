"""The pure World Cup league model: draw, schedule, tables and bracket."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import pytest

from tbs.agents.league import (
    Fixture,
    MatchResult,
    Team,
    aggregate_matchups,
    aggregate_team_stats,
    build_final_fixtures,
    build_group_fixtures,
    build_semifinal_fixtures,
    dashboard_meta,
    draw_groups,
    group_table,
    report_payload,
    render_markdown,
    resolve_knockout,
)

MAPS = ["m1", "m2", "m3", "m4", "m5"]
KNOCKOUT_MAPS = ["k1", "k2", "k3", "k4"]


def make_teams() -> list[Team]:
    teams = [
        Team(id=f"ncm-{name}", name=f"NCM {name.title()}", kind="ncm", seed=index)
        for index, name in enumerate(("blitz", "anvil", "line", "vanguard", "siege"))
    ]
    teams.extend(
        Team(
            id=f"heuristic-{seed}",
            name=f"Conventional {seed}",
            kind="heuristic",
            seed=5 + index,
            meta={"heuristic_seed": seed},
        )
        for index, seed in enumerate((900000, 900001, 900002, 900003))
    )
    teams.append(Team(id="greedy", name="Conventional Greedy", kind="greedy", seed=8))
    return teams


def make_result(
    fixture: Fixture,
    winner: str | None,
    *,
    home_score: float = 100.0,
    away_score: float = 90.0,
    stats: dict | None = None,
) -> MatchResult:
    stats = stats or {}
    return MatchResult(
        fixture_id=fixture.id,
        stage=fixture.stage,
        group=fixture.group,
        map=fixture.map,
        home=fixture.home,
        away=fixture.away,
        home_side="A",
        away_side="B",
        winner=winner,
        winner_side=("A" if winner == fixture.home else "B" if winner == fixture.away else None),
        decided_by="victory" if winner else "draw",
        score={fixture.home: home_score, fixture.away: away_score},
        status="victory" if winner else "draw",
        turns=20,
        seconds=1.5,
        orders=30,
        replay_matches=True,
        stats={
            fixture.home: stats.get(fixture.home, {}),
            fixture.away: stats.get(fixture.away, {}),
        },
        save="save.json",
        orders_path="save.orders.json",
        meta_path="save.meta.json",
        seed=fixture.seed,
    )


def group_results(
    fixtures: list[Fixture],
    winner_by_pair: dict[frozenset[str], str | None],
    *,
    scores: tuple[float, float] = (100.0, 100.0),
) -> dict[str, MatchResult]:
    results: dict[str, MatchResult] = {}
    for fixture in fixtures:
        if fixture.stage != "group":
            continue
        winner = winner_by_pair.get(frozenset({fixture.home, fixture.away}), "unset")
        if winner == "unset":
            raise AssertionError(f"no winner for {fixture.home} vs {fixture.away}")
        results[fixture.id] = make_result(
            fixture,
            winner,
            home_score=scores[0],
            away_score=scores[1],
        )
    return results


def test_draw_splits_all_ten_teams_and_keeps_ncm_in_both_groups():
    teams = make_teams()
    groups = draw_groups(teams, 20261025)
    assert set(groups) == {"A", "B"}
    assert all(len(group) == 5 for group in groups.values())
    assert {team for group in groups.values() for team in group} == {team.id for team in teams}
    ncm_ids = {team.id for team in teams if team.kind == "ncm"}
    assert 2 <= len(set(groups["A"]) & ncm_ids) <= 3
    assert 2 <= len(set(groups["B"]) & ncm_ids) <= 3


def test_schedule_has_five_rounds_one_bye_and_four_distinct_opponents_per_team():
    groups = draw_groups(make_teams(), 20261025)
    fixtures = build_group_fixtures(groups, MAPS, 20261025)
    assert len(fixtures) == 20
    assert len({fixture.id for fixture in fixtures}) == 20
    assert len({fixture.seed for fixture in fixtures}) == 20

    for group_name, team_ids in groups.items():
        group_fixtures = [fixture for fixture in fixtures if fixture.group == group_name]
        assert len(group_fixtures) == 10
        for round_number in range(1, 6):
            round_fixtures = [fixture for fixture in group_fixtures if fixture.round == round_number]
            assert len(round_fixtures) == 2
            bye = set(team_ids) - {
                team for fixture in round_fixtures for team in (fixture.home, fixture.away)
            }
            assert len(bye) == 1
        for team_id in team_ids:
            matches = [
                fixture
                for fixture in group_fixtures
                if team_id in (fixture.home, fixture.away)
            ]
            assert len(matches) == 4
            opponents = {
                fixture.away if fixture.home == team_id else fixture.home
                for fixture in matches
            }
            assert len(opponents) == 4
            assert sum(fixture.home == team_id for fixture in matches) == 2
            assert sum(fixture.away == team_id for fixture in matches) == 2
            maps = {fixture.map for fixture in matches}
            assert len(maps) == 4
        assert {
            fixture.map for fixture in group_fixtures
        } == set(MAPS)


def test_standings_use_points_and_then_head_to_head():
    teams = make_teams()
    teams_by_id = {team.id: team for team in teams}
    group = [team.id for team in teams[:5]]
    groups = {"A": group, "B": []}
    # A synthetic complete round-robin for the five team ids, in the same
    # shape the schedule builder emits.
    fixtures: list[Fixture] = []
    for index, home in enumerate(group):
        for away in group[index + 1 :]:
            fixtures.append(
                Fixture(
                    id=f"gA-{home}-{away}",
                    stage="group",
                    round=1,
                    group="A",
                    map="m1",
                    home=home,
                    away=away,
                    seed=index,
                )
            )
    # a and b both finish 3-1; b beat a, so b must win the head-to-head tie.
    winners = {
        frozenset({group[0], group[1]}): group[1],
        frozenset({group[0], group[2]}): group[0],
        frozenset({group[0], group[3]}): group[0],
        frozenset({group[0], group[4]}): group[0],
        frozenset({group[1], group[2]}): group[1],
        frozenset({group[1], group[3]}): group[1],
        frozenset({group[1], group[4]}): group[4],
        frozenset({group[2], group[3]}): group[3],
        frozenset({group[2], group[4]}): group[2],
        frozenset({group[3], group[4]}): group[3],
    }
    results = {}
    for fixture in fixtures:
        results[fixture.id] = make_result(
            fixture,
            winners[frozenset({fixture.home, fixture.away})],
            home_score=100.0,
            away_score=100.0,
        )
    table = group_table("A", teams_by_id, fixtures, results)
    assert [row["team"] for row in table] == [group[1], group[0], group[3], group[2], group[4]]
    assert table[0]["points"] == table[1]["points"] == 9
    assert table[0]["score_diff"] == table[1]["score_diff"] == 0


def test_semifinals_cross_the_groups_and_finals_use_winners_and_losers():
    tables = {
        "A": [{"team": "a1", "pos": 1}, {"team": "a2", "pos": 2}],
        "B": [{"team": "b1", "pos": 1}, {"team": "b2", "pos": 2}],
    }
    semis = build_semifinal_fixtures(tables, KNOCKOUT_MAPS, 7)
    assert [(fixture.id, fixture.home, fixture.away, fixture.map) for fixture in semis] == [
        ("sf1", "a1", "b2", "k1"),
        ("sf2", "b1", "a2", "k2"),
    ]
    first = make_result(semis[0], "a1")
    second = make_result(semis[1], "a2")
    finals = build_final_fixtures(
        {"sf1": first, "sf2": second},
        KNOCKOUT_MAPS,
        7,
        team_seeds={"a1": 1, "b2": 4, "b1": 2, "a2": 3},
    )
    assert [(fixture.id, fixture.home, fixture.away) for fixture in finals] == [
        ("third", "b1", "b2"),
        ("final", "a1", "a2"),
    ]
    assert [fixture.map for fixture in finals] == ["k3", "k4"]


def test_knockout_falls_back_to_board_score_then_seed():
    fixture = Fixture(
        id="sf1",
        stage="semifinal",
        round=1,
        group=None,
        map="k1",
        home="home",
        away="away",
        seed=1,
    )
    assert resolve_knockout(
        fixture, "A", "B", None, {"home": 110.0, "away": 100.0}
    ) == ("home", "A", "board")
    assert resolve_knockout(
        fixture, "A", "B", None, {"home": 100.0, "away": 100.0}, {"home": 2, "away": 1}
    ) == ("away", "B", "seed")
    assert resolve_knockout(fixture, "A", "B", "away", {"home": 1.0, "away": 0.0}) == (
        "away",
        "B",
        "victory",
    )


def test_dashboard_meta_names_each_side_and_marks_the_winner():
    teams = make_teams()
    by_id = {team.id: team for team in teams}
    fixture = Fixture(
        id="gA-r1-ncm-blitz-vs-heuristic-900000",
        stage="group",
        round=1,
        group="A",
        map="m1",
        home="ncm-blitz",
        away="heuristic-900000",
        seed=1,
    )
    result = make_result(fixture, "ncm-blitz")
    meta = dashboard_meta(result, teams, "personas/world_cup_test")
    assert meta["ncm_side"] == "A"
    assert meta["opponent"] == "heuristic"
    assert meta["team_labels"] == {"A": "NCM Blitz", "B": "Conventional 900000"}
    assert meta["world_cup"]["winner_name"] == "NCM Blitz"
    assert meta["batch"] == "personas/world_cup_test"

    ncm_fixture = Fixture(
        id="gA-r1-ncm-blitz-vs-ncm-anvil",
        stage="group",
        round=1,
        group="A",
        map="m1",
        home="ncm-blitz",
        away="ncm-anvil",
        seed=2,
    )
    assert dashboard_meta(make_result(ncm_fixture, None), teams, "x")["ncm_side"] == "both"

    conventional_fixture = Fixture(
        id="gA-r1-heuristic-900000-vs-greedy",
        stage="group",
        round=1,
        group="A",
        map="m1",
        home="heuristic-900000",
        away="greedy",
        seed=3,
    )
    conventional = dashboard_meta(make_result(conventional_fixture, "greedy"), teams, "x")
    assert conventional["ncm_side"] is None
    assert conventional["opponent"] == "conventional"
    assert conventional["world_cup"]["winner_name"] == "Conventional Greedy"


def test_aggregates_and_report_payload_feed_the_dashboard():
    teams = make_teams()
    groups = draw_groups(teams, 20261025)
    fixtures = build_group_fixtures(groups, MAPS, 20261025)
    # Every home team wins; this is only a report-shape test, not a strength claim.
    results = {fixture.id: make_result(fixture, fixture.home) for fixture in fixtures}
    state = {
        "schema": 1,
        "tournament": "TBS World Cup 2026",
        "seed": 20261025,
        "teams": [asdict(team) for team in teams],
        "groups": groups,
        "group_fixtures": [fixture.to_dict() for fixture in fixtures],
        "knockout_fixtures": [],
        "results": {key: result.to_dict() for key, result in results.items()},
        "maps": {"group": MAPS, "knockout": KNOCKOUT_MAPS},
        "format": {"total_matches": 24},
        "settings": {"max_turns": 30, "max_steps": 10},
        "batch": {"id": "personas/world_cup_test", "dir": ""},
        "match_dir": "matches",
        "state_path": "state.json",
        "out_dir": "docs/world_cup",
    }
    payload = report_payload(state)
    assert payload["matches"] == 20
    assert len(payload["games"]) == 20
    assert payload["games"][0]["pair"]
    assert payload["games"][0]["seed"]
    assert payload["ncm_personas"]
    assert payload["heuristics"]
    assert payload["tables"]["A"][0]["points"] == 6

    stats = aggregate_team_stats(teams, results.values())
    assert sum(row["games"] for row in stats.values()) == 40
    matchups = aggregate_matchups(teams, results.values())
    assert (
        matchups["ncm_vs_ncm"]["games"]
        + matchups["ncm_vs_conventional"]["games"]
        + matchups["conventional_vs_conventional"]["games"]
        == 20
    )
    text = render_markdown(state)
    assert "Group A" in text
    assert "Group B" in text
    assert "NCM vs conventional AI" in text
    assert "TBS World Cup 2026" in text


def test_match_result_round_trips_for_resume(tmp_path):
    fixture = Fixture(
        id="final",
        stage="final",
        round=2,
        group=None,
        map="k4",
        home="home",
        away="away",
        seed=1,
    )
    result = make_result(
        fixture,
        "home",
        stats={
            "home": {
                "steps": 10,
                "orders": 10,
                "actions": {"move": 7, "attack": 3},
                "builds": {},
            },
            "away": {},
        },
    )
    restored = MatchResult.from_dict(result.to_dict())
    assert restored == result
