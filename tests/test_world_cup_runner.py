"""The World Cup runner's model-free setup path."""

from __future__ import annotations

from pathlib import Path

from scripts.world_cup import build_teams, create_state, parse_args, resolve_paths


def test_default_world_cup_state_has_ten_teams_and_twenty_group_games(tmp_path: Path):
    args = parse_args(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--match-dir",
            str(tmp_path / "matches"),
            "--dashboard-dir",
            str(tmp_path / "dashboard"),
            "--batch-name",
            "world_cup_test",
        ]
    )
    teams = build_teams(args)
    assert len(teams) == 10
    assert sum(team.kind == "ncm" for team in teams) == 5
    assert sum(team.kind == "heuristic" for team in teams) == 4
    assert sum(team.kind == "greedy" for team in teams) == 1

    paths = resolve_paths(args)
    state = create_state(
        teams=teams,
        args=args,
        paths=paths,
        batch_name="world_cup_test",
    )
    assert state["batch"]["id"] == "personas/world_cup_test"
    assert state["format"]["total_matches"] == 24
    assert len(state["group_fixtures"]) == 20
    assert len(state["maps"]["group"]) == 5
    assert len(set(state["maps"]["group"])) == 5
    assert len(state["maps"]["knockout"]) == 4
    assert all(map_name.endswith("_m") for map_name in state["maps"]["group"])


def test_asymmetric_opt_in_puts_last_stand_in_the_knockout_rotation(tmp_path: Path):
    args = parse_args(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--allow-asymmetric-knockout",
        ]
    )
    teams = build_teams(args)
    paths = resolve_paths(args)
    state = create_state(
        teams=teams,
        args=args,
        paths=paths,
        batch_name="world_cup_asymmetric",
    )
    assert "last_stand_m" in state["maps"]["knockout"]
