"""World Cup league format: draw, round-robin, standings, bracket and reports.

The engine and the agent layer only know about one match at a time.  This module
owns the tournament shape around those matches and stays completely pure: no
torch, no NCM import and no file I/O.  ``scripts/world_cup.py`` is the thin
runner that feeds it saved matches from ``tbs_ncm.arena``.

The format implemented here is the one used by the TBS World Cup:

* 10 teams split into two groups of five by a seeded pot draw;
* every group plays a five-round round-robin (each team has one bye);
* the top two teams in each group meet cross-group in the semifinals;
* the semifinal losers play for third place and the winners play the final;
* group matches award 3/1/0 points, knockouts resolve a board-score draw by the
  board score and then by the group seed.

Group fixtures are stored as plain dataclasses.  Results are stored separately
so an interrupted run can persist and resume without replaying finished games.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

__all__ = [
    "GROUP_LABELS",
    "Fixture",
    "MatchResult",
    "Team",
    "aggregate_matchups",
    "aggregate_map_summary",
    "aggregate_team_stats",
    "build_final_fixtures",
    "build_group_fixtures",
    "build_semifinal_fixtures",
    "draw_groups",
    "dashboard_meta",
    "dumps_report",
    "group_table",
    "match_seed",
    "report_payload",
    "render_markdown",
    "resolve_knockout",
]

GROUP_LABELS = ("A", "B")
GROUP_SIZE = 5
POINTS_WIN = 3
POINTS_DRAW = 1


@dataclass(frozen=True)
class Team:
    """One tournament entrant.

    ``kind`` is ``ncm``, ``heuristic`` or ``greedy``.  ``seed`` is the draw
    position used for the final deterministic tiebreak.
    """

    id: str
    name: str
    kind: str
    seed: int = 0
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Fixture:
    """One scheduled match.  ``home``/``away`` are side-A/side-B slots."""

    id: str
    stage: str
    round: int
    group: str | None
    map: str
    home: str
    away: str
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """One completed match in a tournament-friendly shape."""

    fixture_id: str
    stage: str
    group: str | None
    map: str
    home: str
    away: str
    home_side: str
    away_side: str
    winner: str | None
    winner_side: str | None
    decided_by: str
    score: dict[str, float]
    status: str
    turns: int
    seconds: float
    orders: int
    replay_matches: bool | None
    stats: dict[str, dict[str, Any]]
    save: str
    orders_path: str
    meta_path: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchResult":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in known})


# ---------------------------------------------------------------------------
# draw and schedule


def draw_groups(teams: Iterable[Team], seed: int) -> dict[str, list[str]]:
    """Split ten teams into two seeded groups of five.

    The five NCM teams are the first pot.  One group receives three of them and
    the other two, so both groups contain NCM and conventional opposition.  The
    conventional pot is shuffled independently and fills the remaining slots.
    """
    teams = list(teams)
    if len(teams) != GROUP_SIZE * len(GROUP_LABELS):
        raise ValueError(f"a World Cup needs {GROUP_SIZE * len(GROUP_LABELS)} teams, got {len(teams)}")
    ncm = [team for team in teams if team.kind == "ncm"]
    conventional = [team for team in teams if team.kind != "ncm"]
    if len(ncm) != GROUP_SIZE or len(conventional) != GROUP_SIZE:
        raise ValueError(
            "the seeded World Cup needs exactly five NCM and five conventional teams, "
            f"got {len(ncm)} NCM and {len(conventional)} conventional"
        )

    rng = random.Random(seed)
    rng.shuffle(ncm)
    rng.shuffle(conventional)
    ncm_in_a = 3 if seed % 2 == 0 else 2
    conventional_in_a = GROUP_SIZE - ncm_in_a
    groups: dict[str, list[str]] = {
        "A": [team.id for team in ncm[:ncm_in_a]]
        + [team.id for team in conventional[:conventional_in_a]],
        "B": [team.id for team in ncm[ncm_in_a:]]
        + [team.id for team in conventional[conventional_in_a:]],
    }
    return groups


def _cyclic_home(left: str, right: str, order: list[str]) -> bool:
    """Orient a K5 edge so every team gets two home and two away games.

    On five vertices the cyclic orientation ``i -> i+1, i+2 (mod 5)`` gives
    every vertex out-degree two.  Using it for all group fixtures means a team
    never gets a side advantage in the round-robin.
    """
    try:
        a = order.index(left)
        b = order.index(right)
    except ValueError as exc:  # pragma: no cover - guarded by callers
        raise ValueError(f"{left!r} or {right!r} is not in the group order") from exc
    distance = (b - a) % len(order)
    return distance in (1, 2)


def match_seed(base: int, fixture_id: str) -> int:
    """A stable, unique match seed from a tournament seed and fixture id."""
    # A tiny FNV-1a keeps fixtures independent of list order while remaining
    # deterministic across processes and Python versions.
    value = 2166136261
    for char in f"{base}:{fixture_id}":
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return (int(base) * 1_000_003 + value) % 2_147_483_647


def build_group_fixtures(
    groups: dict[str, list[str]],
    round_maps: list[str],
    seed: int,
) -> list[Fixture]:
    """Build a five-round round-robin for each group, one bye per team."""
    if len(round_maps) < GROUP_SIZE:
        raise ValueError(f"five group rounds need at least five maps, got {len(round_maps)}")
    fixtures: list[Fixture] = []
    for group_index, group in enumerate(GROUP_LABELS):
        order = list(groups[group])
        if len(order) != GROUP_SIZE:
            raise ValueError(f"group {group} has {len(order)} teams, expected {GROUP_SIZE}")
        slots: list[str | None] = order + [None]
        for round_index in range(len(slots) - 1):
            pairs: list[tuple[str, str]] = []
            for index in range(len(slots) // 2):
                left = slots[index]
                right = slots[len(slots) - 1 - index]
                if left is None or right is None:
                    continue
                home, away = (left, right) if _cyclic_home(left, right, order) else (right, left)
                pairs.append((home, away))
            for pair_index, (home, away) in enumerate(pairs):
                fixture_id = f"g{group}-r{round_index + 1}-{home}-{away}"
                fixtures.append(
                    Fixture(
                        id=fixture_id,
                        stage="group",
                        round=round_index + 1,
                        group=group,
                        map=round_maps[round_index],
                        home=home,
                        away=away,
                        seed=match_seed(seed + group_index * 1000, fixture_id),
                    )
                )
            slots = [slots[0], slots[-1], *slots[1:-1]]
    return fixtures


def build_semifinal_fixtures(
    tables: dict[str, list[dict[str, Any]]],
    knockout_maps: list[str],
    seed: int,
) -> list[Fixture]:
    """Cross-group semifinals: A1 v B2 and B1 v A2."""
    if len(knockout_maps) < 4:
        raise ValueError("the knockout stage needs four maps")
    if any(len(tables.get(group, [])) < 2 for group in GROUP_LABELS):
        raise ValueError("both groups need at least two ranked teams")
    a1 = tables["A"][0]["team"]
    a2 = tables["A"][1]["team"]
    b1 = tables["B"][0]["team"]
    b2 = tables["B"][1]["team"]
    pairs = [("sf1", a1, b2, knockout_maps[0]), ("sf2", b1, a2, knockout_maps[1])]
    return [
        Fixture(
            id=fixture_id,
            stage="semifinal",
            round=1,
            group=None,
            map=map_name,
            home=home,
            away=away,
            seed=match_seed(seed + 5000, fixture_id),
        )
        for fixture_id, home, away, map_name in pairs
    ]


def build_final_fixtures(
    semifinals: dict[str, MatchResult],
    knockout_maps: list[str],
    seed: int,
    *,
    team_seeds: dict[str, int] | None = None,
) -> list[Fixture]:
    """Third-place match and final from the two semifinal results."""
    if len(knockout_maps) < 4:
        raise ValueError("the knockout stage needs four maps")
    try:
        first = semifinals["sf1"]
        second = semifinals["sf2"]
    except KeyError as exc:
        raise ValueError(f"missing semifinal result {exc.args[0]!r}") from None
    for result in (first, second):
        if result.winner is None:
            raise ValueError(f"semifinal {result.fixture_id} has no winner")

    first_loser = first.away if first.winner == first.home else first.home
    second_loser = second.away if second.winner == second.home else second.home
    first_winner = first.winner
    second_winner = second.winner
    assert first_winner is not None and second_winner is not None

    def seeded_pair(left: str, right: str) -> tuple[str, str]:
        seeds = team_seeds or {}
        return (left, right) if seeds.get(left, 0) <= seeds.get(right, 0) else (right, left)

    third_home, third_away = seeded_pair(first_loser, second_loser)
    final_home, final_away = seeded_pair(first_winner, second_winner)
    return [
        Fixture(
            id="third",
            stage="third_place",
            round=2,
            group=None,
            map=knockout_maps[2],
            home=third_home,
            away=third_away,
            seed=match_seed(seed + 5200, "third"),
        ),
        Fixture(
            id="final",
            stage="final",
            round=2,
            group=None,
            map=knockout_maps[3],
            home=final_home,
            away=final_away,
            seed=match_seed(seed + 5300, "final"),
        ),
    ]


# ---------------------------------------------------------------------------
# standings


def _new_row(team: str, seed: int) -> dict[str, Any]:
    return {
        "team": team,
        "seed": seed,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "score_for": 0.0,
        "score_against": 0.0,
        "score_diff": 0.0,
    }


def _result_points(winner: str | None, team: str) -> int:
    if winner is None:
        return POINTS_DRAW
    return POINTS_WIN if winner == team else 0


def group_table(
    group: str,
    teams: dict[str, Team],
    fixtures: Iterable[Fixture],
    results: dict[str, MatchResult],
) -> list[dict[str, Any]]:
    """Rank one group by points, board-score difference, then head-to-head."""
    # Use the fixture appearance order, not set iteration.
    ordered: list[str] = []
    for fixture in fixtures:
        if fixture.stage != "group" or fixture.group != group:
            continue
        for team_id in (fixture.home, fixture.away):
            if team_id not in ordered:
                ordered.append(team_id)
    if not ordered:
        # A group with no fixtures still has a draw order in the caller; this
        # branch is for defensive use only.
        return []
    rows = {team_id: _new_row(team_id, teams[team_id].seed) for team_id in ordered}
    for fixture in fixtures:
        if fixture.stage != "group" or fixture.group != group:
            continue
        result = results.get(fixture.id)
        if result is None:
            continue
        home = result.home
        away = result.away
        home_score = float(result.score.get(home, 0.0))
        away_score = float(result.score.get(away, 0.0))
        row_home = rows.setdefault(home, _new_row(home, teams[home].seed))
        row_away = rows.setdefault(away, _new_row(away, teams[away].seed))
        row_home["played"] += 1
        row_away["played"] += 1
        row_home["score_for"] += home_score
        row_home["score_against"] += away_score
        row_away["score_for"] += away_score
        row_away["score_against"] += home_score
        if result.winner is None:
            row_home["draws"] += 1
            row_away["draws"] += 1
            row_home["points"] += POINTS_DRAW
            row_away["points"] += POINTS_DRAW
        else:
            winner_row = rows[result.winner]
            loser = away if result.winner == home else home
            winner_row["wins"] += 1
            winner_row["points"] += POINTS_WIN
            rows[loser]["losses"] += 1

    for row in rows.values():
        row["score_diff"] = row["score_for"] - row["score_against"]

    tied_points: dict[int, set[str]] = {}
    for row in rows.values():
        tied_points.setdefault(row["points"], set()).add(row["team"])

    def head_to_head(team_id: str) -> int:
        tied = tied_points.get(rows[team_id]["points"], set())
        if len(tied) < 2:
            return 0
        total = 0
        for fixture in fixtures:
            if fixture.stage != "group" or fixture.group != group:
                continue
            if fixture.home not in tied or fixture.away not in tied:
                continue
            result = results.get(fixture.id)
            if result is None:
                continue
            if team_id in (result.home, result.away):
                total += _result_points(result.winner, team_id)
        return total

    ordered_rows = sorted(
        rows.values(),
        key=lambda row: (
            -row["points"],
            -row["score_diff"],
            -row["score_for"],
            -head_to_head(row["team"]),
            -row["wins"],
            row["seed"],
        ),
    )
    for index, row in enumerate(ordered_rows, start=1):
        row["pos"] = index
    return ordered_rows


# ---------------------------------------------------------------------------
# knockout resolution


def resolve_knockout(
    fixture: Fixture,
    home_side: str,
    away_side: str,
    engine_winner: str | None,
    score: dict[str, float],
    seed_order: dict[str, int] | None = None,
) -> tuple[str, str, str]:
    """Return ``(winner_team, winner_side, decided_by)`` for a knockout tie.

    A game that reaches the turn cap is decided by board score, then by the
    stored group seed.  This function never leaves a knockout match without a
    winner.
    """
    if engine_winner in (fixture.home, fixture.away):
        return (
            engine_winner,
            home_side if engine_winner == fixture.home else away_side,
            "victory",
        )
    home_score = float(score.get(fixture.home, 0.0))
    away_score = float(score.get(fixture.away, 0.0))
    if home_score > away_score:
        return fixture.home, home_side, "board"
    if away_score > home_score:
        return fixture.away, away_side, "board"
    seeds = seed_order or {}
    if seeds.get(fixture.home, 99) <= seeds.get(fixture.away, 99):
        return fixture.home, home_side, "seed"
    return fixture.away, away_side, "seed"


# ---------------------------------------------------------------------------
# aggregation


def _team_by_id(teams: Iterable[Team]) -> dict[str, Team]:
    return {team.id: team for team in teams}


def aggregate_team_stats(
    teams: Iterable[Team],
    results: Iterable[MatchResult],
) -> dict[str, dict[str, Any]]:
    """Aggregate results and agent counters per team."""
    team_map = _team_by_id(teams)
    totals: dict[str, dict[str, Any]] = {}
    for team in team_map.values():
        totals[team.id] = {
            "team": team.id,
            "name": team.name,
            "kind": team.kind,
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "score_for": 0.0,
            "score_against": 0.0,
            "score_diff": 0.0,
            "turns": 0,
            "orders": 0,
            "steps": 0,
            "samples": 0,
            "model_orders": 0,
            "generated": 0,
            "repaired": 0,
            "constrained": 0,
            "tactical": 0,
            "fallbacks": 0,
            "actions": Counter(),
            "builds": Counter(),
        }

    for result in results:
        for team_id in (result.home, result.away):
            row = totals.get(team_id)
            if row is None:
                continue
            row["games"] += 1
            row["turns"] += int(result.turns or 0)
            row["score_for"] += float(result.score.get(team_id, 0.0))
            other = result.away if team_id == result.home else result.home
            row["score_against"] += float(result.score.get(other, 0.0))
            if result.winner is None:
                row["draws"] += 1
                row["points"] += POINTS_DRAW
            elif result.winner == team_id:
                row["wins"] += 1
                row["points"] += POINTS_WIN
            else:
                row["losses"] += 1

            stats = result.stats.get(team_id) or {}
            row["orders"] += int(stats.get("orders", 0) or 0)
            row["steps"] += int(stats.get("steps", 0) or 0)
            row["samples"] += int(stats.get("samples", 0) or 0)
            row["model_orders"] += int(stats.get("model_orders", 0) or 0)
            row["generated"] += int(stats.get("generated", 0) or 0)
            row["repaired"] += int(stats.get("repaired", 0) or 0)
            row["constrained"] += int(stats.get("constrained", 0) or 0)
            row["tactical"] += int(stats.get("tactical", 0) or 0)
            row["fallbacks"] += int(stats.get("fallbacks", 0) or 0)
            actions = stats.get("actions")
            if isinstance(actions, dict):
                row["actions"].update({str(k): int(v) for k, v in actions.items()})
            builds = stats.get("builds")
            if isinstance(builds, dict):
                row["builds"].update({str(k): int(v) for k, v in builds.items()})

    for row in totals.values():
        row["score_diff"] = row["score_for"] - row["score_against"]
        row["score_for"] = round(row["score_for"], 2)
        row["score_against"] = round(row["score_against"], 2)
        row["score_diff"] = round(row["score_diff"], 2)
        total_orders = max(1, row["orders"])
        total_steps = max(1, row["steps"] or row["orders"])
        row["action_share"] = {
            key: round(value / total_orders, 4)
            for key, value in sorted(row["actions"].items())
        }
        row["actions"] = dict(row["actions"])
        row["builds"] = dict(row["builds"])
        if row["kind"] == "ncm":
            row["provenance"] = {
                "legal_share": round(
                    (row["generated"] + row["repaired"] + row["constrained"])
                    / total_steps,
                    4,
                ),
                "fallback_rate": round(row["fallbacks"] / total_steps, 4),
                "tactical_share": round(row["tactical"] / total_steps, 4),
                "model_share": round(row["model_orders"] / total_steps, 4),
                "sample_legal_rate": round(
                    row["generated"] / max(1, row["samples"]), 4
                ),
            }
        else:
            row["provenance"] = None
    return totals


def aggregate_matchups(
    teams: Iterable[Team],
    results: Iterable[MatchResult],
) -> dict[str, Any]:
    """NCM-vs-conventional, NCM-vs-NCM and conventional-vs-conventional splits."""
    team_map = _team_by_id(teams)
    out: dict[str, Any] = {
        "ncm_vs_conventional": {
            "games": 0,
            "ncm_wins": 0,
            "conventional_wins": 0,
            "draws": 0,
            "ncm_score": 0.0,
            "conventional_score": 0.0,
        },
        "ncm_vs_ncm": {"games": 0, "draws": 0},
        "conventional_vs_conventional": {"games": 0, "draws": 0},
    }
    for result in results:
        home = team_map[result.home]
        away = team_map[result.away]
        kinds = {home.kind, away.kind}
        ncm_is_home = home.kind == "ncm"
        ncm_is_away = away.kind == "ncm"
        if ncm_is_home and ncm_is_away:
            out["ncm_vs_ncm"]["games"] += 1
            if result.winner is None:
                out["ncm_vs_ncm"]["draws"] += 1
            continue
        if not ncm_is_home and not ncm_is_away:
            out["conventional_vs_conventional"]["games"] += 1
            if result.winner is None:
                out["conventional_vs_conventional"]["draws"] += 1
            continue
        row = out["ncm_vs_conventional"]
        row["games"] += 1
        ncm_team = result.home if ncm_is_home else result.away
        conventional = result.away if ncm_is_home else result.home
        row["ncm_score"] += float(result.score.get(ncm_team, 0.0))
        row["conventional_score"] += float(result.score.get(conventional, 0.0))
        if result.winner is None:
            row["draws"] += 1
        elif result.winner == ncm_team:
            row["ncm_wins"] += 1
        else:
            row["conventional_wins"] += 1
    for row in out.values():
        for key in ("ncm_score", "conventional_score"):
            if key in row:
                row[key] = round(float(row[key]), 2)
    return out


def aggregate_map_summary(results: Iterable[MatchResult]) -> dict[str, dict[str, Any]]:
    """Per-map games, decisive rate, turns and score difference."""
    summary: dict[str, dict[str, Any]] = {}
    for result in results:
        row = summary.setdefault(
            result.map,
            {
                "map": result.map,
                "games": 0,
                "decisive": 0,
                "draws": 0,
                "turns": 0,
                "score_delta": 0.0,
            },
        )
        row["games"] += 1
        row["turns"] += int(result.turns or 0)
        row["score_delta"] += abs(
            float(result.score.get(result.home, 0.0))
            - float(result.score.get(result.away, 0.0))
        )
        if result.winner is None:
            row["draws"] += 1
        else:
            row["decisive"] += 1
    for row in summary.values():
        games = max(1, row["games"])
        row["avg_turns"] = round(row["turns"] / games, 1)
        row["avg_score_delta"] = round(row["score_delta"] / games, 2)
    return summary


# ---------------------------------------------------------------------------
# dashboard compatibility


def dashboard_meta(
    result: MatchResult,
    teams: Iterable[Team],
    batch_id: str,
) -> dict[str, Any]:
    """Build the meta-sidecar fields the NCM dashboard needs.

    ``source='play'`` keeps the game in the dashboard's Games tab.  The extra
    ``world_cup`` and ``team_labels`` fields are read by the small compatibility
    branch in ``tbs_ncm/gamefeed.py``.
    """
    team_map = _team_by_id(teams)
    home = team_map[result.home]
    away = team_map[result.away]
    labels = {result.home_side: home.name, result.away_side: away.name}
    ncm_sides = [
        result.home_side if home.kind == "ncm" else None,
        result.away_side if away.kind == "ncm" else None,
    ]
    ncm_sides = [side for side in ncm_sides if side is not None]
    if len(ncm_sides) == 2:
        ncm_side = "both"
    elif ncm_sides:
        ncm_side = ncm_sides[0]
    else:
        ncm_side = None
    if home.kind == "ncm" and away.kind == "ncm":
        opponent = "ncm"
    elif home.kind != "ncm" and away.kind == "ncm":
        opponent = home.kind
    elif home.kind == "ncm":
        opponent = away.kind
    else:
        opponent = "conventional"
    winner = team_map.get(result.winner) if result.winner else None
    world = {
        "stage": result.stage,
        "fixture": result.fixture_id,
        "home": result.home,
        "away": result.away,
        "home_name": home.name,
        "away_name": away.name,
        "home_side": result.home_side,
        "away_side": result.away_side,
        "winner": result.winner,
        "winner_name": winner.name if winner is not None else None,
        "winner_side": result.winner_side,
        "decided_by": result.decided_by,
        "score": dict(result.score),
        "batch": batch_id,
    }
    return {
        "source": "play",
        "label": f"World Cup {result.stage}: {home.name} vs {away.name} - {result.map}",
        "batch": batch_id,
        "team_labels": labels,
        "world_cup": world,
        "ncm_side": ncm_side,
        "opponent": opponent,
        "opponent_kind": opponent,
        "learner_persona": home.name,
        "opponent_persona": away.name,
    }


# ---------------------------------------------------------------------------
# report


def _score_text(result: MatchResult) -> str:
    return f"{result.score.get(result.home, 0.0):.1f} - {result.score.get(result.away, 0.0):.1f}"


def _winner_text(result: MatchResult, teams: dict[str, Team]) -> str:
    if result.winner is None:
        return "draw"
    team = teams.get(result.winner)
    return team.name if team is not None else result.winner


def _result_rows(
    fixtures: Iterable[Fixture],
    results: dict[str, MatchResult],
    teams: dict[str, Team],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        result = results.get(fixture.id)
        if result is None:
            rows.append(
                {
                    "fixture": fixture.id,
                    "stage": fixture.stage,
                    "round": fixture.round,
                    "group": fixture.group,
                    "map": fixture.map,
                    "home": fixture.home,
                    "away": fixture.away,
                    "home_name": teams[fixture.home].name,
                    "away_name": teams[fixture.away].name,
                    "score": None,
                    "winner": None,
                    "winner_name": None,
                    "decided_by": "unplayed",
                    "turns": 0,
                    "seconds": 0.0,
                    "save": None,
                }
            )
            continue
        rows.append(
            {
                "fixture": fixture.id,
                "stage": fixture.stage,
                "round": fixture.round,
                "group": fixture.group,
                "map": fixture.map,
                "home": fixture.home,
                "away": fixture.away,
                "home_name": teams[fixture.home].name,
                "away_name": teams[fixture.away].name,
                "score": _score_text(result),
                "winner": result.winner,
                "winner_name": _winner_text(result, teams),
                "decided_by": result.decided_by,
                "turns": result.turns,
                "seconds": result.seconds,
                "save": result.save,
            }
        )
    return rows


def _group_table_lines(
    group: str,
    rows: list[dict[str, Any]],
    teams: dict[str, Team],
) -> list[str]:
    lines = [
        f"### Group {group}",
        "",
        "| Pos | Team | Kind | P | W | D | L | PF | PA | PD | Pts |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        team = teams[row["team"]]
        lines.append(
            f"| {row['pos']} | {team.name} | {team.kind} | {row['played']} | "
            f"{row['wins']} | {row['draws']} | {row['losses']} | "
            f"{row['score_for']:.1f} | {row['score_against']:.1f} | "
            f"{row['score_diff']:+.1f} | {row['points']} |"
        )
    lines.append("")
    return lines


def _result_lines(
    title: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Round | Map | Home | Score | Away | Winner | Decided by | Turns |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        score = row["score"] or "-"
        lines.append(
            f"| {row['round']} | {row['map']} | {row['home_name']} | {score} | "
            f"{row['away_name']} | {row['winner_name'] or '-'} | "
            f"{row['decided_by']} | {row['turns']} |"
        )
    lines.append("")
    return lines


def _summary_state(state: dict[str, Any]) -> dict[str, Any]:
    teams = [Team(**item) for item in state.get("teams", [])]
    team_map = _team_by_id(teams)
    fixtures = [
        Fixture(**item)
        for item in state.get("group_fixtures", []) + state.get("knockout_fixtures", [])
    ]
    results = {
        key: MatchResult.from_dict(value)
        for key, value in (state.get("results") or {}).items()
    }
    tables = {
        group: group_table(group, team_map, fixtures, results)
        for group in state.get("groups", {})
    }
    aggregates = aggregate_team_stats(teams, results.values())
    matchups = aggregate_matchups(teams, results.values())
    map_summary = aggregate_map_summary(results.values())
    final = results.get("final")
    third = results.get("third")
    champion = team_map.get(final.winner) if final and final.winner else None
    runner_up = None
    if final and final.winner:
        runner_up = team_map.get(final.away if final.winner == final.home else final.home)
    third_place = team_map.get(third.winner) if third and third.winner else None
    return {
        "teams": teams,
        "team_map": team_map,
        "fixtures": fixtures,
        "results": results,
        "tables": tables,
        "aggregates": aggregates,
        "matchups": matchups,
        "map_summary": map_summary,
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place,
    }


def report_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Return the complete JSON payload used by the docs and the dashboard."""
    summary = _summary_state(state)
    teams = summary["team_map"]
    results = summary["results"]
    games = []
    for result in results.values():
        games.append(
            {
                "fixture": result.fixture_id,
                "stage": result.stage,
                "group": result.group,
                "seed": result.seed,
                "map": result.map,
                "pair": [teams[result.home].name, teams[result.away].name],
                "home": result.home,
                "away": result.away,
                "winner": result.winner,
                "winner_name": _winner_text(result, teams),
                "decided_by": result.decided_by,
                "score": dict(result.score),
                "turns": result.turns,
                "seconds": result.seconds,
                "orders": result.orders,
                "replay_matches": result.replay_matches,
                "save": result.save,
                "stats": result.stats,
            }
        )
    games.sort(key=lambda row: (str(row["stage"]), row["fixture"]))
    payload = dict(state)
    settings = state.get("settings") or {}
    payload["generator"] = settings.get("generator")
    payload["decode"] = settings.get("decode")
    payload["max_turns"] = settings.get("max_turns")
    payload["prefix_len"] = settings.get("prefix_len")
    map_schedule = state.get("maps") or {}
    payload["map_schedule"] = map_schedule
    payload["maps"] = (
        list(map_schedule.get("group") or [])
        if isinstance(map_schedule, dict)
        else list(map_schedule)
    )
    payload["teams"] = [asdict(team) for team in summary["teams"]]
    payload["tables"] = summary["tables"]
    payload["aggregates"] = summary["aggregates"]
    payload["matchups"] = summary["matchups"]
    payload["map_summary"] = summary["map_summary"]
    payload["champion"] = summary["champion"].name if summary["champion"] else None
    payload["runner_up"] = summary["runner_up"].name if summary["runner_up"] else None
    payload["third_place"] = summary["third_place"].name if summary["third_place"] else None
    payload["games"] = games
    payload["matches"] = len(games)
    payload["ncm_quality"] = {
        key: value
        for key, value in summary["aggregates"].items()
        if value.get("kind") == "ncm"
    }
    payload["conventional_quality"] = {
        key: value
        for key, value in summary["aggregates"].items()
        if value.get("kind") != "ncm"
    }
    ncm_names = [team.name for team in summary["teams"] if team.kind == "ncm"]
    heuristic_names = [
        {"name": team.name, "seed": team.meta.get("heuristic_seed")}
        for team in summary["teams"]
        if team.kind == "heuristic"
    ]
    payload["ncm_personas"] = ncm_names
    payload["personas"] = ncm_names
    payload["heuristics"] = heuristic_names
    payload["generated_at"] = state.get("updated") or state.get("created")
    return payload


def render_markdown(state: dict[str, Any]) -> str:
    """Render the human report for a (possibly partial) tournament."""
    payload = report_payload(state)
    summary = _summary_state(state)
    teams = summary["team_map"]
    results = summary["results"]
    tables = summary["tables"]
    aggregates = summary["aggregates"]
    matchups = summary["matchups"]
    map_summary = summary["map_summary"]
    settings = state.get("settings") or {}
    group_rows = _result_rows(
        [fixture for fixture in summary["fixtures"] if fixture.stage == "group"],
        results,
        teams,
    )
    knockout_rows = _result_rows(
        [fixture for fixture in summary["fixtures"] if fixture.stage != "group"],
        results,
        teams,
    )
    lines = [
        "# TBS World Cup 2026",
        "",
        f"**{len(teams)} teams · {len(state.get('groups', {}))} groups · "
        f"{len(results)}/{state.get('format', {}).get('total_matches', 24)} matches played**",
        "",
        "The five NCM personas, four randomized conventional AIs and the greedy "
        "baseline play a seeded two-group World Cup on the new `_m` maps. Group "
        "matches award 3/1/0 points; the top two in each group reach the "
        "cross-group semifinals. `PF`/`PA` are board scores (2 × sites + live "
        "units + HP/100); at the turn cap they are the goal difference, and a "
        "knockout game level on board score is decided by the better group seed.",
        "",
        f"- Generator: `{settings.get('generator', 'mixed_generator.pt')}`",
        f"- Decode: `{settings.get('decode', 'legal')}`",
        f"- Prefix length: `{settings.get('prefix_len', 4096)}`",
        f"- Turn cap: `{settings.get('max_turns')}` · steps per turn: `{settings.get('max_steps')}`",
        f"- Tournament seed: `{state.get('seed')}`",
        f"- Dashboard batch: `{state.get('batch', {}).get('id', '')}`",
        "",
        "## Draw",
        "",
        "| Group | Teams |",
        "|---|---|",
    ]
    for group, team_ids in state.get("groups", {}).items():
        names = ", ".join(teams[team_id].name for team_id in team_ids)
        lines.append(f"| {group} | {names} |")
    lines += [
        "",
        f"Group map order: {', '.join(f'`{item}`' for item in state.get('maps', {}).get('group', []))}.",
        "",
        "`last_stand_m` is the asymmetric attack/defence map; it appears in the "
        "group rotation with sides alternating, while the knockout uses the four "
        "hex-true mirrored maps.",
        "",
        "## Group stage",
        "",
    ]
    for group in state.get("groups", {}):
        lines += _group_table_lines(group, tables.get(group, []), teams)
    lines += ["## Group results", ""]
    for group in state.get("groups", {}):
        rows = [row for row in group_rows if row["group"] == group]
        lines += _result_lines(f"Group {group}", rows)
    lines += ["## Knockout", ""]
    if knockout_rows:
        for title, stage in (
            ("Semifinals", "semifinal"),
            ("Third place", "third_place"),
            ("Final", "final"),
        ):
            rows = [row for row in knockout_rows if row["stage"] == stage]
            if rows:
                lines += _result_lines(title, rows)
    else:
        lines += ["No knockout matches yet.", ""]

    lines += ["## Podium", ""]
    if payload.get("champion"):
        lines += [
            f"- **Champion:** {payload['champion']}",
            f"- **Runner-up:** {payload.get('runner_up') or '-'}",
            f"- **Third place:** {payload.get('third_place') or '-'}",
        ]
    else:
        lines += ["Tournament in progress."]
    lines += [
        "",
        "## NCM vs conventional AI",
        "",
        "| Split | Games | NCM wins | Conventional wins | Draws | NCM score | Conventional score |",
        "|---|---|---|---|---|---|---|",
    ]
    mixed = matchups["ncm_vs_conventional"]
    lines.append(
        f"| NCM vs conventional | {mixed['games']} | {mixed['ncm_wins']} | "
        f"{mixed['conventional_wins']} | {mixed['draws']} | "
        f"{mixed['ncm_score']:.1f} | {mixed['conventional_score']:.1f} |"
    )
    lines += [
        f"| NCM vs NCM | {matchups['ncm_vs_ncm']['games']} | - | - | "
        f"{matchups['ncm_vs_ncm']['draws']} | - | - |",
        f"| Conventional vs conventional | "
        f"{matchups['conventional_vs_conventional']['games']} | - | - | "
        f"{matchups['conventional_vs_conventional']['draws']} | - | - |",
        "",
        "## NCM quality",
        "",
        "| Team | W-D-L | Pts | Games | Orders | Legal% | Fallback% | Tactical% | Model% | Action mix |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for team_id, row in aggregates.items():
        if row["kind"] != "ncm":
            continue
        prov = row.get("provenance") or {}
        actions = ", ".join(
            f"{key} {value:.0%}" for key, value in row["action_share"].items()
        ) or "-"
        lines.append(
            f"| {row['name']} | {row['wins']}-{row['draws']}-{row['losses']} | "
            f"{row['points']} | {row['games']} | {row['orders']} | "
            f"{prov.get('legal_share', 0):.0%} | {prov.get('fallback_rate', 0):.0%} | "
            f"{prov.get('tactical_share', 0):.0%} | {prov.get('model_share', 0):.0%} | "
            f"{actions} |"
        )
    lines += [
        "",
        "## Conventional AI",
        "",
        "| Team | Kind | W-D-L | Pts | Games | Orders | Action mix | Builds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for team_id, row in aggregates.items():
        if row["kind"] == "ncm":
            continue
        actions = ", ".join(
            f"{key} {value:.0%}" for key, value in row["action_share"].items()
        ) or "-"
        builds = ", ".join(f"{key} {value}" for key, value in row["builds"].items()) or "-"
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['wins']}-{row['draws']}-{row['losses']} | "
            f"{row['points']} | {row['games']} | {row['orders']} | {actions} | {builds} |"
        )
    lines += [
        "",
        "## Maps",
        "",
        "| Map | Games | Decisive | Draws | Avg turns | Avg board-score gap |",
        "|---|---|---|---|---|---|",
    ]
    for map_name, row in sorted(map_summary.items()):
        lines.append(
            f"| {map_name} | {row['games']} | {row['decisive']} | {row['draws']} | "
            f"{row['avg_turns']} | {row['avg_score_delta']:.1f} |"
        )
    replay_ok = sum(1 for result in results.values() if result.replay_matches)
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Replay verified: **{replay_ok}/{len(results)}** completed matches.",
        f"- Match saves: `{state.get('match_dir', '')}`",
        f"- Dashboard report: `{state.get('batch', {}).get('dir', '')}`",
        f"- Resume state: `{state.get('state_path', '')}`",
        "",
        "Use `--resume` to continue an interrupted tournament and `--report-only` "
        "to rebuild this report without loading the models.",
        "",
    ]
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    """Recursively convert Counters and tuples into JSON primitives."""
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def dumps_report(state: dict[str, Any]) -> str:
    """JSON text for the docs and the dashboard batch."""
    return json.dumps(_json_safe(report_payload(state)), indent=2, ensure_ascii=False) + "\n"
