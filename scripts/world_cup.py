"""TBS World Cup: NCM personas and conventional AIs on the large ``_m`` maps.

The tournament is a 10-team, two-group World Cup followed by cross-group
semifinals, a third-place match and a final.  The five trained NCM personas use
the NCM checkout's ``mixed_generator.pt``; the other five teams are four
randomized conventional AIs from ``tbs_ncm.heuristic`` and the deterministic
greedy baseline from this repository.

Games are written where the NCM dashboard's Games tab finds them
(``<ncm-root>/artifacts/tbs/matches``) and a league report is written under
``<ncm-root>/artifacts/tbs/personas/<batch>/report.json`` for the Overview tab.
The human report and the resume state live in ``docs/world_cup``.

Run it with the NCM virtualenv so torch and the model checkpoints are available:

    C:\\AIs\\NCM\\.venv\\Scripts\\python.exe scripts\\world_cup.py --resume

Use ``--dry-run`` without loading models to see the draw and fixtures, and
``--report-only`` to rebuild the reports from the saved state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

GAME_REPO = Path(__file__).resolve().parents[1]
if str(GAME_REPO) not in sys.path:
    sys.path.insert(0, str(GAME_REPO))

from tbs.agents.greedy import GreedyAgent
from tbs.agents.league import (
    GROUP_LABELS,
    Fixture,
    MatchResult,
    Team,
    build_final_fixtures,
    build_group_fixtures,
    build_semifinal_fixtures,
    dashboard_meta,
    draw_groups,
    dumps_report,
    group_table,
    render_markdown,
    resolve_knockout,
)
from tbs.agents.protocol import AgentPlan, TurnRequest

DEFAULT_NCM_ROOT = Path(
    os.environ.get("TBS_NCM_ROOT", str(GAME_REPO.parent / "NCM_fork"))
)
DEFAULT_MAPS = (
    "border_skirmish_m",
    "mountain_pass_m",
    "archipelago_m",
    "urban_delta_m",
    "last_stand_m",
)
FAIR_MAPS = (
    "border_skirmish_m",
    "mountain_pass_m",
    "archipelago_m",
    "urban_delta_m",
)
NCM_PERSONAS = ("aggressive", "defensive", "infantry", "armor", "artillery")
PERSONA_TITLES = {
    "aggressive": "Blitz",
    "defensive": "Anvil",
    "infantry": "Line",
    "armor": "Vanguard",
    "artillery": "Siege",
}
DEFAULT_HEURISTIC_SEEDS = (900000, 900001, 900002, 900003)


class CountingGreedy(GreedyAgent):
    """GreedyAgent plus the counters ``tbs_ncm.arena.play_match`` reads.

    Greedy returns a whole-turn plan, so ``steps`` counts calls and ``orders``
    counts the individual orders it proposed.  The normal NCM match runner
    resets these counters before every game.
    """

    def __init__(self) -> None:
        self.name = "greedy"
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.steps = 0
        self.samples = 0
        self.generated = 0
        self.repaired = 0
        self.repaired_samples = 0
        self.constrained = 0
        self.tactical = 0
        self.fallbacks = 0
        self.illegal = 0
        self.rejected = 0
        self.recovered = 0
        self.state_too_long = 0
        self.generation_errors = 0
        self.lines: list[str] = []
        self.constrained_lines: list[str] = []
        self.fallback_reasons: list[str] = []
        self.repairs: Counter[str] = Counter()
        self.actions: Counter[str] = Counter()
        self.builds: Counter[str] = Counter()
        self.orders = 0
        self.model_actions: Counter[str] = Counter()
        self.model_builds: Counter[str] = Counter()
        self.model_orders = 0
        self.geometry_samples = 0
        self.enemy_distance = 0.0
        self.home_distance = 0.0
        self._last_line = ""
        self._last_order: dict | None = None

    def plan(self, request: TurnRequest) -> AgentPlan:
        plan = super().plan(request)
        self.steps += 1
        for order in plan.orders:
            action = str(order.get("action", "unknown"))
            self.orders += 1
            self.actions[action] += 1
            if action == "build":
                self.builds[str(order.get("unit", "unknown"))] += 1
        return plan


def _iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _split_list(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _safe_path(value: str | None, default: Path) -> Path:
    return Path(value) if value else default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TBS World Cup: NCM personas vs conventional AI on the _m maps"
    )
    parser.add_argument("--ncm-root", default=str(DEFAULT_NCM_ROOT))
    parser.add_argument("--persona-dir", default="")
    parser.add_argument("--generator", default="mixed_generator.pt")
    parser.add_argument("--decode", choices=("sample", "legal"), default="legal")
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-temperature", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS))
    parser.add_argument(
        "--knockout-maps",
        default=",".join(FAIR_MAPS),
        help="four fair maps for the semifinals, third-place match and final",
    )
    parser.add_argument("--heuristic-seeds", default=",".join(str(s) for s in DEFAULT_HEURISTIC_SEEDS))
    parser.add_argument("--out-dir", default=str(GAME_REPO / "docs" / "world_cup"))
    parser.add_argument(
        "--match-dir",
        default="",
        help="default: <ncm-root>/artifacts/tbs/matches (the dashboard directory)",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="",
        help="default: <ncm-root>/artifacts/tbs/personas",
    )
    parser.add_argument("--batch-name", default="")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many completed games")
    parser.add_argument(
        "--allow-asymmetric-knockout",
        action="store_true",
        help="allow last_stand_m in the knockout maps",
    )
    parser.add_argument("--verify-replay", action="store_true", default=True)
    parser.add_argument("--no-verify-replay", dest="verify_replay", action="store_false")
    return parser.parse_args(argv)


def build_teams(args: argparse.Namespace) -> list[Team]:
    teams: list[Team] = []
    for index, persona in enumerate(NCM_PERSONAS):
        teams.append(
            Team(
                id=f"ncm-{persona}",
                name=f"NCM {PERSONA_TITLES.get(persona, persona.title())}",
                kind="ncm",
                seed=index,
                title=persona,
                meta={"persona": persona},
            )
        )
    seeds = [int(token) for token in _split_list(args.heuristic_seeds)]
    if len(seeds) != 4:
        raise SystemExit("--heuristic-seeds must name exactly four seeds")
    for offset, heuristic_seed in enumerate(seeds, start=5):
        teams.append(
            Team(
                id=f"heuristic-{heuristic_seed}",
                name=f"Conventional {heuristic_seed}",
                kind="heuristic",
                seed=offset,
                title="randomized scripted AI",
                meta={"heuristic_seed": heuristic_seed},
            )
        )
    teams.append(
        Team(
            id="greedy",
            name="Conventional Greedy",
            kind="greedy",
            seed=9,
            title="deterministic baseline",
        )
    )
    return teams


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    ncm_root = Path(args.ncm_root)
    out_dir = Path(args.out_dir)
    persona_dir = _safe_path(args.persona_dir, ncm_root / "artifacts" / "tbs" / "personas")
    match_default = ncm_root / "artifacts" / "tbs" / "matches"
    match_dir = _safe_path(args.match_dir, match_default)
    dashboard_dir: Path | None = None
    if not args.no_dashboard:
        dashboard_dir = _safe_path(
            args.dashboard_dir, ncm_root / "artifacts" / "tbs" / "personas"
        )
    return {
        "ncm_root": ncm_root,
        "out_dir": out_dir,
        "persona_dir": persona_dir,
        "match_dir": match_dir,
        "dashboard_dir": dashboard_dir,
    }


def _shuffled(values: list[str], seed: int) -> list[str]:
    import random

    result = list(values)
    random.Random(seed).shuffle(result)
    return result


def create_state(
    *,
    teams: list[Team],
    args: argparse.Namespace,
    paths: dict[str, Path | None],
    batch_name: str,
) -> dict[str, Any]:
    maps = _split_list(args.maps)
    if len(maps) != 5 or len(set(maps)) != 5:
        raise SystemExit("the World Cup needs exactly five distinct --maps entries")
    knockout = _split_list(args.knockout_maps)
    if args.allow_asymmetric_knockout and "last_stand_m" in maps:
        # Keep four venues but let the asymmetric map replace the last fair
        # venue when the caller explicitly opts in.
        if "last_stand_m" not in knockout:
            knockout = knockout[:3] + ["last_stand_m"]
    if len(knockout) != 4 or len(set(knockout)) != 4:
        raise SystemExit("--knockout-maps must name four distinct maps")
    if not set(knockout).issubset(set(maps)):
        raise SystemExit("--knockout-maps must be a subset of --maps")

    group_maps = _shuffled(maps, args.seed + 11)
    knockout_maps = _shuffled(knockout, args.seed + 23)
    groups = draw_groups(teams, args.seed)
    fixtures = build_group_fixtures(groups, group_maps, args.seed)
    batch_id = f"personas/{batch_name}"
    match_dir = paths["match_dir"]
    dashboard_dir = paths["dashboard_dir"]
    batch_dir = (dashboard_dir / batch_name) if dashboard_dir is not None else None
    state: dict[str, Any] = {
        "schema": 1,
        "tournament": "TBS World Cup 2026",
        "created": _iso_now(),
        "updated": _iso_now(),
        "started_epoch": time.time(),
        "seconds": 0.0,
        "seed": args.seed,
        "teams": [asdict(team) for team in teams],
        "groups": groups,
        "group_fixtures": [fixture.to_dict() for fixture in fixtures],
        "knockout_fixtures": [],
        "results": {},
        "maps": {"group": group_maps, "knockout": knockout_maps},
        "format": {
            "teams": len(teams),
            "groups": len(GROUP_LABELS),
            "group_size": 5,
            "qualifiers_per_group": 2,
            "total_matches": 24,
        },
        "settings": {
            "generator": args.generator,
            "decode": args.decode,
            "prefix_len": args.prefix_len,
            "max_turns": args.max_turns,
            "max_steps": args.max_steps,
            "verify_replay": bool(args.verify_replay),
            "heuristic_seeds": [int(token) for token in _split_list(args.heuristic_seeds)],
        },
        "batch": {
            "id": batch_id,
            "name": batch_name,
            "dir": str(batch_dir) if batch_dir is not None else "",
        },
        "match_dir": str(match_dir) if match_dir is not None else "",
        "state_path": str(Path(paths["out_dir"]) / "tournament.json"),  # type: ignore[arg-type]
        "out_dir": str(paths["out_dir"]),
    }
    return state


def load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read tournament state {path}: {exc}") from None
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise SystemExit(f"{path} is not a supported World Cup state file")
    return data


def write_state(state: dict[str, Any]) -> None:
    state["updated"] = _iso_now()
    state["seconds"] = round(time.time() - float(state.get("started_epoch", time.time())), 1)
    path = Path(str(state["state_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_reports(state: dict[str, Any]) -> None:
    out_dir = Path(str(state["out_dir"]))
    markdown = render_markdown(state)
    payload = dumps_report(state)
    write_text(out_dir / "report.md", markdown)
    write_text(out_dir / "report.json", payload)

    batch = state.get("batch") or {}
    batch_dir = str(batch.get("dir") or "")
    if not batch_dir:
        return
    dashboard_dir = Path(batch_dir)
    # Only write into the NCM checkout when the batch directory is real.  A
    # report-only run on a machine without the checkout must still work.  A
    # custom --dashboard-dir may have one missing intermediate directory, so
    # create it when its grandparent exists.
    parent = dashboard_dir.parent
    if not parent.is_dir():
        if not parent.parent.is_dir():
            return
        dashboard_dir.mkdir(parents=True, exist_ok=True)
    write_text(dashboard_dir / "report.md", markdown)
    write_text(dashboard_dir / "report.json", payload)


def state_fixtures(state: dict[str, Any]) -> list[Fixture]:
    return [
        Fixture(**item)
        for item in state.get("group_fixtures", []) + state.get("knockout_fixtures", [])
    ]


def state_results(state: dict[str, Any]) -> dict[str, MatchResult]:
    return {
        key: MatchResult.from_dict(value)
        for key, value in (state.get("results") or {}).items()
    }


def state_teams(state: dict[str, Any]) -> dict[str, Team]:
    return {item["id"]: Team(**item) for item in state.get("teams", [])}


def _team_for_side(fixture: Fixture, side: str) -> str:
    return fixture.home if side == "A" else fixture.away


def _result_from_match(
    *,
    fixture: Fixture,
    match: dict[str, Any],
    teams: dict[str, Team],
    stats_by_side: dict[str, dict[str, Any]],
    score_by_side: dict[str, float],
    sides: dict[str, str],
    seed_order: dict[str, int] | None = None,
) -> MatchResult:
    home_side = sides["home"]
    away_side = sides["away"]
    winner_side = match.get("winner") or None
    winner = None
    if winner_side == home_side:
        winner = fixture.home
    elif winner_side == away_side:
        winner = fixture.away

    if fixture.stage != "group":
        score_by_team = {
            fixture.home: float(score_by_side.get(home_side, 0.0)),
            fixture.away: float(score_by_side.get(away_side, 0.0)),
        }
        winner, winner_side, decided_by = resolve_knockout(
            fixture,
            home_side,
            away_side,
            winner,
            score_by_team,
            seed_order or {team_id: team.seed for team_id, team in teams.items()},
        )
    else:
        decided_by = "victory" if winner is not None else "draw"

    stats = {
        fixture.home: stats_by_side.get(home_side, {}),
        fixture.away: stats_by_side.get(away_side, {}),
    }
    return MatchResult(
        fixture_id=fixture.id,
        stage=fixture.stage,
        group=fixture.group,
        map=fixture.map,
        home=fixture.home,
        away=fixture.away,
        home_side=home_side,
        away_side=away_side,
        winner=winner,
        winner_side=winner_side if winner is not None else None,
        decided_by=decided_by,
        score={
            fixture.home: float(score_by_side.get(home_side, 0.0)),
            fixture.away: float(score_by_side.get(away_side, 0.0)),
        },
        status=str(match.get("status", "")),
        turns=int(match.get("turns", 0) or 0),
        seconds=float(match.get("seconds", 0.0) or 0.0),
        orders=int(match.get("orders", 0) or 0),
        replay_matches=match.get("replay_matches"),
        stats=stats,
        save=str(match.get("save", "")),
        orders_path=str(match.get("orders_path", "")),
        meta_path=str(match.get("meta_path", "")),
        seed=fixture.seed,
    )


def _match_output(match_dir: Path, fixture: Fixture) -> Path:
    return match_dir / (
        f"worldcup-{fixture.id}-{fixture.home}-vs-{fixture.away}-"
        f"{fixture.map}-seed{fixture.seed}.json"
    )


def _agent_for_fixture(
    fixture: Fixture,
    agents: dict[str, Any],
    teams: dict[str, Team],
) -> tuple[Any, Any, str, str, dict[str, str]]:
    """Return learner, opponent, learner_side, opponent_side and side map.

    When exactly one side is NCM, the NCM agent is passed as the learner so the
    dashboard's quality panel always shows the model's order provenance.
    """
    home_ncm = teams[fixture.home].kind == "ncm"
    away_ncm = teams[fixture.away].kind == "ncm"
    if home_ncm and not away_ncm:
        return agents[fixture.home], agents[fixture.away], "A", "B", {"home": "A", "away": "B"}
    if away_ncm and not home_ncm:
        return agents[fixture.away], agents[fixture.home], "B", "A", {"home": "A", "away": "B"}
    return agents[fixture.home], agents[fixture.away], "A", "B", {"home": "A", "away": "B"}


def play_fixture(
    state: dict[str, Any],
    fixture: Fixture,
    agents: dict[str, Any],
    *,
    match_dir: Path,
) -> MatchResult:
    from tbs_ncm.arena import play_match

    teams = state_teams(state)
    learner, opponent, learner_side, opponent_side, sides = _agent_for_fixture(
        fixture, agents, teams
    )
    out = _match_output(match_dir, fixture)
    match = play_match(
        learner,
        opponent,
        seed=fixture.seed,
        learner_side=learner_side,
        max_turns=int(state["settings"]["max_turns"]),
        max_steps=int(state["settings"]["max_steps"]),
        scenario_path=GAME_REPO / "data" / "maps" / f"{fixture.map}.json",
        out=out,
        verify_replay=bool(state.get("settings", {}).get("verify_replay", True)),
    )
    match_by_side = {
        learner_side: match.get("learner", {}),
        opponent_side: match.get("opponent", {}),
    }
    board = match.get("board") or {}
    score_by_side = {
        side: float((board.get(side) or {}).get("score", 0.0))
        for side in ("A", "B")
    }
    result = _result_from_match(
        fixture=fixture,
        match=match,
        teams=teams,
        stats_by_side=match_by_side,
        score_by_side=score_by_side,
        sides=sides,
        seed_order=state.get("knockout_seed"),
    )

    meta_path = Path(result.meta_path)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        batch_id = str(state["batch"]["id"])
        meta.update(dashboard_meta(result, teams.values(), batch_id))
        meta["world_cup"]["score"] = {
            teams[team_id].name: round(score, 2)
            for team_id, score in result.score.items()
            if team_id in teams
        }
        write_text(meta_path, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return result


def load_agents(state: dict[str, Any], args: argparse.Namespace, paths: dict[str, Path | None]) -> dict[str, Any]:
    ncm_root = Path(str(paths["ncm_root"]))
    if str(ncm_root) not in sys.path:
        sys.path.insert(0, str(ncm_root))
    # The GPT-2 tokenizer warns when a long compact state exceeds its nominal
    # 1024-token model_max_length even though the NCM itself runs at 4096.  Keep
    # that harmless warning out of the run log so a successful run does not look
    # like a PowerShell native-command failure.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from tbs_ncm.arena import load_agent
        from tbs_ncm.heuristic import HeuristicAgent
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the NCM player from {ncm_root}: {exc}\n"
            "run this script with C:\\AIs\\NCM\\.venv\\Scripts\\python.exe "
            "or point --ncm-root at the NCM checkout"
        ) from None

    persona_dir = Path(str(paths["persona_dir"]))
    agents: dict[str, Any] = {}
    teams = state_teams(state)
    for team in teams.values():
        if team.kind == "ncm":
            persona = str(team.meta.get("persona") or team.title)
            checkpoint = persona_dir / persona
            if not (checkpoint / args.generator).is_file():
                raise SystemExit(f"missing NCM checkpoint {checkpoint / args.generator}")
            print(f"loading NCM {persona} ...", flush=True)
            agents[team.id] = load_agent(
                checkpoint,
                device=args.device or None,
                temperature=0.0,
                retries=args.retries,
                retry_temperature=args.retry_temperature,
                prefix_len=args.prefix_len,
                generator=args.generator,
                fallback="persona",
                persona=persona,
                decode=args.decode,
            )
        elif team.kind == "heuristic":
            agents[team.id] = HeuristicAgent(
                int(team.meta.get("heuristic_seed") or team.seed),
                name=team.name,
            )
        else:
            agents[team.id] = CountingGreedy()
    return agents


def _build_knockout_if_ready(state: dict[str, Any]) -> None:
    if state.get("knockout_fixtures"):
        return
    results = state_results(state)
    group_fixtures = [Fixture(**item) for item in state.get("group_fixtures", [])]
    if len([key for key in results if key.startswith("g")]) < len(group_fixtures):
        return
    teams = state_teams(state)
    tables = {
        group: group_table(group, teams, group_fixtures, results)
        for group in GROUP_LABELS
    }
    state["tables"] = tables
    state["knockout_seed"] = {
        str(row["team"]): int(row["pos"]) + (0 if group == "A" else 10)
        for group, table in tables.items()
        for row in table
    }
    state["knockout_fixtures"] = [
        fixture.to_dict()
        for fixture in build_semifinal_fixtures(
            tables,
            state["maps"]["knockout"],
            int(state["seed"]),
        )
    ]


def _build_finals_if_ready(state: dict[str, Any]) -> None:
    results = state_results(state)
    if "final" in results or "third" in results:
        return
    if not all(key in results for key in ("sf1", "sf2")):
        return
    teams = state_teams(state)
    table_by_team: dict[str, int] = {}
    for group in GROUP_LABELS:
        table = state.get("tables", {}).get(group) or []
        for row in table:
            table_by_team[str(row["team"])] = int(row["pos"]) + (
                0 if group == "A" else 10
            )
    finals = build_final_fixtures(
        {key: results[key] for key in ("sf1", "sf2")},
        state["maps"]["knockout"],
        int(state["seed"]),
        team_seeds=table_by_team,
    )
    # Keep the semifinal fixtures already in the bracket; only add the
    # third-place match and final.  Replacing the whole list would make the
    # report lose the semifinal results.
    existing = [Fixture(**item) for item in state.get("knockout_fixtures", [])]
    semis = [fixture for fixture in existing if fixture.stage == "semifinal"]
    state["knockout_fixtures"] = [
        fixture.to_dict() for fixture in [*semis, *finals]
    ]


def run_tournament(state: dict[str, Any], args: argparse.Namespace, paths: dict[str, Path | None]) -> int:
    match_dir = Path(str(paths["match_dir"]))
    match_dir.mkdir(parents=True, exist_ok=True)
    agents = load_agents(state, args, paths)
    limit = int(args.limit or 0)
    played_this_run = 0

    while True:
        _build_knockout_if_ready(state)
        _build_finals_if_ready(state)
        fixtures = state_fixtures(state)
        results = state_results(state)
        pending = [fixture for fixture in fixtures if fixture.id not in results]
        if not pending:
            break
        if limit and len(results) >= limit:
            print(f"limit reached after {len(results)} games", flush=True)
            break

        fixture = pending[0]
        teams = state_teams(state)
        home = teams[fixture.home]
        away = teams[fixture.away]
        print(
            f"[{len(results) + 1:02d}] {fixture.stage} {fixture.group or ''} "
            f"{home.name} vs {away.name} on {fixture.map} ...",
            flush=True,
        )
        try:
            result = play_fixture(state, fixture, agents, match_dir=match_dir)
        except Exception as exc:  # noqa: BLE001 - keep the resume state and report the failure
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            write_state(state)
            write_reports(state)
            raise
        state["results"][result.fixture_id] = result.to_dict()
        played_this_run += 1
        write_state(state)
        write_reports(state)
        print(
            f"  -> {home.name} {result.score.get(fixture.home, 0):.1f} - "
            f"{result.score.get(fixture.away, 0):.1f} {away.name}  "
            f"winner {teams[result.winner].name if result.winner else 'draw'} "
            f"({result.decided_by})  {result.seconds:.0f}s",
            flush=True,
        )

    write_state(state)
    write_reports(state)
    print(
        f"\n{len(state['results'])} games complete; "
        f"report: {Path(str(state['out_dir'])) / 'report.md'}",
        flush=True,
    )
    return 0


def print_schedule(state: dict[str, Any]) -> None:
    teams = state_teams(state)
    print(f"TBS World Cup 2026 · seed {state['seed']}")
    print(f"Group maps: {', '.join(state['maps']['group'])}")
    print(f"Knockout maps: {', '.join(state['maps']['knockout'])}")
    for group, team_ids in state["groups"].items():
        print(f"\nGroup {group}:")
        for team_id in team_ids:
            team = teams[team_id]
            print(f"  {team.name:24s} {team.kind}")
    print("\nGroup fixtures:")
    for fixture in state_fixtures(state):
        if fixture.stage != "group":
            continue
        print(
            f"  {fixture.id:32s} {teams[fixture.home].name:24s} vs "
            f"{teams[fixture.away].name:24s} on {fixture.map}"
        )
    print(f"\nTotal matches: {state['format']['total_matches']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_paths(args)
    out_dir = Path(str(paths["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "tournament.json"
    batch_name = args.batch_name or f"world_cup_{args.seed}"
    state: dict[str, Any]

    if args.report_only:
        if not state_path.is_file():
            raise SystemExit(f"no resume state at {state_path}; run the tournament first")
        state = load_state(state_path)
        write_reports(state)
        print(f"reports rebuilt from {state_path}")
        return 0

    if state_path.is_file():
        if not args.resume:
            raise SystemExit(
                f"{state_path} already exists; use --resume to continue it, "
                "or remove it to start a new tournament"
            )
        state = load_state(state_path)
        if int(state.get("seed", -1)) != args.seed:
            raise SystemExit(
                f"state seed {state.get('seed')} does not match --seed {args.seed}"
            )
        print(f"resuming {state_path}: {len(state.get('results', {}))} games complete")
    else:
        if args.resume:
            print(f"no state at {state_path}; starting a new tournament")
        teams = build_teams(args)
        state = create_state(teams=teams, args=args, paths=paths, batch_name=batch_name)
        if not args.dry_run:
            write_state(state)
            write_reports(state)

    if args.dry_run:
        print_schedule(state)
        return 0

    if not Path(str(paths["ncm_root"])).is_dir():
        raise SystemExit(f"NCM checkout not found: {paths['ncm_root']}")
    return run_tournament(state, args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
