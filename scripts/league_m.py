"""Round-robin scripted-persona league on the large ``_m`` maps.

This is the fast map-pack league: it uses the five scripted persona teachers
from ``C:\\AIs\\NCM\\tbs_ncm\\personas.py``, not the trained NCM checkpoints.
The teachers are pure Python and produce one legal order per step, so the whole
round-robin runs in a few minutes instead of tying up the model server.

    py scripts/league_m.py --max-turns 40 --max-steps 12

The report is written to ``docs/league_m/report.json`` and ``report.md``.  Use
``--ncm-root`` when the NCM checkout is somewhere other than the default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

GAME_REPO = Path(__file__).resolve().parents[1]
if str(GAME_REPO) not in sys.path:
    sys.path.insert(0, str(GAME_REPO))

from tbs.agents.turn import run_turn_stepwise
from tbs.engine import rules
from tbs.engine.data import DATA_ROOT, load_scenario, load_specs
from tbs.engine.state import new_game

DEFAULT_MAPS = (
    "border_skirmish_m",
    "mountain_pass_m",
    "archipelago_m",
    "urban_delta_m",
    "last_stand_m",
)
DEFAULT_PERSONAS = ("aggressive", "defensive", "infantry", "armor", "artillery")
DEFAULT_NCM_ROOT = Path(
    os.environ.get("TBS_NCM_ROOT", str(GAME_REPO.parent / "NCM_fork"))
)
DEFAULT_OUT = GAME_REPO / "docs" / "league_m"


def _load_personas(ncm_root: Path):
    if str(ncm_root) not in sys.path:
        sys.path.insert(0, str(ncm_root))
    try:
        from tbs_ncm.personas import PersonaTeacher
    except ImportError as exc:
        raise SystemExit(
            f"cannot import tbs_ncm.personas from {ncm_root}: {exc}\n"
            "point --ncm-root at the NCM checkout"
        ) from None
    return PersonaTeacher


def _site_count(state, player: str) -> int:
    total = 0
    for tile in state.owned_tiles(player):
        terrain = state.terrain_of(tile)
        if terrain.capture_points is not None or terrain.build:
            total += 1
    return total


def board_score(state, player: str) -> float:
    """A cap-safe score: sites, live units and remaining HP."""
    units = state.units_of(player)
    return 2.0 * _site_count(state, player) + len(units) + sum(unit.hp for unit in units) / 100.0


def play_game(
    scenario: dict[str, Any],
    specs,
    first: str,
    second: str,
    PersonaTeacher,
    *,
    seed: int,
    max_turns: int,
    max_steps: int,
) -> dict[str, Any]:
    state = new_game(scenario, specs)
    rules.start_game(state)
    side = "A" if seed % 2 == 0 else "B"
    other = "A" if side == "B" else "B"
    agents = {
        side: PersonaTeacher(first, seed=seed),
        other: PersonaTeacher(second, seed=seed + 1),
    }
    started = time.perf_counter()
    while state.status == "ongoing" and state.turn_number <= max_turns:
        result = run_turn_stepwise(
            state,
            agents[state.current_player],
            max_steps=max_steps,
        )
        if not result.applied and not result.ended_turn:
            break
    elapsed = time.perf_counter() - started
    scores = {player: board_score(state, player) for player in state.player_ids()}
    return {
        "pair": [first, second],
        "map": scenario["name"],
        "seed": seed,
        "sides": {first: side, second: other},
        "winner": state.winner or "",
        "status": state.status,
        "turns": state.turn_number,
        "seconds": round(elapsed, 2),
        "scores": scores,
        "score_delta": round(scores[side] - scores[other], 2),
        "units": {player: len(state.units_of(player)) for player in state.player_ids()},
    }


def aggregate(matches: list[dict], persona: str) -> dict[str, Any]:
    wins = losses = unfinished = 0
    score = 0.0
    turns = 0.0
    for match in matches:
        if persona not in match["sides"]:
            continue
        side = match["sides"][persona]
        if match["winner"] == side:
            wins += 1
        elif match["winner"]:
            losses += 1
        else:
            unfinished += 1
        score += match["score_delta"] if side == match["sides"][match["pair"][0]] else -match["score_delta"]
        turns += match["turns"]
    games = wins + losses + unfinished
    return {
        "persona": persona,
        "games": games,
        "wins": wins,
        "losses": losses,
        "unfinished": unfinished,
        "win_rate": wins / games if games else 0.0,
        "avg_score_delta": score / games if games else 0.0,
        "avg_turns": turns / games if games else 0.0,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Scripted persona league - large `_m` maps",
        "",
        f"{report['matches']} games on {len(report['maps'])} maps, "
        f"max turns `{report['max_turns']}`, max steps per turn `{report['max_steps']}`.",
        "",
        "The five scripted persona teachers from the NCM checkout (no model weights) "
        "play a round-robin. This is the fast balance league; the NCM model league "
        "command is in the main README.",
        "",
        "| persona | W-L-U | win% | score | avg turns |",
        "|---|---|---|---|---|",
    ]
    for persona in report["personas"]:
        row = report["aggregates"][persona]
        lines.append(
            f"| {persona} | {row['wins']}-{row['losses']}-{row['unfinished']} | "
            f"{row['win_rate']:.0%} | {row['avg_score_delta']:+.1f} | {row['avg_turns']:.1f} |"
        )
    lines += ["", "## Results by map", "", "| map | games | finished | avg turns |", "|---|---|---|---|"]
    for map_name in report["maps"]:
        rows = [match for match in report["games"] if match["map"] == map_name]
        finished = sum(1 for match in rows if match["winner"])
        lines.append(
            f"| {map_name} | {len(rows)} | {finished} | "
            f"{sum(match['turns'] for match in rows) / max(1, len(rows)):.1f} |"
        )
    lines += ["", "## Pairwise (row vs column, W-L-U)", "", "| | " + " | ".join(report["personas"]) + " |"]
    lines.append("|---|" + "|".join("---" for _ in report["personas"]) + "|")
    for persona in report["personas"]:
        cells = []
        for other in report["personas"]:
            if persona == other:
                cells.append("-")
                continue
            rows = [
                match
                for match in report["games"]
                if set(match["pair"]) == {persona, other}
            ]
            wins = sum(1 for match in rows if match["winner"] == match["sides"][persona])
            losses = sum(1 for match in rows if match["winner"] and match["winner"] != match["sides"][persona])
            unfinished = sum(1 for match in rows if not match["winner"])
            cell = f"{wins}-{losses}"
            if unfinished:
                cell += f" ({unfinished}U)"
            cells.append(cell)
        lines.append(f"| **{persona}** | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scripted persona round-robin on the _m maps")
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS))
    parser.add_argument("--personas", default=",".join(DEFAULT_PERSONAS))
    parser.add_argument("--games-per-map", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=880000)
    parser.add_argument("--ncm-root", default=str(DEFAULT_NCM_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    PersonaTeacher = _load_personas(Path(args.ncm_root))
    personas = [name.strip() for name in args.personas.split(",") if name.strip()]
    wanted = {name.strip() for name in args.maps.split(",") if name.strip()}
    maps = [DATA_ROOT / "maps" / f"{name}.json" for name in DEFAULT_MAPS if name in wanted]
    missing = wanted - {path.stem for path in maps}
    if missing:
        parser.error(f"unknown maps: {', '.join(sorted(missing))}")
    if len(personas) < 2:
        parser.error("a league needs at least two personas")

    specs = load_specs()
    pairs = list(itertools.combinations(personas, 2))
    matches: list[dict] = []
    started = time.perf_counter()
    game_index = 0
    for pair in pairs:
        for map_path in maps:
            for game in range(args.games_per_map):
                scenario = load_scenario(map_path)
                seed = args.seed_base + game_index
                game_index += 1
                match = play_game(
                    scenario,
                    specs,
                    pair[0],
                    pair[1],
                    PersonaTeacher,
                    seed=seed,
                    max_turns=args.max_turns,
                    max_steps=args.max_steps,
                )
                matches.append(match)
                print(
                    f"  {pair[0]} vs {pair[1]} {match['map']}: "
                    f"{match['winner'] or 'unfinished'}  turns {match['turns']}  "
                    f"score {match['score_delta']:+.1f}  {match['seconds']:.1f}s",
                    flush=True,
                )

    report = {
        "maps": [path.stem for path in maps],
        "personas": personas,
        "games_per_map": args.games_per_map,
        "max_turns": args.max_turns,
        "max_steps": args.max_steps,
        "matches": len(matches),
        "seconds": round(time.perf_counter() - started, 1),
        "aggregates": {persona: aggregate(matches, persona) for persona in personas},
        "games": matches,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = render(report)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    print("\n" + "\n".join(text.splitlines()[: 10 + len(personas)]))
    print(f"\nreport: {out_dir / 'report.json'}\n        {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
