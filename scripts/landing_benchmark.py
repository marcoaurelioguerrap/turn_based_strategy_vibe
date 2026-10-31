"""Measure whether the scripted agents actually cross water and land.

The World Cup report measures legal orders and board score, which can look
healthy while every island game ends in a draw on the home island.  This script
runs the same scripted ``GreedyAgent`` on both sides of the amphibious maps and
reports the facts a landing policy must move:

* first landing turn,
* land units that ended on a different landmass,
* units that reached the enemy home island,
* load/unload and transport-use counts.

It is deliberately model-free, so it is the fast go/no-go check for a view,
teacher or rules change before spending GPU time on NCM.

    py scripts/landing_benchmark.py
    py scripts/landing_benchmark.py --maps archipelago --games 20 --max-turns 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `py scripts/landing_benchmark.py` from anywhere

from tbs.agents.greedy import GreedyAgent
from tbs.agents.turn import run_turn
from tbs.engine import rules
from tbs.engine.amphibious import landmass_index
from tbs.engine.data import DATA_ROOT, load_scenario, load_specs
from tbs.engine.state import new_game

DEFAULT_MAPS = ("archipelago", "archipelago_m")


@dataclass
class GameReport:
    map: str
    seed: int
    status: str
    winner: str | None
    turns: int
    first_landing_turn: int | None = None
    first_landing_side: str | None = None
    landed_units: list[dict[str, Any]] = field(default_factory=list)
    enemy_home_units: list[dict[str, Any]] = field(default_factory=list)
    ever_landed: int = 0
    ever_enemy_home: int = 0
    loads: int = 0
    unloads: int = 0
    builds: int = 0
    captures: int = 0
    moves: int = 0
    holds: int = 0

    @property
    def landed(self) -> int:
        return len(self.landed_units)

    @property
    def reached_enemy_home(self) -> int:
        return len(self.enemy_home_units)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _starting_islands(state) -> dict[str, int | None]:
    islands = landmass_index(state)
    return {
        unit.id: islands.of(unit.pos)
        for unit in state.units.values()
        if unit.pos is not None
    }


def _enemy_home_islands(state) -> dict[str, int | None]:
    islands = landmass_index(state)
    home: dict[str, int | None] = {}
    for unit in state.units.values():
        if unit.type_id != "hq" or unit.pos is None:
            continue
        component = islands.of(unit.pos)
        for player in state.players:
            if player != unit.owner:
                home[player] = component
    return home


def _landing_state(state, start: dict[str, int | None]) -> list[dict[str, Any]]:
    islands = landmass_index(state)
    rows: list[dict[str, Any]] = []
    for unit in state.units.values():
        if unit.pos is None:
            continue
        before = start.get(unit.id)
        after = islands.of(unit.pos)
        if before is None or after is None or before == after:
            continue
        rows.append({"unit": unit.id, "owner": unit.owner, "type": unit.type_id, "island": after})
    return sorted(rows, key=lambda row: (row["owner"], row["unit"]))


def play_one(map_name: str, seed: int, max_turns: int | None) -> GameReport:
    scenario = load_scenario(DATA_ROOT / "maps" / f"{map_name}.json")
    scenario["seed"] = seed
    if max_turns is not None:
        scenario["max_turns"] = max_turns
    state = new_game(scenario, load_specs())
    rules.start_game(state)
    start = _starting_islands(state)
    enemy_home = _enemy_home_islands(state)
    report = GameReport(map=map_name, seed=seed, status=state.status, winner=None, turns=0)
    agent = GreedyAgent()
    ever_landed: set[tuple[str, str]] = set()
    ever_enemy_home: set[tuple[str, str]] = set()

    while state.status == "ongoing" and state.turn_number <= int(state.config.get("max_turns", 200)):
        result = run_turn(state, agent, attempts=1)
        for order in result.orders:
            action = order.get("action")
            if action == "load":
                report.loads += 1
            elif action == "unload":
                report.unloads += 1
            elif action == "build":
                report.builds += 1
            elif action == "capture":
                report.captures += 1
            elif action == "move":
                report.moves += 1
            elif action == "hold":
                report.holds += 1
        landed = _landing_state(state, start)
        if landed and report.first_landing_turn is None:
            report.first_landing_turn = state.turn_number
            report.first_landing_side = landed[0]["owner"]
        for row in landed:
            key = (row["owner"], row["unit"])
            ever_landed.add(key)
            if row["island"] is not None and row["island"] == enemy_home.get(row["owner"]):
                ever_enemy_home.add(key)

    report.status = state.status
    report.winner = state.winner
    report.turns = state.turn_number
    report.landed_units = _landing_state(state, start)
    report.enemy_home_units = [
        row
        for row in report.landed_units
        if row["island"] is not None and row["island"] == enemy_home.get(row["owner"])
    ]
    report.ever_landed = len(ever_landed)
    report.ever_enemy_home = len(ever_enemy_home)
    return report


def _summary(reports: list[GameReport]) -> dict[str, Any]:
    landed = [report for report in reports if report.ever_landed]
    enemy_home = [report for report in reports if report.ever_enemy_home]
    first = [report.first_landing_turn for report in reports if report.first_landing_turn is not None]
    return {
        "games": len(reports),
        "landed_games": len(landed),
        "enemy_home_games": len(enemy_home),
        "first_landing_avg": round(statistics.mean(first), 2) if first else None,
        "first_landing_best": min(first) if first else None,
        "units_landed": sum(report.ever_landed for report in reports),
        "enemy_home_units": sum(report.ever_enemy_home for report in reports),
        "final_landed_units": sum(report.landed for report in reports),
        "final_enemy_home_units": sum(report.reached_enemy_home for report in reports),
        "loads": sum(report.loads for report in reports),
        "unloads": sum(report.unloads for report in reports),
        "wins": sum(1 for report in reports if report.winner),
        "draws": sum(1 for report in reports if report.status == "draw"),
    }


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Amphibious landing benchmark",
        "",
        "Scripted `GreedyAgent` on both sides. A landing means a land unit ended on a",
        "different landmass from where it started; an enemy-home landing means it",
        "reached the opponent's HQ island.",
        "",
        "| Map | Games | Games with landing | Games reaching enemy home | Avg first landing | Units landed (ever) | Enemy-home units (ever) | Loads | Unloads | Wins | Draws |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for map_name, summary in payload["maps"].items():
        lines.append(
            "| {map} | {games} | {landed_games} | {enemy_home_games} | {first_landing_avg} | "
            "{units_landed} | {enemy_home_units} | {loads} | {unloads} | {wins} | {draws} |".format(
                map=map_name,
                **summary,
            )
        )
    lines += ["", "## Games", ""]
    for report in payload["games"]:
        lines.append(
            f"- `{report['map']}` seed {report['seed']}: {report['status']}"
            f"{' winner ' + report['winner'] if report['winner'] else ''}, "
            f"turn {report['turns']}, first landing {report['first_landing_turn']}, "
            f"landed-ever {report['ever_landed']}, enemy-home-ever {report['ever_enemy_home']}, "
            f"final-landed {len(report['landed_units'])}, "
            f"loads {report['loads']}, unloads {report['unloads']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS), help="comma-separated map stems")
    parser.add_argument("--games", type=int, default=5, help="games per map")
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="scenario turn cap override; 0 uses the map's own max_turns",
    )
    parser.add_argument("--out", default="docs/landing_benchmark/report.md")
    args = parser.parse_args(argv)

    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps:
        parser.error("--maps must name at least one map")
    max_turns = args.max_turns or None
    reports: list[GameReport] = []
    for map_index, map_name in enumerate(maps):
        print(f"map {map_name}", flush=True)
        for game_index in range(args.games):
            seed = args.seed_base + map_index * 1000 + game_index
            report = play_one(map_name, seed, max_turns)
            reports.append(report)
            print(
                f"  seed {seed}: {report.status}"
                f"{' winner ' + report.winner if report.winner else ''}, "
                f"landed-ever {report.ever_landed}, enemy-home-ever {report.ever_enemy_home}, "
                f"first {report.first_landing_turn}, loads {report.loads}, unloads {report.unloads}",
                flush=True,
            )

    payload = {
        "maps": {
            map_name: _summary([report for report in reports if report.map == map_name])
            for map_name in maps
        },
        "games": [report.to_dict() for report in reports],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render(payload), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
