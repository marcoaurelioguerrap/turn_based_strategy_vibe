"""Generate NCM-style (state, order) rows from ferry-capable scripted games.

The original NCM dataset was generated on ``border_skirmish`` with a teacher
that never loaded or unloaded, so the released model has no amphibious action
examples.  This script plays the scripted ``GreedyAgent`` (which now executes
the engine's ferry hints) on the island maps and writes the same compact
``prefix`` / ``target`` JSONL rows that ``tbs_ncm.train_generator`` consumes.

    py scripts/generate_ferry_data.py --games 20
    py scripts/generate_ferry_data.py --maps archipelago,archipelago_m --games 50 \
        --amphibious-repeat 4 --out tbs_ncm/data/ferry.jsonl

The output is intentionally dependency-free; run it with either the game
repository's Python or the NCM virtualenv.  Rows whose target is a load or
unload carry ``"amphibious": true`` so a trainer can oversample them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tbs.agents.codec import encode_order, encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.protocol import AgentPlan
from tbs.agents.turn import apply_plan, build_request
from tbs.engine import rules
from tbs.engine.data import DATA_ROOT, available_scenarios, load_scenario, load_specs
from tbs.engine.state import new_game

DEFAULT_MAPS = ("archipelago", "archipelago_m")


def units_can_act(state, player: str) -> bool:
    return any(
        not unit.acted and not unit.loaded and not unit.deployed_this_turn
        for unit in state.units_of(player)
    )


def scenario_paths(maps: str) -> list[Path]:
    available = [Path(path) for path in available_scenarios()]
    if maps.strip().lower() in ("", "all", "*"):
        return available
    wanted: list[Path] = []
    for token in maps.split(","):
        name = token.strip()
        if not name:
            continue
        match = next((path for path in available if path.stem == name), None)
        if match is None:
            raise SystemExit(
                f"unknown map {name!r}; available: "
                + ", ".join(path.stem for path in available)
            )
        wanted.append(match)
    return wanted


def play_game(
    scenario_path: Path,
    seed: int,
    *,
    max_turns: int,
    max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario = load_scenario(scenario_path)
    scenario["seed"] = seed
    state = new_game(scenario, load_specs())
    rules.start_game(state)
    agent = GreedyAgent()
    rows: list[dict[str, Any]] = []
    rejected = 0
    steps = 0

    while state.status == "ongoing" and state.turn_number <= max_turns:
        player = state.current_player
        for _ in range(max_steps):
            if state.status != "ongoing" or state.current_player != player:
                break
            if not units_can_act(state, player):
                break
            request = build_request(state, player, history=0)
            plan = agent.plan(request)
            if not plan.orders:
                break
            order = dict(plan.orders[0])
            prefix = encode_state(request.view)
            target = encode_order(order)
            result = apply_plan(state, AgentPlan(orders=[order]))
            if not result.applied:
                rejected += 1
                break
            ferry = request.view.get("ferry") or {}
            operations = [
                item for item in (ferry.get("operations") or ()) if isinstance(item, dict)
            ]
            unit = str(order.get("unit") or "")
            operation = next(
                (
                    item
                    for item in operations
                    if str(item.get("transport") or "") == unit
                    or str(item.get("cargo") or "") == unit
                ),
                None,
            )
            amphibious = order.get("action") in ("load", "unload") or operation is not None
            rows.append(
                {
                    "prefix": prefix,
                    "target": target,
                    "seed": seed,
                    "scenario": scenario_path.stem,
                    "turn": request.view["turn"],
                    "player": player,
                    "amphibious": amphibious,
                    "ferry_action": (
                        str(operation.get("action")) if operation is not None else ""
                    ),
                }
            )
            steps += 1
        if state.status != "ongoing":
            break
        if state.current_player == player:
            rules.perform(state, {"action": "end_turn"})

    for row in rows:
        row["winner"] = state.winner or ""
        row["status"] = state.status
        row["learner_won"] = bool(state.winner and state.winner == row["player"])

    stats = {
        "scenario": scenario_path.stem,
        "seed": seed,
        "status": state.status,
        "winner": state.winner or "",
        "turns": state.turn_number,
        "steps": steps,
        "rows": len(rows),
        "rejected": rejected,
        "actions": dict(Counter(row["target"].split()[0] for row in rows)),
    }
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS))
    parser.add_argument("--games", type=int, default=10, help="games per map")
    parser.add_argument("--seed-base", type=int, default=700000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument(
        "--amphibious-repeat",
        type=int,
        default=1,
        help="write load/unload rows this many times to oversample the rare action",
    )
    parser.add_argument("--out", default="saves/ferry_data.jsonl")
    args = parser.parse_args(argv)

    maps = scenario_paths(args.maps)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for map_index, scenario_path in enumerate(maps):
        for game_index in range(args.games):
            seed = args.seed_base + map_index * 1000 + game_index
            rows, stats = play_game(
                scenario_path,
                seed,
                max_turns=args.max_turns,
                max_steps=args.max_steps,
            )
            all_rows.extend(rows)
            games.append(stats)
            print(
                f"{scenario_path.stem} seed {seed}: {stats['rows']} rows, "
                f"{stats['steps']} steps, {stats['status']}"
                f"{' winner ' + stats['winner'] if stats['winner'] else ''}, "
                f"rejected {stats['rejected']}",
                flush=True,
            )

    repeat = max(1, int(args.amphibious_repeat))
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            copies = repeat if row["amphibious"] else 1
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for _ in range(copies):
                handle.write(line)
                written += 1
    summary = {
        "maps": [path.stem for path in maps],
        "games": games,
        "rows": written,
        "raw_rows": len(all_rows),
        "amphibious_rows": sum(1 for row in all_rows if row["amphibious"]),
        "amphibious_repeat": repeat,
        "actions": dict(Counter(row["target"].split()[0] for row in all_rows)),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} and {summary_path} ({summary['rows']} rows, {summary['amphibious_rows']} amphibious)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
