"""Headless match: an LLM against a scripted agent, or two LLMs.

This is the fastest way to validate a real local model without opening the
window.  Every accepted order is recorded, then replayed from scratch to prove
the game is reproducible even though the LLM's choices are not.

    py -m tbs.agents.match --a greedy --b llm \
        --llm-model "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF:UD-IQ2_M" \
        --out saves/llm_match.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tbs.agents.council import CouncilAgent
from tbs.agents.generals import DEFAULT_GENERALS, load_generals, select_generals
from tbs.agents.greedy import GreedyAgent
from tbs.agents.llm import (
    DEFAULT_BASE_URL,
    JsonlLogger,
    LlmClient,
    LlmConfig,
    LlmTransportError,
    SingleLlmAgent,
)
from tbs.agents.turn import run_turn
from tbs.engine import rules
from tbs.engine.data import default_scenario, load_scenario, load_specs
from tbs.engine.serialize import replay, save_game, state_hash
from tbs.engine.state import new_game

__all__ = ["main"]


def _build_agent(
    kind: str,
    *,
    config: LlmConfig | None,
    logger: JsonlLogger | None,
    history: int,
    council_members: list[str] | None = None,
    council_use_commander: bool = True,
    council_parallel: bool = False,
    council_config: str | None = None,
):
    if kind == "greedy":
        return GreedyAgent()
    if kind in ("llm", "council"):
        if config is None:
            raise SystemExit(f'side "{kind}" needs --llm-model (or TBS_LLM_MODEL)')
    if kind == "llm":
        return SingleLlmAgent(LlmClient(config), logger=logger, history=history)
    if kind == "council":
        generals = load_generals(council_config) if council_config else DEFAULT_GENERALS
        try:
            generals = select_generals(generals, council_members)
            return CouncilAgent(
                LlmClient(config),
                generals=generals,
                logger=logger,
                history=history,
                use_commander=council_use_commander,
                parallel=council_parallel,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    raise SystemExit(f"unknown agent kind {kind!r}; use 'greedy', 'llm' or 'council'")


def _write_orders(path: Path, orders: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(orders, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbs.agents.match",
        description="headless LLM-vs-agent match with a replayable order log",
    )
    parser.add_argument("--a", choices=("greedy", "llm", "council"), default="greedy")
    parser.add_argument("--b", choices=("greedy", "llm", "council"), default="llm")
    parser.add_argument("--scenario", help="scenario JSON path (default: data/maps/border_skirmish.json)")
    parser.add_argument("--seed", type=int, help="override the scenario seed")
    parser.add_argument("--max-turns", type=int, default=60, help="stop after this many full turns")
    parser.add_argument("--history", type=int, default=8, help="recent log lines sent to the LLM")
    parser.add_argument("--llm-base-url", default=os.environ.get("TBS_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--llm-model", default=os.environ.get("TBS_LLM_MODEL"))
    parser.add_argument("--llm-api-key", default=os.environ.get("TBS_LLM_API_KEY"))
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1500)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument("--llm-retries", type=int, default=2, help="transport retries")
    parser.add_argument("--llm-attempts", type=int, default=2, help="whole-turn attempts after rejections")
    parser.add_argument("--llm-json-mode", action="store_true")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--llm-no-thinking",
        action="store_true",
        help="disable model reasoning (recommended for the 8k-context Nemotron server)",
    )
    thinking.add_argument("--llm-thinking", action="store_true", help="force model reasoning on")
    parser.add_argument(
        "--llm-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "max", "xhigh"),
        help="server-side reasoning effort, when supported",
    )
    parser.add_argument("--llm-log", help="append prompt/response JSONL here")
    parser.add_argument(
        "--council-members",
        help="comma-separated general ids for a council side (default: all)",
    )
    parser.add_argument(
        "--council-config",
        help="JSON roster override for a council side",
    )
    parser.add_argument(
        "--council-no-commander",
        action="store_true",
        help="skip the supreme commander's pre-brief (one fewer LLM call per turn)",
    )
    parser.add_argument(
        "--council-parallel",
        action="store_true",
        help="ask generals concurrently (the local server may still serialize them)",
    )
    parser.add_argument("--out", default="saves/llm_match.json", help="final save file")
    parser.add_argument("--orders", help="order-log path (default: <out>.orders.json)")
    args = parser.parse_args(argv)

    scenario_path = Path(args.scenario) if args.scenario else default_scenario()
    scenario = load_scenario(scenario_path)
    if args.seed is not None:
        scenario["seed"] = args.seed
    state = new_game(scenario, load_specs())
    rules.start_game(state)

    config = None
    if any(kind in ("llm", "council") for kind in (args.a, args.b)):
        if not args.llm_model:
            parser.error('one side is "llm"/"council": pass --llm-model or set TBS_LLM_MODEL')
        config = LlmConfig(
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
            retries=args.llm_retries,
            json_mode=args.llm_json_mode,
            enable_thinking=True if args.llm_thinking else False,
            reasoning_effort=args.llm_reasoning_effort,
        )

    logger = JsonlLogger(args.llm_log) if args.llm_log else None
    council_members = (
        [item.strip() for item in args.council_members.split(",") if item.strip()]
        if args.council_members
        else None
    )
    agents = {
        "A": _build_agent(
            args.a,
            config=config,
            logger=logger,
            history=args.history,
            council_members=council_members,
            council_use_commander=not args.council_no_commander,
            council_parallel=args.council_parallel,
            council_config=args.council_config,
        ),
        "B": _build_agent(
            args.b,
            config=config,
            logger=logger,
            history=args.history,
            council_members=council_members,
            council_use_commander=not args.council_no_commander,
            council_parallel=args.council_parallel,
            council_config=args.council_config,
        ),
    }
    for player_id in agents:
        if player_id not in state.players:
            parser.error(f"the scenario has no player {player_id!r}")

    included_turns = 0
    orders: list[dict[str, Any]] = []
    print(
        f"scenario {state.scenario_name!r} seed {state.seed}  A={args.a}  B={args.b}  "
        f"model={args.llm_model or '-'}"
    )
    while state.status == "ongoing" and state.turn_number <= args.max_turns:
        player = state.current_player
        agent = agents[player]
        try:
            result = run_turn(
                state,
                agent,
                attempts=args.llm_attempts,
                history=args.history,
                logger=logger,
            )
        except LlmTransportError as exc:
            print(f"\nLLM FAILURE on turn {state.turn_number}, player {player}: {exc}", file=sys.stderr)
            _write_orders(_orders_path(args), orders)
            save_game(state, _ensure_parent(Path(args.out)))
            return 2
        orders.extend(result.orders)
        included_turns += 1
        reasons = f"  rejected={len(result.errors)}" if result.errors else ""
        if result.ignored:
            reasons += f"  ignored={len(result.ignored)}"
        if isinstance(agent, CouncilAgent):
            stats = agent.last_stats
            reasons += (
                f"  council={len(stats.get('generals', []))}"
                f" calls={stats.get('calls', 0)}"
                f" dropped={len(stats.get('dropped', []))}"
            )
        print(
            f"turn {state.turn_number:>3} {player}  orders={len(result.orders):>2}  "
            f"attempts={result.attempts}{reasons}  "
            f"units A={len(state.units_of('A'))} B={len(state.units_of('B'))}  "
            f"funds A={state.players['A'].funds} B={state.players['B'].funds}"
        )

    out_path = _ensure_parent(Path(args.out))
    orders_path = _ensure_parent(_orders_path(args))
    save_game(state, out_path)
    _write_orders(orders_path, orders)

    print(f"\nstatus {state.status}" + (f" winner {state.winner}" if state.winner else ""))
    print(f"save   {out_path}")
    print(f"orders {orders_path} ({len(orders)} orders, {included_turns} player-turns)")
    rebuilt = replay(scenario, orders)
    matched = state_hash(rebuilt) == state_hash(state)
    print(f"replay {'MATCHES' if matched else 'DIFFERS'} state hash {state_hash(state)}")
    if state.status == "ongoing":
        print(f"stopped at the {args.max_turns}-turn cap while still ongoing")
        return 1
    return 0 if matched else 3


def _orders_path(args: argparse.Namespace) -> Path:
    if args.orders:
        return Path(args.orders)
    out = Path(args.out)
    return out.with_name(out.stem + ".orders.json")


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    sys.exit(main())
