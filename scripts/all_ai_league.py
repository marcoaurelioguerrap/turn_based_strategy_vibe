"""Round-robin league: LLM, NCM and conventional AIs on every bundled map.

The four representatives are:

* ``llm``    - the local OpenAI-compatible model (Qwen by default),
* ``ncm-vanguard`` - the trained NCM Vanguard persona (armour checkpoint),
* ``ncm-siege``    - the trained NCM Siege persona (artillery checkpoint),
* ``conventional`` - an aggressive seeded conventional scripted AI.

Every unordered pair plays once on each map.  Sides alternate across fixtures
so no representative always gets the first move.  Each map uses its authored
turn limit unless ``--max-turns`` overrides it.  Capped games are awarded on
board score, with exact ties drawn.  Games are saved in the NCM dashboard's
match directory, with the usual ``.orders.json`` and ``.meta.json`` sidecars.
The league report is written to ``docs/all_ai_league/<batch>/report.json`` and
``report.md``.

Run it with the NCM virtualenv so torch and the NCM checkpoints are available:

    $env:TBS_LLM_API_KEY = "<the key Unsloth Desktop shows>"
    C:\\AIs\\NCM\\.venv\\Scripts\\python.exe scripts\\all_ai_league.py --dry-run
    C:\\AIs\\NCM\\.venv\\Scripts\\python.exe scripts\\all_ai_league.py --resume

``--dry-run`` lists the fixtures without loading a model.  ``--report-only``
rebuilds the report from the saved match sidecars.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GAME_REPO = Path(__file__).resolve().parents[1]
if str(GAME_REPO) not in sys.path:
    sys.path.insert(0, str(GAME_REPO))

from tbs.agents.greedy import GreedyAgent
from tbs.agents.llm import (
    JsonlLogger,
    LlmClient,
    LlmConfig,
    LlmTransportError,
    SingleLlmAgent,
)
from tbs.agents.turn import run_turn, run_turn_stepwise
from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.serialize import replay, save_game, state_hash
from tbs.engine.state import new_game

DEFAULT_NCM_ROOT = Path(
    os.environ.get("TBS_NCM_ROOT", str(GAME_REPO.parent / "NCM_fork"))
)
DEFAULT_LLM_MODEL = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8888/v1"
DEFAULT_BATCH = "all_ai_league_20261030_v2"

TEAM_LABELS = {
    "llm": "Qwen LLM",
    "ncm-vanguard": "NCM Vanguard",
    "ncm-siege": "NCM Siege",
    "conventional": "Conventional heuristic",
}
TEAM_ORDER = ("llm", "ncm-vanguard", "ncm-siege", "conventional")
NCM_PERSONAS = {
    "ncm-vanguard": "armor",
    "ncm-siege": "artillery",
}
NCM_KINDS = set(NCM_PERSONAS)
STEPWISE_KINDS = NCM_KINDS | {"conventional"}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _time_text_now() -> str:
    """Compact local timestamp for the dashboard's narrow Games table."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")


def _check_monitor(url: str) -> None:
    """Fail before a long run when the NCM dashboard is not watching the Games tab."""
    endpoint = url.rstrip("/") + "/api/games"
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            if response.status != 200:
                raise OSError(f"HTTP {response.status}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SystemExit(
            f"monitor dashboard is not reachable at {endpoint}: {exc}\n"
            "start it with: C:\\AIs\\NCM\\.venv\\Scripts\\python.exe -m tbs_ncm.dashboard "
            "--host 127.0.0.1 --port 8765"
        ) from None
    print(f"monitor: {endpoint} reachable", flush=True)


def _board_stats(state, player: str) -> dict[str, Any]:
    """The cap-safe board score used by the NCM dashboard."""
    units = state.units_of(player)
    sites = state.owned_tiles(player)
    hp = sum(unit.hp for unit in units)
    board = {
        "units": len(units),
        "hp": hp,
        "sites": len(sites),
        "funds": state.players[player].funds,
        "hq_hp": sum(unit.hp for unit in units if unit.type_id == "hq"),
    }
    board["score"] = round(2 * board["sites"] + board["units"] + hp / 100.0, 2)
    return board


def _turn_cap(args: argparse.Namespace, scenario: dict[str, Any]) -> int:
    """The requested cap, or the map's authored limit when the CLI says 0."""
    requested = int(getattr(args, "max_turns", 0) or 0)
    return requested if requested > 0 else int(scenario.get("max_turns", 200))


def _board_winner(board: dict[str, Any], margin: float = 0.0) -> str | None:
    """Return ('A'|'B') when one side's score is clearly higher, else None."""
    a = float((board.get("A") or {}).get("score", 0.0))
    b = float((board.get("B") or {}).get("score", 0.0))
    if abs(b - a) <= float(margin):
        return None
    return "A" if a > b else "B"


def _ncm_stats(agent) -> dict[str, Any]:
    """Counters copied from ``tbs_ncm.play`` so the dashboard reads them."""
    steps = max(1, agent.steps or agent.orders)
    return {
        "steps": agent.steps,
        "samples": agent.samples,
        "generated": agent.generated,
        "repaired": agent.repaired,
        "constrained": agent.constrained,
        "tactical": agent.tactical,
        "fallbacks": agent.fallbacks,
        "illegal": agent.illegal,
        "rejected": agent.rejected,
        "recovered": agent.recovered,
        "state_too_long": agent.state_too_long,
        "generation_errors": agent.generation_errors,
        "repairs": dict(agent.repairs),
        "model_orders": agent.model_orders,
        "fallback_rate": agent.fallbacks / steps,
        "constrained_share": agent.constrained / steps,
        "tactical_share": agent.tactical / steps,
        "legal_share": (agent.generated + agent.repaired + agent.constrained) / steps,
        "model_share": agent.model_orders / steps,
        "fallback_reasons": dict(Counter(agent.fallback_reasons)),
    }


def _delta(before: dict[str, Any], after: dict[str, Any], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


@dataclass(frozen=True)
class Fixture:
    id: str
    map_name: str
    map_path: str
    agent_a: str
    agent_b: str
    seed: int

    @property
    def label(self) -> str:
        return (
            f"{TEAM_LABELS[self.agent_a]} vs {TEAM_LABELS[self.agent_b]} "
            f"- {self.map_name}"
        )

    @property
    def filename(self) -> str:
        return f"{self.id}.json"


# ---------------------------------------------------------------------------
# roster and fixtures
# ---------------------------------------------------------------------------


def _resolve_maps(value: str) -> list[Path]:
    paths = [Path(path) for path in available_scenarios()]
    if value.strip().lower() in ("", "all", "*"):
        return paths
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    known = {path.stem: path for path in paths}
    missing = sorted(wanted - set(known))
    if missing:
        raise SystemExit(f"unknown map(s): {', '.join(missing)}")
    return [known[name] for name in sorted(wanted)]


def build_fixtures(
    agents: list[str],
    maps: list[Path],
    *,
    seed_base: int,
    batch: str = "",
) -> list[Fixture]:
    """One game for every unordered pair on every map, alternating sides."""
    fixtures: list[Fixture] = []
    prefix = f"{batch}-" if batch else ""
    pair_index = 0
    for map_index, map_path in enumerate(maps):
        for left, right in itertools.combinations(agents, 2):
            if (map_index + pair_index) % 2:
                left, right = right, left
            seed = seed_base + map_index * 100 + pair_index
            fixture_id = f"{prefix}all-ai-{map_path.stem}-{left}-vs-{right}-seed{seed}"
            fixtures.append(
                Fixture(
                    id=fixture_id,
                    map_name=map_path.stem,
                    map_path=str(map_path),
                    agent_a=left,
                    agent_b=right,
                    seed=seed,
                )
            )
            pair_index += 1
    return fixtures


# ---------------------------------------------------------------------------
# agent construction
# ---------------------------------------------------------------------------


class AgentFactory:
    """Build fresh agents per game; NCM samplers are cached per persona/prefix."""

    def __init__(self, args: argparse.Namespace, logger: JsonlLogger | None):
        self.args = args
        self.logger = logger
        self.kinds = set(getattr(args, "agent_list", []))
        self._samplers: dict[tuple[str, int], Any] = {}
        self._personas: dict[str, Any] = {}
        if self.kinds & NCM_KINDS or "conventional" in self.kinds:
            if str(args.ncm_root) not in sys.path:
                sys.path.insert(0, str(args.ncm_root))
            try:
                from tbs_ncm.ncm_agent import NcmAgent, NcmSampler
                from tbs_ncm.personas import PersonaTeacher
            except ImportError as exc:
                raise SystemExit(
                    f"cannot import the NCM player from {args.ncm_root}: {exc}\n"
                    "run this script with the NCM virtualenv or pass --ncm-root"
                ) from None
            self._ncm_agent_class = NcmAgent
            self._ncm_sampler_class = NcmSampler
            self._persona_class = PersonaTeacher
            if "conventional" in self.kinds:
                from tbs_ncm.heuristic import HeuristicAgent

                self._heuristic_class = HeuristicAgent

    def prefix_for(self, map_name: str) -> int:
        if self.args.prefix_len:
            return int(self.args.prefix_len)
        if map_name.endswith("_m"):
            return int(self.args.prefix_len_large)
        return int(self.args.prefix_len_small)

    def sampler(self, persona: str, prefix_len: int):
        key = (persona, prefix_len)
        if key not in self._samplers:
            checkpoint = Path(self.args.ncm_persona_dir) / persona
            self._samplers[key] = self._ncm_sampler_class(
                checkpoint,
                device=self.args.ncm_device,
                prefix_len=prefix_len,
                generator=self.args.ncm_generator,
            )
        return self._samplers[key]

    def persona(self, persona_id: str):
        if persona_id not in self._personas:
            self._personas[persona_id] = self._persona_class(persona_id)
        return self._personas[persona_id]

    def build(self, kind: str, *, fixture_id: str, map_name: str):
        if kind == "greedy":  # legacy roster id, kept for older commands
            return GreedyAgent()
        if kind == "llm":
            if not self.args.llm_model:
                raise SystemExit("the llm representative needs --llm-model or TBS_LLM_MODEL")
            config = LlmConfig(
                model=self.args.llm_model,
                base_url=self.args.llm_base_url,
                api_key=self.args.llm_api_key,
                temperature=self.args.llm_temperature,
                max_tokens=self.args.llm_max_tokens,
                timeout=self.args.llm_timeout,
                retries=self.args.llm_retries,
                enable_thinking=False,
            )
            return SingleLlmAgent(
                LlmClient(config),
                logger=self.logger,
                history=self.args.history,
                name=f"llm-{fixture_id}",
                auto_tactics=self.args.llm_auto_tactics,
            )
        if kind == "conventional":
            return self._heuristic_class(self.args.heuristic_seed)
        if kind in NCM_PERSONAS:
            persona_id = NCM_PERSONAS[kind]
            fallback = (
                self.persona(persona_id) if self.args.ncm_fallback == "persona" else None
            )
            return self._ncm_agent_class(
                self.sampler(persona_id, self.prefix_for(map_name)),
                temperature=self.args.ncm_temperature,
                retries=self.args.ncm_retries,
                retry_temperature=self.args.ncm_retry_temperature,
                decode=self.args.ncm_decode,
                fallback=fallback,
            )
        raise SystemExit(f"unknown representative {kind!r}")


# ---------------------------------------------------------------------------
# one game
# ---------------------------------------------------------------------------


def play_game(
    fixture: Fixture,
    *,
    factory: AgentFactory,
    specs,
    args: argparse.Namespace,
    logger: JsonlLogger | None,
) -> dict[str, Any]:
    scenario = load_scenario(Path(fixture.map_path))
    scenario["seed"] = fixture.seed
    turn_cap = _turn_cap(args, scenario)
    scenario["max_turns"] = turn_cap
    state = new_game(scenario, specs)
    rules.start_game(state)

    agents = {
        "A": factory.build(fixture.agent_a, fixture_id=fixture.id, map_name=fixture.map_name),
        "B": factory.build(fixture.agent_b, fixture_id=fixture.id, map_name=fixture.map_name),
    }
    kinds = {"A": fixture.agent_a, "B": fixture.agent_b}
    ncm_sides = [side for side, kind in kinds.items() if kind in NCM_KINDS]
    ncm_side = "both" if len(ncm_sides) > 1 else (ncm_sides[0] if ncm_sides else None)
    ncm = agents[ncm_sides[0]] if ncm_sides else None

    match_path = Path(args.match_dir) / fixture.filename
    orders_path = match_path.with_name(match_path.stem + ".orders.json")
    meta_path = match_path.with_name(match_path.stem + ".meta.json")
    started = _time_text_now()
    started_clock = time.time()
    orders: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    error: str | None = None

    print(
        f"\n[{fixture.id}] {fixture.label} seed={fixture.seed} "
        f"A={fixture.agent_a} B={fixture.agent_b}",
        flush=True,
    )

    try:
        while state.status == "ongoing" and state.turn_number <= turn_cap:
            player = state.current_player
            agent = agents[player]
            kind = kinds[player]
            before = _ncm_stats(agent) if kind in NCM_KINDS else None
            if kind in STEPWISE_KINDS:
                result = run_turn_stepwise(
                    state,
                    agent,
                    max_steps=(
                        args.conventional_max_steps
                        if kind == "conventional"
                        else args.max_steps
                    ),
                    logger=logger,
                )
            else:
                result = run_turn(
                    state,
                    agent,
                    attempts=args.llm_attempts,
                    history=args.history,
                    logger=logger,
                )
            orders.extend(result.orders)
            after = _ncm_stats(agent) if kind in NCM_KINDS else None
            entry: dict[str, Any] = {
                "turn": state.turn_number,
                "player": player,
                "orders": len(result.orders),
                "attempts": result.attempts,
                "board": {"A": _board_stats(state, "A"), "B": _board_stats(state, "B")},
                "ncm": None,
            }
            if before is not None and after is not None:
                entry["ncm"] = {
                    "agent": kind,
                    **_delta(
                        before,
                        after,
                        ("steps", "generated", "repaired", "fallbacks", "illegal"),
                    ),
                }
            history.append(entry)
            print(
                f"  turn {entry['turn']:>3} {player}  orders={entry['orders']:>2}  "
                f"attempts={entry['attempts']}  "
                f"units A={len(state.units_of('A'))} B={len(state.units_of('B'))}",
                flush=True,
            )
    except LlmTransportError as exc:
        error = str(exc)
        print(f"  LLM TRANSPORT FAILURE: {error}", flush=True)
    except Exception as exc:  # keep the league alive, save the partial game
        error = f"{type(exc).__name__}: {exc}"
        print(f"  GAME FAILURE: {error}", flush=True)

    match_path.parent.mkdir(parents=True, exist_ok=True)
    save_game(state, match_path)
    orders_path.write_text(json.dumps(orders, indent=2) + "\n", encoding="utf-8")

    ncm_stats = _ncm_stats(ncm) if ncm is not None else None
    agent_stats = {side: _ncm_stats(agents[side]) for side in ncm_sides}
    board = {"A": _board_stats(state, "A"), "B": _board_stats(state, "B")}
    board_margin = round(float(board["B"]["score"]) - float(board["A"]["score"]), 2)
    board_winner = _board_winner(board, getattr(args, "cap_margin", 0.0))
    winner_side = state.winner
    decided_by = "victory" if winner_side else "unfinished"
    if error is not None:
        decided_by = "error"
    elif winner_side is None and state.status in ("draw", "ongoing"):
        if getattr(args, "cap_decision", "board") == "board":
            if board_winner is not None:
                winner_side = board_winner
                decided_by = "board"
            else:
                decided_by = "draw"
        else:
            decided_by = "draw" if state.status == "draw" else "unfinished"
    unfinished = winner_side is None and decided_by == "unfinished"
    leader = max(("A", "B"), key=lambda side: (board[side]["score"], board[side]["units"]))
    if ncm_side == "both":
        opponent = "ncm"
    elif ncm_sides:
        opponent = kinds["B" if ncm_sides[0] == "A" else "A"]
    else:
        opponent = kinds["B"] if kinds["A"] == "llm" else kinds["A"]

    ncm_configs = {
        side: {
            "kind": kinds[side],
            "checkpoint": str(
                Path(args.ncm_persona_dir) / NCM_PERSONAS[kinds[side]]
            ),
            "generator": args.ncm_generator,
            "decode": args.ncm_decode,
            "prefix_len": factory.prefix_for(fixture.map_name),
            "max_steps": args.max_steps,
        }
        for side in ncm_sides
    }
    first_config = ncm_configs[ncm_sides[0]] if ncm_sides else None
    meta = {
        "schema": 1,
        "source": "all_ai_league",
        "label": fixture.label,
        "scenario": state.scenario_name,
        "scenario_path": fixture.map_path,
        "seed": state.seed,
        "ncm_side": ncm_side,
        "opponent": opponent,
        "max_steps": args.max_steps,
        "checkpoint": first_config["checkpoint"] if first_config else None,
        "ncm_config": first_config,
        "ncm_configs": ncm_configs,
        "learner": ncm_stats,
        "ncm": ncm_stats,
        "agent_stats": agent_stats,
        "fallback_reasons": dict(Counter(ncm.fallback_reasons)) if ncm is not None else {},
        "history": history,
        "team_labels": {"A": TEAM_LABELS[fixture.agent_a], "B": TEAM_LABELS[fixture.agent_b]},
        "started_at": started,
        "finished_at": _time_text_now(),
        "seconds": round(time.time() - started_clock, 2),
        "winner": winner_side,
        "finished": decided_by in ("victory", "board", "draw"),
        "league": {
            "id": fixture.id,
            "map": fixture.map_name,
            "agents": {"A": fixture.agent_a, "B": fixture.agent_b},
            "board": board,
            "board_margin": board_margin,
            "status": state.status,
            "winner": winner_side,
            "decided_by": decided_by,
            "turn_cap": turn_cap,
            "unfinished": unfinished,
            "leader": leader if unfinished and winner_side is None else None,
            "agent_stats": agent_stats,
            "ncm_configs": ncm_configs,
            "error": error,
        },
    }

    # A replay mismatch is a real error; record it rather than hiding it.
    try:
        rebuilt = replay(scenario, json.loads(json.dumps(orders)))
        meta["replay_matches"] = state_hash(rebuilt) == state_hash(state)
    except Exception as exc:  # pragma: no cover - defensive for a corrupt partial save
        meta["replay_matches"] = False
        meta["replay_error"] = f"{type(exc).__name__}: {exc}"

    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return _result_from_meta(meta, match_path)


def _result_from_meta(meta: dict[str, Any], match_path: Path) -> dict[str, Any]:
    league = meta.get("league") or {}
    board = league.get("board") or {}
    winner = league.get("winner")
    agents = league.get("agents") or {}
    winner_agent = agents.get(winner) if winner in agents else None
    unfinished = bool(league.get("unfinished"))
    leader_side = league.get("leader")
    leader_agent = agents.get(leader_side) if leader_side in agents else None
    return {
        "id": league.get("id"),
        "map": league.get("map"),
        "agents": agents,
        "board": board,
        "status": league.get("status"),
        "winner_side": winner,
        "winner": winner_agent,
        "decided_by": league.get("decided_by"),
        "board_margin": league.get("board_margin"),
        "turn_cap": league.get("turn_cap"),
        "unfinished": unfinished,
        "leader": leader_agent if unfinished and winner_agent is None else None,
        "error": league.get("error"),
        "match_path": str(match_path),
        "meta_path": str(match_path.with_name(match_path.stem + ".meta.json")),
        "seconds": meta.get("seconds"),
        "replay_matches": meta.get("replay_matches"),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _blank_row(agent: str) -> dict[str, Any]:
    return {
        "agent": agent,
        "label": TEAM_LABELS.get(agent, agent),
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "unfinished": 0,
        "failed": 0,
        "points": 0.0,
        "board_total": 0.0,
        "units_total": 0.0,
    }


def _standings(results: list[dict[str, Any]], agents: list[str]) -> dict[str, dict[str, Any]]:
    rows = {agent: _blank_row(agent) for agent in agents}
    for result in results:
        sides = result.get("agents") or {}
        for side, agent in sides.items():
            if agent not in rows:
                continue
            row = rows[agent]
            row["games"] += 1
            board = (result.get("board") or {}).get(side) or {}
            row["board_total"] += float(board.get("score", 0.0))
            row["units_total"] += int(board.get("units", 0))
            if result.get("error"):
                row["failed"] += 1
                continue
            if result.get("unfinished"):
                row["unfinished"] += 1
                row["draws"] += 1
                row["points"] += 0.5
            elif result.get("winner") == agent:
                row["wins"] += 1
                row["points"] += 1.0
            elif result.get("winner"):
                row["losses"] += 1
            else:
                row["draws"] += 1
                row["points"] += 0.5
    for row in rows.values():
        games = row["games"] or 1
        row["average_score"] = round(row["board_total"] / games, 2)
        row["average_units"] = round(row["units_total"] / games, 2)
    return rows


def _head_to_head(results: list[dict[str, Any]], agents: list[str]) -> dict[str, dict[str, dict[str, int]]]:
    table = {
        left: {right: {"wins": 0, "losses": 0, "draws": 0, "failed": 0} for right in agents if right != left}
        for left in agents
    }
    for result in results:
        sides = result.get("agents") or {}
        for side, agent in sides.items():
            other_side = "B" if side == "A" else "A"
            other = sides.get(other_side)
            if agent not in table or other not in table[agent]:
                continue
            entry = table[agent][other]
            if result.get("error"):
                entry["failed"] += 1
            elif result.get("unfinished") or not result.get("winner"):
                entry["draws"] += 1
            elif result.get("winner") == agent:
                entry["wins"] += 1
            else:
                entry["losses"] += 1
    return table


def _render_markdown(
    results: list[dict[str, Any]],
    agents: list[str],
    *,
    batch: str,
    maps: list[str],
    max_turns: int,
    max_steps: int = 0,
    only_ncm: bool = False,
) -> str:
    standings = _standings(results, agents)
    h2h = _head_to_head(results, agents)
    ordered = sorted(
        standings.values(),
        key=lambda row: (-row["points"], -row["average_score"], row["agent"]),
    )
    lines = [
        f"# All-AI league — {batch}",
        "",
        f"- Representatives: {', '.join(TEAM_LABELS.get(agent, agent) for agent in agents)}",
        f"- Maps: {', '.join(maps)}",
        f"- Turn cap: {'map defaults' if max_turns == 0 else max_turns}",
        f"- NCM steps/turn: {max_steps if max_steps else 'default'}",
        f"- Scope: {'NCM-involving fixtures only' if only_ncm else 'complete round-robin'}",
        f"- Games: {len(results)}",
        "",
        "## Standings",
        "",
        "| Agent | Games | W-L-D-U | Points | Avg board | Avg units | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['label']} | {row['games']} | "
            f"{row['wins']}-{row['losses']}-{row['draws'] - row['unfinished']}-{row['unfinished']} | "
            f"{row['points']:.1f} | {row['average_score']:.2f} | "
            f"{row['average_units']:.1f} | {row['failed']} |"
        )
    lines += [
        "",
        "Capped games are awarded to the higher board score; exact score ties are draws.",
        "The standings tiebreak is total points, then average board score.",
        "",
        "## Head to head",
        "",
        "| Agent | " + " | ".join(TEAM_LABELS.get(agent, agent) for agent in agents if agent != agents[0]) + " |",
        "|---|" + "|".join("---" for _ in agents[1:]) + "|",
    ]
    for left in agents:
        cells = []
        for right in agents:
            if left == right:
                continue
            row = h2h[left][right]
            cells.append(f"{row['wins']}-{row['losses']}-{row['draws']}" + (f" ({row['failed']} failed)" if row["failed"] else ""))
        lines.append(f"| {TEAM_LABELS.get(left, left)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Games",
        "",
        "| Map | Match | Result | Board A / B | Replay |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        agents_by_side = result.get("agents") or {}
        match = f"{TEAM_LABELS.get(agents_by_side.get('A'), agents_by_side.get('A'))} vs {TEAM_LABELS.get(agents_by_side.get('B'), agents_by_side.get('B'))}"
        if result.get("error"):
            verdict = f"failed: {result['error']}"
        elif result.get("winner"):
            verdict = f"winner: {TEAM_LABELS.get(result['winner'], result['winner'])}"
            if result.get("decided_by") == "board":
                verdict += " (board)"
        elif result.get("unfinished"):
            verdict = f"unfinished; leader: {TEAM_LABELS.get(result.get('leader'), result.get('leader') or '-')}"
        else:
            verdict = "draw"
        board = result.get("board") or {}
        board_text = f"{board.get('A', {}).get('score', '-')} / {board.get('B', {}).get('score', '-')}"
        replay_text = "yes" if result.get("replay_matches") else "NO"
        lines.append(f"| {result.get('map')} | {match} | {verdict} | {board_text} | {replay_text} |")
    lines += ["", "## Match files", ""]
    for result in results:
        lines.append(f"- `{result.get('match_path')}`")
    return "\n".join(lines) + "\n"


def write_report(
    results: list[dict[str, Any]],
    agents: list[str],
    *,
    batch: str,
    maps: list[str],
    max_turns: int,
    out_dir: Path,
    max_steps: int = 0,
    only_ncm: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch": batch,
        "agents": agents,
        "labels": {agent: TEAM_LABELS.get(agent, agent) for agent in agents},
        "maps": maps,
        "max_turns": max_turns,
        "max_steps": max_steps,
        "only_ncm": only_ncm,
        "games": len(results),
        "standings": _standings(results, agents),
        "head_to_head": _head_to_head(results, agents),
        "results": results,
        "updated_at": _now(),
    }
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(
        _render_markdown(
            results,
            agents,
            batch=batch,
            maps=maps,
            max_turns=max_turns,
            max_steps=max_steps,
            only_ncm=only_ncm,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _load_saved_results(
    fixtures: list[Fixture],
    *,
    match_dir: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    results: list[dict[str, Any]] = []
    done: set[str] = set()
    for fixture in fixtures:
        match_path = match_dir / fixture.filename
        meta_path = match_path.with_name(match_path.stem + ".meta.json")
        if not match_path.is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("source") != "all_ai_league":
            continue
        results.append(_result_from_meta(meta, match_path))
        done.add(fixture.id)
    return results, done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="all_ai_league",
        description="round-robin LLM vs NCM vs conventional AI league",
    )
    parser.add_argument(
        "--agents",
        default=",".join(TEAM_ORDER),
        help="comma-separated representatives (default: all four)",
    )
    parser.add_argument("--maps", default="all", help="all or comma-separated map stems")
    parser.add_argument(
        "--only-ncm",
        action="store_true",
        help="only play fixtures with at least one NCM representative",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="override every map's authored limit; 0 = use each map's max_turns",
    )
    parser.add_argument("--seed-base", type=int, default=880000)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument(
        "--match-dir",
        default=str(DEFAULT_NCM_ROOT / "artifacts" / "tbs" / "matches"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(GAME_REPO / "docs" / "all_ai_league"),
    )
    parser.add_argument("--ncm-root", default=str(DEFAULT_NCM_ROOT))
    parser.add_argument(
        "--ncm-persona-dir",
        default=str(DEFAULT_NCM_ROOT / "artifacts" / "tbs" / "personas"),
        help="directory holding the armor/artillery persona checkpoints",
    )
    parser.add_argument("--ncm-device", default=None)
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="override prefix length for every map; 0 = auto (small/large defaults)",
    )
    parser.add_argument("--prefix-len-small", type=int, default=2048)
    parser.add_argument("--prefix-len-large", type=int, default=4096)
    parser.add_argument(
        "--ncm-generator",
        default="mixed_generator.pt",
        help="generator filename inside each persona checkpoint",
    )
    parser.add_argument("--ncm-decode", choices=("sample", "legal"), default="legal")
    parser.add_argument("--ncm-temperature", type=float, default=0.0)
    parser.add_argument("--ncm-retries", type=int, default=3)
    parser.add_argument("--ncm-retry-temperature", type=float, default=0.5)
    parser.add_argument("--ncm-fallback", choices=("persona", "none"), default="persona")
    parser.add_argument("--max-steps", type=int, default=8, help="NCM orders per turn")
    parser.add_argument(
        "--conventional-max-steps",
        type=int,
        default=64,
        help="stepwise orders per turn for the conventional heuristic",
    )
    parser.add_argument("--heuristic-seed", type=int, default=900000)
    parser.add_argument(
        "--cap-decision",
        choices=("board", "draw"),
        default="board",
        help="award capped games on board score, or keep them as draws",
    )
    parser.add_argument(
        "--cap-margin",
        type=float,
        default=0.0,
        help="minimum board-score margin for a board decision; 0 awards any lead",
    )
    parser.add_argument(
        "--llm-auto-tactics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fill or replace holds with legal HQ attacks/captures",
    )
    parser.add_argument("--llm-model", default=os.environ.get("TBS_LLM_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("TBS_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
    )
    parser.add_argument("--llm-api-key", default=os.environ.get("TBS_LLM_API_KEY"))
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1500)
    parser.add_argument("--llm-timeout", type=float, default=600.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-attempts", type=int, default=2)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument(
        "--llm-log",
        default=str(GAME_REPO / "logs" / "llm" / "all_ai_league.jsonl"),
    )
    parser.add_argument("--resume", action="store_true", help="skip saved fixtures")
    parser.add_argument("--report-only", action="store_true", help="only rebuild the report")
    parser.add_argument("--dry-run", action="store_true", help="list fixtures and exit")
    parser.add_argument(
        "--monitor-url",
        default="http://127.0.0.1:8765",
        help="NCM dashboard base URL, checked before the league starts",
    )
    parser.add_argument("--limit", type=int, default=0, help="run at most N fixtures")
    args = parser.parse_args(argv)

    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    unknown = [agent for agent in agents if agent not in TEAM_LABELS]
    if unknown:
        parser.error(f"unknown representative(s): {', '.join(unknown)}")
    if len(set(agents)) != len(agents):
        parser.error("representatives must be unique")
    if len(agents) < 2:
        parser.error("a league needs at least two representatives")
    args.agent_list = agents

    maps = _resolve_maps(args.maps)
    fixtures = build_fixtures(agents, maps, seed_base=args.seed_base, batch=args.batch)
    if args.only_ncm:
        fixtures = [
            fixture
            for fixture in fixtures
            if fixture.agent_a in NCM_KINDS or fixture.agent_b in NCM_KINDS
        ]
        if not fixtures:
            parser.error("--only-ncm selected no fixtures")
    match_dir = Path(args.match_dir)
    out_dir = Path(args.out_dir) / args.batch

    if args.dry_run:
        print(f"batch {args.batch}: {len(fixtures)} fixtures on {len(maps)} maps")
        for fixture in fixtures:
            print(
                f"{fixture.map_name:22} {fixture.agent_a:8} vs {fixture.agent_b:8} "
                f"seed={fixture.seed}  -> {fixture.filename}"
            )
        return 0

    saved_results, done = _load_saved_results(fixtures, match_dir=match_dir)
    if args.report_only:
        write_report(
            saved_results,
            agents,
            batch=args.batch,
            maps=[path.stem for path in maps],
            max_turns=args.max_turns,
            max_steps=args.max_steps,
            only_ncm=args.only_ncm,
            out_dir=out_dir,
        )
        print(f"report {out_dir / 'report.json'}")
        return 0

    _check_monitor(args.monitor_url)

    if args.resume:
        results = saved_results
    else:
        results = []

    logger = JsonlLogger(args.llm_log) if "llm" in agents else None
    specs = load_specs()
    factory = AgentFactory(args, logger)
    pending = [fixture for fixture in fixtures if not (args.resume and fixture.id in done)]
    if args.limit:
        pending = pending[: args.limit]

    if not pending:
        write_report(
            results,
            agents,
            batch=args.batch,
            maps=[path.stem for path in maps],
            max_turns=args.max_turns,
            max_steps=args.max_steps,
            only_ncm=args.only_ncm,
            out_dir=out_dir,
        )
        print("nothing to do; every fixture already has a saved game")
        return 0

    print(
        f"batch {args.batch}: {len(pending)} to play / {len(fixtures)} fixtures, "
        f"{len(maps)} maps, cap {'each map default' if args.max_turns == 0 else args.max_turns} turns",
        flush=True,
    )
    for index, fixture in enumerate(pending, start=1):
        print(f"\n=== {index}/{len(pending)} {fixture.id} ===", flush=True)
        try:
            result = play_game(
                fixture,
                factory=factory,
                specs=specs,
                args=args,
                logger=logger,
            )
        except Exception as exc:
            print(f"FIXTURE FAILED: {type(exc).__name__}: {exc}", flush=True)
            result = {
                "id": fixture.id,
                "map": fixture.map_name,
                "agents": {"A": fixture.agent_a, "B": fixture.agent_b},
                "board": {},
                "status": "failed",
                "winner_side": None,
                "winner": None,
                "unfinished": False,
                "leader": None,
                "error": f"{type(exc).__name__}: {exc}",
                "match_path": str(match_dir / fixture.filename),
                "meta_path": str(match_dir / (fixture.filename[:-5] + ".meta.json")),
                "seconds": None,
                "replay_matches": False,
            }
        results.append(result)
        write_report(
            results,
            agents,
            batch=args.batch,
            maps=[path.stem for path in maps],
            max_turns=args.max_turns,
            max_steps=args.max_steps,
            only_ncm=args.only_ncm,
            out_dir=out_dir,
        )
        board = result.get("board") or {}
        print(
            f"result {result.get('map')} {result.get('agents', {}).get('A')} vs "
            f"{result.get('agents', {}).get('B')}: "
            f"status={result.get('status')} winner={result.get('winner')} "
            f"board={board.get('A', {}).get('score')}/{board.get('B', {}).get('score')}",
            flush=True,
        )

    write_report(
        results,
        agents,
        batch=args.batch,
        maps=[path.stem for path in maps],
        max_turns=args.max_turns,
        max_steps=args.max_steps,
        only_ncm=args.only_ncm,
        out_dir=out_dir,
    )
    print(f"\nreport {out_dir / 'report.json'}\n        {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
