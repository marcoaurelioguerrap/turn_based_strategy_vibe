"""Saving, loading and hashing a game state.

Save files are plain JSON of :meth:`~tbs.engine.state.GameState.to_dict`.
Spec data is not stored in a save file: it is loaded from ``data/`` when the
save is read, so rebalancing units or terrain does not invalidate saved games.

:func:`state_hash` is what the determinism tests compare: the same seed driven
by the same actions must produce the same hash.  The action log is deliberately
excluded (it is history, not position) and has its own :func:`log_hash`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import GameSpecs, load_specs
from .state import GameState, new_game


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def state_to_dict(state: GameState) -> dict[str, Any]:
    return state.to_dict()


def state_hash(state: GameState) -> str:
    """SHA-256 over the canonical JSON of the position (excluding the log)."""
    data = state.to_dict()
    data.pop("log", None)
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def log_hash(state: GameState) -> str:
    """SHA-256 over the recorded action log."""
    return hashlib.sha256(canonical_json(state.log).encode("utf-8")).hexdigest()


def state_from_dict(data: dict[str, Any], specs: GameSpecs | None = None) -> GameState:
    return GameState.from_dict(data, specs or load_specs())


def save_game(state: GameState, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_game(path: Path | str, specs: GameSpecs | None = None) -> GameState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return state_from_dict(data, specs)


def replay(scenario: dict[str, Any], actions: list[dict[str, Any]], seed: int | None = None, specs: GameSpecs | None = None) -> GameState:
    """Rebuild a game by replaying actions, for debugging and determinism checks."""
    from . import rules

    scenario = dict(scenario)
    if seed is not None:
        scenario["seed"] = seed
    state = new_game(scenario, specs or load_specs())
    rules.start_game(state)
    rules.apply_actions(state, actions)
    return state
