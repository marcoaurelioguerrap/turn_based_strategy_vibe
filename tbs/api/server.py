"""STDIO JSON-lines agent server: one JSON object per line in, one out.

The engine is pure and I/O-free; this module is the outermost shell around it.
Every reply carries ``"ok"`` plus either the result or a one-sentence
``"error"`` an agent can act on and retry from - rejected orders never touch
the game.  On start the server writes one greeting line (its ``hello`` answer)
so a client learns the scenario and protocol unprompted; after that it answers
exactly one line per request line.

Run::

    py -m tbs.api.server                              # Border Skirmish (the default map)
    py -m tbs.api.server --scenario data/maps/x.json --seed 7
    py -m tbs.api.server --session saves/mid_game.json

Commands are listed in ``COMMANDS``/``ACTION_SHAPES`` below and every one of
them is also served over the wire by ``help``, so an agent can discover the
whole protocol at runtime without reading this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePath
from typing import Any, Callable

from tbs.engine import rules, view
from tbs.engine.data import default_scenario, load_scenario, load_specs
from tbs.engine.errors import ActionError
from tbs.engine.serialize import load_game, save_game, state_hash
from tbs.engine.state import GameState, new_game

PROTOCOL = "tbs-1"

#: The commands this server answers, as data - served verbatim by ``help``.
COMMANDS = {
    "hello": "who the server is: protocol, scenario, players, command list",
    "help": "the whole protocol: command list and order shapes",
    "state": 'the prompt-sized view for a player: {"cmd":"state","player":"A"} (default: whoever is to move)',
    "status": "a cheap poll: turn, status, current player, funds and unit counts per player",
    "options": 'everything one unit may be ordered to do right now, with damage previews: {"cmd":"options","unit":"A5"}',
    "order": 'perform one order, fields inlined next to the cmd: {"cmd":"order","action":"attack","unit":"A5","target":"B5"}',
    "save": 'write the game as plain JSON: {"cmd":"save","path":"mid_game.json"} (path relative to the server root)',
    "load": 'continue from such a file: {"cmd":"load","path":"mid_game.json"}',
}

#: Every order shape the engine accepts, as data - served verbatim by ``help``.
ACTION_SHAPES: dict[str, dict[str, Any]] = {
    "move": {"action": "move", "unit": "<id>", "path": "[[q,r], ...] contiguous, starting at the unit's own hex"},
    "attack": {"action": "attack", "unit": "<id>", "target": "<enemy unit id>", "weapon": "<optional weapon id>"},
    "capture": {"action": "capture", "unit": "<id of a unit standing on a capturable site>"},
    "load": {"action": "load", "unit": "<transport id>", "cargo": "<adjacent loadable unit id>"},
    "unload": {"action": "unload", "unit": "<transport id>", "cargo": "<carried id>", "target": "[q,r] adjacent to the transport"},
    "build": {"action": "build", "site": "[q,r] own factory (land) / port (naval) / airport (air)", "unit": "<unit type id>"},
    "hold": {"action": "hold", "unit": "<id>"},
    "end_turn": {"action": "end_turn"},
}

_ENGINE_ERRORS = (ActionError, KeyError, ValueError, OSError)


class Session:
    """One served game: pure engine calls behind a request/response shell."""

    def __init__(self, state: GameState, root: str | Path | None = None):
        self.state = state
        self.root = Path(root) if root is not None else Path.cwd()

    @classmethod
    def new(cls, scenario: str | Path | None = None, seed: int | None = None, root: str | Path | None = None) -> "Session":
        path = Path(scenario) if scenario is not None else default_scenario()
        data = load_scenario(path)
        if seed is not None:
            data["seed"] = seed
        return cls(new_game(data, load_specs()), root=root)

    def handle(self, request: Any) -> dict[str, Any]:
        """Answer one request object with one response object. Never raises."""
        if not isinstance(request, dict):
            return _error("a request must be one JSON object with a 'cmd' field")
        cmd = request.get("cmd")
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "hello": self._hello,
            "help": self._help,
            "state": self._state,
            "status": self._status,
            "options": self._options,
            "order": self._order,
            "save": self._save,
            "load": self._load,
        }
        if not isinstance(cmd, str) or cmd not in handlers:
            return _error(f"unknown command {cmd!r}; this server answers: {', '.join(sorted(handlers))}")
        try:
            return handlers[cmd](request)
        except _ENGINE_ERRORS as problem:
            return _error(str(problem))

    def handle_line(self, line: str) -> str:
        """The wire-level entry point: a raw line in, a raw JSON line out."""
        try:
            request = json.loads(line)
        except json.JSONDecodeError as problem:
            payload = _error(f"every line must be one JSON object: {problem.msg} (line {problem.lineno} column {problem.colno})")
        else:
            payload = self.handle(request)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # -- the commands, straight onto the tested engine API ------------------

    def _hello(self, request: dict[str, Any]) -> dict[str, Any]:
        state = self.state
        return {
            "ok": True,
            "protocol": PROTOCOL,
            "scenario": state.scenario_name,
            "players": [player.id for player in state.players.values()],
            "commands": COMMANDS,
            **self._status_fields(),
        }

    def _help(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "protocol": PROTOCOL, "commands": COMMANDS, "orders": ACTION_SHAPES}

    def _state(self, request: dict[str, Any]) -> dict[str, Any]:
        player = request.get("player")
        if player is not None and player not in self.state.players:
            return _error(f"there is no player {player!r}; this game has: {', '.join(self.state.players)}")
        return {"ok": True, "view": view.state_view(self.state, player)}

    def _status(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, **self._status_fields()}

    def _options(self, request: dict[str, Any]) -> dict[str, Any]:
        unit_id = request.get("unit")
        if not isinstance(unit_id, str) or not unit_id:
            return _error("options needs a 'unit' id, e.g. {\"cmd\":\"options\",\"unit\":\"A5\"}")
        return {"ok": True, "options": rules.unit_options(self.state, unit_id)}

    def _order(self, request: dict[str, Any]) -> dict[str, Any]:
        order = {key: value for key, value in request.items() if key != "cmd"}
        if "action" not in order:
            return _error(f"an order needs an 'action'; this engine knows: {', '.join(sorted(ACTION_SHAPES))}")
        events = rules.perform(self.state, order)
        return {"ok": True, "events": events, "state_hash": state_hash(self.state), **self._status_fields()}

    def _save(self, request: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(request)
        save_game(self.state, path)
        return {"ok": True, "path": str(path.relative_to(self.root))}

    def _load(self, request: dict[str, Any]) -> dict[str, Any]:
        path = self._safe_path(request)
        self.state = load_game(path)
        return {"ok": True, "loaded": str(path.relative_to(self.root)), **self._status_fields()}

    # -- helpers ------------------------------------------------------------

    def _safe_path(self, request: dict[str, Any]) -> Path:
        raw = request.get("path")
        if not isinstance(raw, str) or not raw:
            raise ValueError('save/load need a "path" string relative to the server root')
        pure = PurePath(raw)
        if pure.is_absolute() or pure.drive or pure.root:
            raise ValueError(f"path {raw!r} must be relative to the server root, not absolute")
        if ".." in pure.parts:
            raise ValueError(f"path {raw!r} may not climb out of the server root with '..'")
        target = (self.root / pure).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise ValueError(f"path {raw!r} resolves outside the server root")
        return target

    def _status_fields(self) -> dict[str, Any]:
        state = self.state
        return {
            "turn": state.turn_number,
            "status": state.status,
            "current_player": state.current_player,
            "winner": state.winner,
            "players": {
                player.id: {"funds": player.funds, "units": len(state.units_of(player.id))}
                for player in state.players.values()
            },
        }


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def serve(session: Session, input_stream=sys.stdin) -> None:
    """Run the request/response loop until the client closes the stream."""
    greeting = session.handle({"cmd": "hello"})
    greeting["event"] = "hello"
    print(json.dumps(greeting, sort_keys=True, separators=(",", ":")), flush=True)
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        print(session.handle_line(line), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tbs.api.server", description="tbs agent server: JSON objects in, JSON objects out, one per line")
    parser.add_argument("--scenario", help="scenario JSON path (default: the bundled border skirmish)")
    parser.add_argument("--seed", type=int, help="override the scenario seed")
    parser.add_argument("--session", help="resume from a save file instead of starting a scenario")
    parser.add_argument("--root", help="directory save/load paths are relative to (default: the current directory)")
    args = parser.parse_args(argv)
    if args.session:
        session = Session(load_game(Path(args.session)), root=args.root)
    else:
        session = Session.new(scenario=args.scenario, seed=args.seed, root=args.root)
    serve(session)


if __name__ == "__main__":
    main()
