"""Compact, deterministic text codecs for a game view and for single orders.

The engine's JSON view is built for debugging and external agents: one full turn
view is roughly 3.8-4.7k GPT-2 tokens.  NCM's prefix is a fraction of that, so
this module defines a line-oriented encoding of the same board facts in about
500 tokens for the bundled scenario, plus a one-line form for orders.  It is
stdlib-only, has no knowledge of the rules, and never mutates a view.

State format
------------

::

    #ncm1 A t3
    players
    A 17300 +3100 8
    B 20400 +3100 8
    terrain
    ..C...##...C..
    ...
    sites
    2,0 C A
    0,2 H A 40/300!
    units
    A1 hq 0,2 100 - mg:20
    A2 infantry 4,3 20 - rifle:10 !>
    B3 infantry 6,3 4 - rifle:9
    builds
    1,1 infantry 700
    ...

The header names the acting player and turn (plus any non-``ongoing`` status).
``terrain`` repeats the view's authored rows verbatim.  Ownership rows are not
repeated: outside a site's owner they have no rule effect, and site and build
records carry what matters.  A unit record is ``id type pos hp fuel ammo
[flags]``; ``-`` means none for fuel, ammo or a loaded unit's position.  Flags
are ``!`` acted, ``>`` moved, ``+`` produced this turn, ``^`` loaded and
``c<ids>`` cargo.  A site record only adds ``progress/needed`` while a capture
is in progress, and ``!`` marks an objective.  Build records are the engine's
legal, affordable production options: ``site type cost``.

The codec is deliberately lossy about static spec data (unit names, max hp,
movement ranges) and about the engine's ``unit_actions`` option lists.  Static
data is shared by every state in a dataset, and legality remains the
coordinator's job: a rejected order comes back as feedback.  That is what keeps
a state near 500 tokens instead of 4,000.

Measured with the released NCM GPT-2 tokenizer, the first five player-turns of
the reference scenario encode to 511-541 tokens (1,061-1,125 characters), against
3,757-4,655 tokens for the full JSON view.  The engine test suite is stdlib-only,
so it pins the character budget and the facts below; the token counts are
recorded here as the measurement that motivated the budget.

Order format
------------

::

    m A2 3,4              move to [3,4]
    a A5 B5 howitzer      attack (weapon optional)
    c A4                  capture the site under the unit
    l A6 A2               load A2 into transport A6
    u A6 A2 2,3           unload A2 at [2,3]
    b 1,1 infantry        build at [1,1]
    h A1                  hold
    e                     end turn

:func:`encode_order` accepts both agent orders (``to``) and engine orders
(``path``); the destination of a path is its last hex.  :func:`decode_order`
returns the canonical agent form that
:func:`tbs.agents.turn.normalize_order` already accepts, so orders can be
stored and replayed without the engine ever importing this module.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CodecError", "canonical_order", "decode_order", "encode_order", "encode_state"]


class CodecError(ValueError):
    """The text or order does not fit the codec's grammar."""


# ---------------------------------------------------------------------------
# coordinates and small shared helpers
# ---------------------------------------------------------------------------


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodecError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


def _coord(value: Any, what: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CodecError(f"{what} must be a [q,r] coordinate, got {value!r}")
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        raise CodecError(f"{what} must be a [q,r] coordinate, got {value!r}") from None


def _coord_text(value: Any, what: str) -> str:
    q, r = _coord(value, what)
    return f"{q},{r}"


def _parse_coord(text: str, what: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise CodecError(f"{what} must be written q,r, got {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise CodecError(f"{what} must be written q,r, got {text!r}") from None


# ---------------------------------------------------------------------------
# orders
# ---------------------------------------------------------------------------

_UNIT_ACTIONS = ("move", "attack", "capture", "load", "unload", "build", "hold")


def canonical_order(order: dict[str, Any]) -> dict[str, Any]:
    """The minimal agent form of one order, with ``path`` reduced to ``to``.

    Extra keys are dropped.  A move that carries an engine ``path`` keeps only
    its destination, because :func:`~tbs.agents.turn.normalize_order` expands
    that back to the same path from the engine's own legal moves.
    """
    if not isinstance(order, dict):
        raise CodecError(f"an order must be a JSON object, got {type(order).__name__}")
    action = order.get("action")
    if action == "end_turn":
        return {"action": "end_turn"}
    if action not in _UNIT_ACTIONS:
        raise CodecError(
            f"unknown action {action!r}; use one of: "
            f"{', '.join((*_UNIT_ACTIONS, 'end_turn'))}"
        )
    unit = _text(order.get("unit"), f"{action} order's unit")

    if action == "move":
        destination = order.get("to")
        if destination is None:
            path = order.get("path")
            if not isinstance(path, (list, tuple)) or not path:
                raise CodecError(
                    'a move needs "to":[q,r] or a non-empty "path"'
                )
            destination = path[-1]
        return {"action": "move", "unit": unit, "to": list(_coord(destination, "move destination"))}

    if action == "attack":
        target = _text(order.get("target"), "attack order's target")
        canonical: dict[str, Any] = {"action": "attack", "unit": unit, "target": target}
        weapon = order.get("weapon")
        if weapon not in (None, ""):
            canonical["weapon"] = _text(weapon, "attack order's weapon")
        return canonical

    if action in ("capture", "hold"):
        return {"action": action, "unit": unit}

    if action == "load":
        cargo = _text(order.get("cargo"), "load order's cargo")
        return {"action": "load", "unit": unit, "cargo": cargo}

    if action == "unload":
        cargo = _text(order.get("cargo"), "unload order's cargo")
        target = _coord(order.get("target"), "unload destination")
        return {"action": "unload", "unit": unit, "cargo": cargo, "target": list(target)}

    # build: "unit" is the unit type and "site" is where it appears.
    site = _coord(order.get("site"), "build site")
    return {"action": "build", "unit": unit, "site": list(site)}


def encode_order(order: dict[str, Any]) -> str:
    """Encode one order as a single mnemonic line."""
    canonical = canonical_order(order)
    action = canonical["action"]
    if action == "end_turn":
        return "e"
    if action == "move":
        return f"m {canonical['unit']} {_coord_text(canonical['to'], 'move destination')}"
    if action == "attack":
        line = f"a {canonical['unit']} {canonical['target']}"
        if "weapon" in canonical:
            line += f" {canonical['weapon']}"
        return line
    if action == "capture":
        return f"c {canonical['unit']}"
    if action == "hold":
        return f"h {canonical['unit']}"
    if action == "load":
        return f"l {canonical['unit']} {canonical['cargo']}"
    if action == "unload":
        return (
            f"u {canonical['unit']} {canonical['cargo']} "
            f"{_coord_text(canonical['target'], 'unload destination')}"
        )
    return f"b {_coord_text(canonical['site'], 'build site')} {canonical['unit']}"


def _arity(line: str, args: list[str], low: int, high: int | None = None) -> None:
    high = low if high is None else high
    if low <= len(args) <= high:
        return
    wanted = str(low) if low == high else f"{low} to {high}"
    raise CodecError(f"{line!r} needs {wanted} argument(s), got {len(args)}")


def decode_order(line: str) -> dict[str, Any]:
    """Parse one mnemonic line into the canonical agent order form."""
    if not isinstance(line, str):
        raise CodecError(f"an order line must be text, got {type(line).__name__}")
    parts = line.split()
    if not parts:
        raise CodecError("an order line cannot be empty")
    code, args = parts[0], parts[1:]

    if code == "e":
        _arity(code, args, 0)
        return {"action": "end_turn"}
    if code == "m":
        _arity(code, args, 2)
        return {"action": "move", "unit": args[0], "to": list(_parse_coord(args[1], "move destination"))}
    if code == "a":
        _arity(code, args, 2, 3)
        order: dict[str, Any] = {"action": "attack", "unit": args[0], "target": args[1]}
        if len(args) == 3:
            order["weapon"] = args[2]
        return order
    if code == "c":
        _arity(code, args, 1)
        return {"action": "capture", "unit": args[0]}
    if code == "h":
        _arity(code, args, 1)
        return {"action": "hold", "unit": args[0]}
    if code == "l":
        _arity(code, args, 2)
        return {"action": "load", "unit": args[0], "cargo": args[1]}
    if code == "u":
        _arity(code, args, 3)
        return {
            "action": "unload",
            "unit": args[0],
            "cargo": args[1],
            "target": list(_parse_coord(args[2], "unload destination")),
        }
    if code == "b":
        _arity(code, args, 2)
        return {
            "action": "build",
            "site": list(_parse_coord(args[0], "build site")),
            "unit": args[1],
        }
    raise CodecError(
        f"unknown order code {code!r}; use m, a, c, l, u, b, h or e"
    )


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def _unit_flags(unit: dict[str, Any]) -> str:
    flags = ""
    if unit.get("acted"):
        flags += "!"
    if unit.get("moved"):
        flags += ">"
    if unit.get("new"):
        flags += "+"
    if unit.get("loaded"):
        flags += "^"
    cargo = unit.get("cargo") or []
    if cargo:
        flags += "c" + ",".join(str(item) for item in cargo)
    return flags


def encode_state(view: dict[str, Any]) -> str:
    """Encode a :func:`tbs.engine.view.state_view` dictionary as compact text.

    The result is canonical: players, sites, units and builds are sorted, so
    two views carrying the same facts encode to the same string regardless of
    input order.
    """
    if not isinstance(view, dict):
        raise CodecError(f"a state view must be a dict, got {type(view).__name__}")
    current = _text(view.get("current_player"), "view.current_player")
    turn = view.get("turn")
    if not isinstance(turn, int):
        raise CodecError(f"view.turn must be an int, got {turn!r}")
    terrain = view.get("map_terrain_rows")
    if not isinstance(terrain, (list, tuple)) or not terrain:
        raise CodecError("view.map_terrain_rows must be a non-empty list of rows")

    header = f"#ncm1 {current} t{turn}"
    status = view.get("status")
    if status not in (None, "ongoing"):
        header += f" {status}"
    lines = [header, "players"]
    for player in sorted(view.get("players", []), key=lambda item: str(item["id"])):
        lines.append(
            f"{_text(player.get('id'), 'player id')} "
            f"{player['funds']} +{player['income_per_turn']} {player['unit_count']}"
        )

    lines.append("terrain")
    lines.extend(str(row) for row in terrain)

    lines.append("sites")
    sites = sorted(view.get("sites", []), key=lambda site: (site["pos"][1], site["pos"][0]))
    for site in sites:
        line = (
            f"{_coord_text(site.get('pos'), 'site position')} "
            f"{_text(site.get('terrain'), 'site terrain')} "
            f"{site.get('owner') or '-'}"
        )
        progress = site.get("capture_progress") or 0
        if progress:
            line += f" {progress}/{site['capture_needed']}"
        if site.get("objective"):
            line += "!"
        lines.append(line)

    lines.append("units")
    units = sorted(view.get("units", []), key=lambda unit: str(unit["id"]))
    for unit in units:
        pos = unit.get("pos")
        ammo = ",".join(
            f"{weapon}:{count}"
            for weapon, count in sorted((unit.get("ammo") or {}).items())
        )
        parts = [
            _text(unit.get("id"), "unit id"),
            _text(unit.get("type"), "unit type"),
            "-" if pos is None else _coord_text(pos, "unit position"),
            str(unit.get("hp")),
            "-" if unit.get("fuel") is None else str(unit.get("fuel")),
            ammo or "-",
        ]
        flags = _unit_flags(unit)
        if flags:
            parts.append(flags)
        lines.append(" ".join(parts))

    lines.append("builds")
    options = sorted(
        view.get("build_options", []),
        key=lambda option: (option["site"][1], option["site"][0], option["cost"], option["unit"]),
    )
    for option in options:
        lines.append(
            f"{_coord_text(option.get('site'), 'build site')} "
            f"{_text(option.get('unit'), 'build type')} {option['cost']}"
        )
    return "\n".join(lines)
