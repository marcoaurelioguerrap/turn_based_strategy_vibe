"""OpenAI-compatible local LLM client and the single-agent driver.

The client is stdlib-only (``urllib``) so the engine's "no dependencies" rule
extends to the agent layer.  It targets the ``/v1/chat/completions`` shape that
Ollama, LM Studio, llama.cpp, vLLM and Unsloth Desktop all expose.

The reference local endpoint for this project is::

    http://127.0.0.1:8888/v1
    model: unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF:UD-IQ2_M

Nothing here mutates a game state: an agent only receives a ``TurnRequest`` and
returns an ``AgentPlan``.  The coordinator in :mod:`tbs.agents.turn` validates
and applies the orders.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tbs.agents.protocol import AgentPlan, PlanFormatError, TurnRequest, parse_plan

__all__ = [
    "DEFAULT_BASE_URL",
    "JsonlLogger",
    "LlmClient",
    "LlmConfig",
    "LlmResponseError",
    "LlmTransportError",
    "RETRYABLE_LLM_ERRORS",
    "SingleLlmAgent",
    "build_messages",
]

#: Unsloth Desktop's local OpenAI-compatible server on this machine.
DEFAULT_BASE_URL = "http://127.0.0.1:8888/v1"

class LlmError(Exception):
    """Base class for client failures."""


class LlmTransportError(LlmError):
    """The server could not be reached or rejected the request (4xx/5xx)."""


class LlmResponseError(LlmError):
    """The server answered, but not with usable chat-completion content."""


RETRYABLE_LLM_ERRORS = (PlanFormatError, LlmResponseError)


# ---------------------------------------------------------------------------
# configuration and logging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmConfig:
    """Everything needed to call one local OpenAI-compatible endpoint."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1500
    timeout: float = 180.0
    retries: int = 2
    json_mode: bool = False
    #: Off by default: a reasoning turn can spend the whole completion budget
    #: before emitting the JSON orders, and the local reference server exposes
    #: only an 8192-token request window.  Set True (or --llm-thinking) for a
    #: reasoning model with a larger window.
    enable_thinking: bool | None = False
    reasoning_effort: str | None = None

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class JsonlLogger:
    """Append one JSON object per line, safe across the worker and UI threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), **payload}
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


# ---------------------------------------------------------------------------
# the HTTP client
# ---------------------------------------------------------------------------

#: (url, body, headers, timeout) -> raw response bytes.
Transport = Callable[[str, bytes, dict[str, str], float], bytes]


def _urllib_transport(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-chosen local URL
        return response.read()


class LlmClient:
    """One chat-completion call, with transport retries and clear errors."""

    def __init__(
        self,
        config: LlmConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.config.model:
            raise LlmTransportError(
                "no model configured; pass --llm-model or set TBS_LLM_MODEL"
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.config.enable_thinking is not None:
            # Unsloth/llama.cpp extension understood by the local server.  A
            # reasoning turn can consume the whole completion budget before it
            # emits JSON, so the game disables it by default on small contexts.
            payload["enable_thinking"] = self.config.enable_thinking
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        body = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                raw = self._transport(
                    self.config.endpoint, body, self.config.headers, self.config.timeout
                )
            except urllib.error.HTTPError as exc:
                detail = _http_detail(exc)
                last_error = LlmTransportError(
                    f"the LLM server answered HTTP {exc.code} at {self.config.endpoint}: {detail}"
                )
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.config.retries:
                    self._sleep(2**attempt * 0.5)
                    continue
                raise last_error from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = LlmTransportError(
                    f"cannot reach the LLM server at {self.config.endpoint}: {exc}"
                )
                if attempt < self.config.retries:
                    self._sleep(2**attempt * 0.5)
                    continue
                raise last_error from None
            else:
                return _content_from_response(raw)

        raise last_error or LlmTransportError("the LLM request failed")

    def as_agent(self, *, logger: JsonlLogger | None = None, history: int = 8) -> "SingleLlmAgent":
        return SingleLlmAgent(self, logger=logger, history=history)


def _http_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:  # pragma: no cover - body may already be consumed
        raw = b""
    text = raw.decode("utf-8", errors="replace").strip()
    if text:
        return " ".join(text.split())[:300]
    return exc.reason or "no response body"


def _content_from_response(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LlmResponseError(f"the LLM server did not return JSON: {exc}") from None
    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        message = error.get("message") if isinstance(error, dict) else error
        raise LlmResponseError(f"the LLM server returned an error: {message}")
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LlmResponseError(
            "the LLM response has no choices[0].message.content; "
            "is this an OpenAI-compatible /chat/completions endpoint?"
        ) from None
    text = _content_text(content)
    if not text.strip():
        message = choice.get("message") or {}
        reasoning = _content_text(message.get("reasoning_content"))
        detail = ""
        if reasoning:
            detail = (
                f" (it produced {len(reasoning)} characters of reasoning_content first; "
                "disable thinking with --llm-no-thinking)"
            )
        finish = choice.get("finish_reason")
        if finish:
            detail += f" (finish reason: {finish})"
        raise LlmResponseError(
            "the LLM returned an empty message; ask it to answer with JSON orders" + detail
        )
    return text


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


# ---------------------------------------------------------------------------
# prompt and single agent
# ---------------------------------------------------------------------------


def _system_prompt(
    player: str,
    rules_summary: str,
    *,
    role: str | None = None,
    brief: str = "",
) -> str:
    if role:
        opening = (
            f"You are one of the generals commanding player {player} in a deterministic hex wargame.\n"
            f"YOUR COMMAND:\n{role.strip()}\n"
        )
        if brief.strip():
            opening += f"SUPREME COMMANDER'S INTENT:\n{brief.strip()}\n"
        opening += (
            "You command only the units listed in unit_actions; other generals command the rest.\n"
            "Your orders are merged with theirs, and only orders for your own units are kept.\n"
            "Only the general whose view lists build_options may build; if yours has none,\n"
            "do not propose a build.\n"
        )
    else:
        opening = f"You are the commander of player {player} in a deterministic hex wargame.\n"
    return (
        opening
        + "Your goal is to win: capture the enemy HQ (the objective) or eliminate all enemy units.\n"
        "You will receive the complete game view as JSON. Plan the whole turn, then reply with\n"
        "exactly one JSON object and nothing else - no markdown and no prose outside the JSON.\n"
        "\n"
        "Response schema:\n"
        '{"orders":[{"action":"move","unit":"A2","to":[q,r]},'
        '{"action":"attack","unit":"A3","target":"B2","weapon":"optional-weapon-id"}],'
        '"commentary":"one short sentence"}\n'
        "\n"
        "Order shapes (use exact ids and coordinates from the view):\n"
        '- move:     {"action":"move","unit":"<id>","to":[q,r]} where [q,r] is a destination\n'
        "            listed in unit_actions[id].move_to; the coordinator expands the path.\n"
        '- attack:   {"action":"attack","unit":"<id>","target":"<enemy id>"} - add "weapon"\n'
        "            only to choose a specific weapon listed for that target.\n"
        '- capture:  {"action":"capture","unit":"<id>"} for a unit standing on a capturable site.\n'
        '- load:     {"action":"load","unit":"<transport id>","cargo":"<adjacent friendly id>"}.\n'
        '- unload:   {"action":"unload","unit":"<transport id>","cargo":"<id>","target":[q,r]}.\n'
        '- build:    {"action":"build","site":[q,r],"unit":"<unit type id>"} using build_options.\n'
        '- hold:     {"action":"hold","unit":"<id>"}.\n'
        "\n"
        "Rules for a legal turn:\n"
        "- Only order units in unit_actions whose entry is not acted/loaded/new.\n"
        "- Each unit gets exactly one order this turn. The attack list is only valid\n"
        "  from the unit's current position, so do not move and attack with the same\n"
        "  unit. Choose one: move OR attack OR capture OR load OR unload OR hold.\n"
        "- For an attack, copy a target/weapon pair from unit_actions[id].attack.  Any\n"
        "  other target or weapon is rejected; never list two attacks for one unit.\n"
        "- Move only to destinations the view lists; never invent a path or a coordinate.\n"
        "- Build only from build_options, only affordable entries, and only if it helps\n"
        "  you win.  Do not build at a site occupied by any unit on the map.\n"
        "- Do not output end_turn; the coordinator ends your turn after applying your orders.\n"
        "- If a unit has no useful action, omit it or hold it. Empty orders are valid.\n"
        "\n"
        "Tactics: focus fire to destroy units; capture cities/factories for income; keep indirect\n"
        "weapons stationary; protect your HQ; use terrain cover; do not leave your own HQ exposed.\n"
        "\n"
        f"ENGINE RULES SUMMARY:\n{rules_summary}\n"
    )


def _compact_view(
    view: dict[str, Any], allowed_units: list[str] | None = None
) -> dict[str, Any]:
    """Trim the engine's full view to the fields a planner needs.

    The engine view is intentionally complete for debugging and external agents;
    a local model with an 8k context pays for every duplicate map layer and every
    empty field.  This keeps the same facts in a smaller prompt without changing
    what the view API serves.  ``allowed_units`` narrows ``unit_actions`` to a
    council member's own slice; the rest of the board stays visible.
    """
    compact: dict[str, Any] = {}
    for key in (
        "scenario",
        "seed",
        "turn",
        "status",
        "current_player",
        "map_size",
        "map_terrain_rows",
        "map_ownership_rows",
        "map_unit_rows",
        "coordinates",
        "players",
    ):
        if key in view:
            compact[key] = view[key]
    if view.get("winner"):
        compact["winner"] = view["winner"]
    if view.get("unit_stacks"):
        compact["unit_stacks"] = view["unit_stacks"]
        for key in ("map_ground_unit_rows", "map_air_unit_rows"):
            if key in view:
                compact[key] = view[key]

    compact["sites"] = [
        {
            key: site[key]
            for key in ("pos", "terrain", "owner", "build", "capture_needed", "capture_progress", "objective")
            if key in site and site[key] not in (None, 0, False)
        }
        for site in view.get("sites", [])
    ]

    compact["units"] = []
    for unit in view.get("units", []):
        entry = {key: unit[key] for key in ("id", "owner", "type", "name", "pos", "hp", "max_hp") if key in unit}
        for key in ("ammo", "fuel", "cargo"):
            if unit.get(key) not in (None, {}, []):
                entry[key] = unit[key]
        for key in ("loaded", "moved", "acted", "new", "can_rearm"):
            if unit.get(key):
                entry[key] = True
        compact["units"].append(entry)

    if "build_options" in view:
        compact["build_options"] = [
            {key: option[key] for key in ("site", "unit", "cost", "affordable") if key in option}
            for option in view["build_options"]
        ]
    if "unit_actions" in view:
        actions = view["unit_actions"]
        if allowed_units is not None:
            allowed = set(allowed_units)
            actions = {unit_id: entry for unit_id, entry in actions.items() if unit_id in allowed}
        compact["unit_actions"] = actions
    return compact


def build_messages(request: TurnRequest) -> list[dict[str, str]]:
    """The two chat messages for one whole-turn decision."""
    view = _compact_view(request.view, request.allowed_units)
    user_payload: dict[str, Any] = {
        "you_are": request.player,
        "turn": view.get("turn"),
        "status": view.get("status"),
        "current_player": view.get("current_player"),
        "recent_events": request.recent_events[-10:],
        "view": view,
    }
    if request.allowed_units is not None:
        user_payload["your_units"] = list(request.allowed_units)
    if request.brief:
        user_payload["brief"] = request.brief
    if request.feedback:
        user_payload["previous_rejections"] = request.feedback
        user_payload["instruction"] = (
            "Your previous orders were rejected or unusable. Return a corrected plan for "
            "the units that can still act; use only legal ids, destinations and builds."
        )
    system = _system_prompt(
        request.player,
        str(request.view.get("rules", "")),
        role=request.role,
        brief=request.brief,
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":")),
        },
    ]


class SingleLlmAgent:
    """One local LLM planning one player's whole turn, or one general's slice."""

    def __init__(
        self,
        client: LlmClient,
        *,
        logger: JsonlLogger | None = None,
        history: int = 8,
        name: str = "llm",
    ):
        self.client = client
        self.logger = logger
        self.history = history
        self.name = name

    def plan(self, request: TurnRequest) -> AgentPlan:
        messages = build_messages(request)
        if self.logger is not None:
            self.logger.write(
                {
                    "kind": "llm_request",
                    "agent": self.name,
                    "model": self.client.config.model,
                    "endpoint": self.client.config.endpoint,
                    "player": request.player,
                    "turn": request.view.get("turn"),
                    "attempt_feedback": request.feedback,
                    "messages": messages,
                }
            )
        try:
            raw = self.client.complete(messages)
            plan = parse_plan(raw)
        except (PlanFormatError, LlmResponseError) as exc:
            if self.logger is not None:
                self.logger.write(
                    {
                        "kind": "llm_error",
                        "agent": self.name,
                        "model": self.client.config.model,
                        "player": request.player,
                        "turn": request.view.get("turn"),
                        "error": str(exc),
                    }
                )
            raise
        if self.logger is not None:
            self.logger.write(
                {
                    "kind": "llm_response",
                    "agent": self.name,
                    "model": self.client.config.model,
                    "player": request.player,
                    "turn": request.view.get("turn"),
                    "raw": raw,
                    "orders": plan.orders,
                    "commentary": plan.commentary,
                }
            )
        return plan
