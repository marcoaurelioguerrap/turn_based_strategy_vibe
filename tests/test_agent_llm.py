"""The stdlib OpenAI-compatible client and the single-agent prompt/response path."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tbs.agents.llm import (
    JsonlLogger,
    LlmClient,
    LlmConfig,
    LlmResponseError,
    LlmTransportError,
    SingleLlmAgent,
    build_messages,
)
from tbs.agents.protocol import TurnRequest


def completion(content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode("utf-8")


def request(feedback: list[str] | None = None) -> TurnRequest:
    return TurnRequest(
        player="A",
        view={
            "turn": 3,
            "status": "ongoing",
            "current_player": "A",
            "rules": "RULES",
            "unit_actions": {"A1": {"hold": True}},
        },
        recent_events=["A1 attacked B1"],
        feedback=feedback or [],
    )


def test_the_client_posts_an_openai_chat_completion_request():
    seen: dict = {}

    def transport(url, body, headers, timeout):
        seen.update(url=url, body=json.loads(body), headers=headers, timeout=timeout)
        return completion('{"orders":[]}')

    config = LlmConfig(
        model="local-model",
        base_url="http://127.0.0.1:8888/v1",
        api_key="secret",
        temperature=0.1,
        max_tokens=777,
        enable_thinking=False,
        reasoning_effort="low",
    )
    text = LlmClient(config, transport=transport).complete([{"role": "user", "content": "hi"}])

    assert text == '{"orders":[]}'
    assert seen["url"] == "http://127.0.0.1:8888/v1/chat/completions"
    assert seen["body"]["model"] == "local-model"
    assert seen["body"]["temperature"] == 0.1
    assert seen["body"]["max_tokens"] == 777
    assert seen["body"]["stream"] is False
    assert seen["body"]["enable_thinking"] is False
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["timeout"] == 180.0


def test_content_parts_are_joined():
    def parts(*_args):
        content = [
            {"type": "text", "text": '{"orders":'},
            {"type": "text", "text": "[]}"},
        ]
        return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    client = LlmClient(LlmConfig(model="m"), transport=parts)
    assert client.complete([]) == '{"orders":[]}'


def test_a_client_error_is_not_retried():
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 400, "bad model", {}, io.BytesIO(b'{"error":"no such model"}'))

    sleeps = []
    client = LlmClient(
        LlmConfig(model="m", retries=2), transport=transport, sleep=sleeps.append
    )
    with pytest.raises(LlmTransportError, match="HTTP 400.*no such model"):
        client.complete([])
    assert len(calls) == 1
    assert sleeps == []


def test_server_errors_are_retried_with_backoff_then_raise():
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 503, "busy", {}, io.BytesIO(b"try later"))

    sleeps = []
    client = LlmClient(
        LlmConfig(model="m", retries=2), transport=transport, sleep=sleeps.append
    )
    with pytest.raises(LlmTransportError, match="HTTP 503"):
        client.complete([])
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


def test_unreachable_server_is_retried_then_reported():
    calls = []

    def transport(*_args):
        calls.append(1)
        raise urllib.error.URLError("connection refused")

    client = LlmClient(LlmConfig(model="m", retries=1), transport=transport, sleep=lambda _s: None)
    with pytest.raises(LlmTransportError, match="cannot reach"):
        client.complete([])
    assert len(calls) == 2


def test_missing_or_empty_content_is_an_output_error():
    def no_content(*_args):
        return json.dumps({"choices": [{"message": {}}]}).encode()

    def empty(*_args):
        return completion("   ")

    with pytest.raises(LlmResponseError, match="no choices"):
        LlmClient(LlmConfig(model="m"), transport=no_content).complete([])
    with pytest.raises(LlmResponseError, match="empty message"):
        LlmClient(LlmConfig(model="m"), transport=empty).complete([])


def test_a_missing_model_is_reported_before_any_network_call():
    client = LlmClient(LlmConfig(model=""), transport=lambda *_: b"")
    with pytest.raises(LlmTransportError, match="no model configured"):
        client.complete([])


def test_the_prompt_carries_the_view_rules_and_previous_rejections():
    messages = build_messages(request(["no legal move for A1 to [9,9]"]))
    assert messages[0]["role"] == "system"
    assert "player A" in messages[0]["content"]
    assert "RULES" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["view"]["turn"] == 3
    assert payload["previous_rejections"] == ["no legal move for A1 to [9,9]"]
    assert "corrected plan" in payload["instruction"]


def test_the_prompt_compacts_duplicate_layers_and_empty_fields():
    view = {
        "turn": 1,
        "status": "ongoing",
        "current_player": "A",
        "rules": "long duplicated rules",
        "coordinates": "axial",
        "map_terrain_rows": ["..."],
        "map_ownership_rows": ["..."],
        "map_unit_rows": ["..."],
        "map_ground_unit_rows": ["..."],
        "map_air_unit_rows": ["..."],
        "unit_stacks": [],
        "players": [],
        "sites": [
            {"pos": [0, 0], "terrain": "C", "name": "City", "owner": None, "build": None,
             "capture_progress": 0, "capture_needed": 300, "objective": False},
        ],
        "units": [
            {"id": "A1", "owner": "A", "type": "infantry", "name": "Infantry", "pos": [0, 0],
             "hp": 100, "max_hp": 100, "move": 3, "ammo": {}, "fuel": None, "can_rearm": False,
             "cargo": [], "loaded": False, "moved": False, "acted": False, "new": False},
        ],
        "build_options": [
            {"site": [0, 0], "site_type": "factory", "unit": "infantry", "cost": 1000, "affordable": True},
        ],
        "unit_actions": {"A1": {"hold": True}},
    }
    messages = build_messages(TurnRequest(player="A", view=view))
    payload = json.loads(messages[1]["content"])
    compact = payload["view"]

    assert "rules" not in compact, "the rules live in the system prompt instead"
    assert "map_ground_unit_rows" not in compact, "no stack means no duplicate layer"
    assert compact["sites"] == [
        {"pos": [0, 0], "terrain": "C", "capture_needed": 300}
    ], "empty site fields are dropped"
    assert compact["units"][0] == {
        "id": "A1", "owner": "A", "type": "infantry", "name": "Infantry",
        "pos": [0, 0], "hp": 100, "max_hp": 100,
    }, "inactive flags and empty ammo/cargo are dropped"
    assert compact["build_options"] == [
        {"site": [0, 0], "unit": "infantry", "cost": 1000, "affordable": True}
    ]
    assert "ENGINE RULES SUMMARY" in messages[0]["content"]
    assert "long duplicated rules" in messages[0]["content"]


def test_single_agent_returns_a_parsed_plan_and_logs_both_sides(tmp_path):
    raw = '{"orders":[{"action":"hold","unit":"A1"}],"commentary":"wait"}'

    def transport(*_args):
        return completion(raw)

    log_path = tmp_path / "calls.jsonl"
    logger = JsonlLogger(log_path)
    agent = SingleLlmAgent(LlmClient(LlmConfig(model="m"), transport=transport), logger=logger)
    plan = agent.plan(request())

    assert plan.orders == [{"action": "hold", "unit": "A1"}]
    assert plan.commentary == "wait"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["kind"] for record in records] == ["llm_request", "llm_response"]
    assert records[0]["messages"][0]["role"] == "system"
    assert records[1]["raw"] == raw


def test_a_real_http_round_trip_against_an_openai_shaped_server():
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps(
                {"choices": [{"message": {"content": '{"orders":[],"commentary":"ok"}'}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        client = LlmClient(LlmConfig(model="real-model", base_url=f"http://127.0.0.1:{port}/v1"))
        text = client.complete([{"role": "user", "content": "hello"}])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert text == '{"orders":[],"commentary":"ok"}'
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["model"] == "real-model"
