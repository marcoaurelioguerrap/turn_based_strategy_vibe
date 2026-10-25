"""The headless match CLI: an autonomous game with a replayable order log."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tbs.agents import match
from tbs.agents.greedy import GreedyAgent
from tbs.agents.turn import run_turn
from tbs.engine import rules
from tbs.engine.serialize import replay, state_hash
from tbs.engine.state import new_game

from .helpers import SPECS, bundled_scenario


def test_the_greedy_agent_plays_from_the_view_alone_and_replays_exactly():
    state = new_game(bundled_scenario(), SPECS)
    rules.start_game(state)
    orders = []
    agent = GreedyAgent()

    while state.status == "ongoing" and state.turn_number <= 4:
        orders.extend(run_turn(state, agent).orders)

    assert orders
    rebuilt = replay(bundled_scenario(), json.loads(json.dumps(orders)))
    assert state_hash(rebuilt) == state_hash(state)


def test_the_match_cli_writes_a_save_and_an_order_log(tmp_path, capsys):
    save = tmp_path / "match.json"
    order_log = tmp_path / "match.orders.json"

    code = match.main(
        [
            "--a",
            "greedy",
            "--b",
            "greedy",
            "--max-turns",
            "3",
            "--out",
            str(save),
            "--orders",
            str(order_log),
        ]
    )

    assert code == 1, "the three-turn cap leaves a healthy game ongoing"
    assert save.exists()
    payload = json.loads(order_log.read_text(encoding="utf-8"))
    assert payload and payload[-1]["action"] == "end_turn"
    output = capsys.readouterr().out
    assert "replay MATCHES" in output


def test_the_match_cli_drives_a_council_side_and_replays(tmp_path, capsys):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            requests.append(body)
            system = body["messages"][0]["content"]
            if "supreme commander" in system:
                content = "Hold the line and build infantry."
            else:
                content = '{"orders":[],"commentary":"holding"}'
            payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
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
        code = match.main(
            [
                "--a",
                "greedy",
                "--b",
                "council",
                "--council-members",
                "vanguard,anvil,quartermaster",
                "--llm-model",
                "fake",
                "--llm-base-url",
                f"http://127.0.0.1:{server.server_address[1]}/v1",
                "--max-turns",
                "1",
                "--out",
                str(tmp_path / "match.json"),
                "--orders",
                str(tmp_path / "match.orders.json"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert code == 1, "the one-turn cap ends an ongoing match"
    assert len(requests) >= 4, "the commander and three generals are consulted"
    assert any("supreme commander" in body["messages"][0]["content"] for body in requests)
    assert any("one of the generals" in body["messages"][0]["content"] for body in requests)
    output = capsys.readouterr().out
    assert "council=" in output
    assert "replay MATCHES" in output


def test_the_match_cli_drives_a_real_openai_compatible_llm_side(tmp_path, capsys):
    seen: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers["Content-Length"])
            seen["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps(
                {"choices": [{"message": {"content": '{"orders":[],"commentary":"pass"}'}}]}
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
        code = match.main(
            [
                "--a",
                "greedy",
                "--b",
                "llm",
                "--llm-model",
                "fake",
                "--llm-base-url",
                f"http://127.0.0.1:{server.server_address[1]}/v1",
                "--max-turns",
                "1",
                "--out",
                str(tmp_path / "match.json"),
                "--orders",
                str(tmp_path / "match.orders.json"),
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert code == 1, "the one-turn cap ends an ongoing match"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["model"] == "fake"
    assert "replay MATCHES" in capsys.readouterr().out
