"""Summarize per-turn latency from the game's JSONL LLM logs.

Each ``llm_request``/``llm_response`` pair is one completion; this reports the
p50/p95 wall-clock seconds per turn for one or more logs.  It makes no network
calls, so a saved match is enough to compare an LLM baseline with the NCM
benchmark.

    py scripts/llm_latency.py --log "logs/llm/qwen*.jsonl"
    py scripts/llm_latency.py --log "logs/llm/*.jsonl" --out docs/llm_latency.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["parse_log", "summarize"]


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"turns": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "turns": len(values),
        "mean_seconds": round(statistics.mean(values), 3),
        "p50_seconds": round(statistics.median(values), 3),
        "p95_seconds": round(percentile(0.95), 3),
        "min_seconds": round(min(values), 3),
        "max_seconds": round(max(values), 3),
    }


def _timestamp(row: dict[str, Any]) -> datetime | None:
    value = row.get("time")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def parse_log(path: Path | str) -> dict[str, Any]:
    """Return the request/response latencies and metadata for one log."""
    requests: dict[tuple[Any, Any], datetime] = {}
    durations: list[float] = []
    model = ""
    endpoint = ""
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip("\x00 \t\r\n")
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        model = model or str(row.get("model") or "")
        endpoint = endpoint or str(row.get("endpoint") or "")
        kind = row.get("kind")
        key = (row.get("turn"), row.get("player"))
        if kind == "llm_request":
            stamp = _timestamp(row)
            if stamp is not None:
                requests[key] = stamp
        elif kind == "llm_response" and key in requests:
            stamp = _timestamp(row)
            if stamp is not None:
                durations.append((stamp - requests.pop(key)).total_seconds())
    return {
        "path": str(Path(path)),
        "model": model,
        "endpoint": endpoint,
        **summarize(durations),
        "_durations": durations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        help="log path or glob; repeat the flag for several patterns",
    )
    parser.add_argument("--out", default="", help="optional JSON output path")
    args = parser.parse_args(argv)

    patterns = args.log or ["logs/llm/*.jsonl"]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in glob.glob(pattern))
    paths = sorted({path for path in paths if path.is_file()})
    if not paths:
        parser.error(f"no LLM logs matched: {', '.join(patterns)}")

    files = []
    durations: list[float] = []
    for path in paths:
        item = parse_log(path)
        durations.extend(item.pop("_durations", []))
        files.append(item)
    payload = {"files": files, "combined": summarize(durations)}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
