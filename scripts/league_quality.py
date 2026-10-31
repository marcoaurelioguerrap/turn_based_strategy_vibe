"""Aggregate NCM order quality from saved match sidecars.

The all-AI league writes one ``*.meta.json`` per game.  Each sidecar carries
``agent_stats`` for the NCM side(s): exact generations, repairs, constrained
decodes, tactical reranks and fallbacks.  This script turns a batch of games
into the table a report or blog post needs.

    py scripts/league_quality.py --batch all_ai_league_20261030_v2
    py scripts/league_quality.py --match-dir ..\\NCM_fork\\artifacts\\tbs\\matches `
      --batch all_ai_league_20261030_v2 --out docs/all_ai_league/quality.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = ["aggregate"]

COUNTERS = (
    "steps",
    "samples",
    "generated",
    "repaired",
    "constrained",
    "tactical",
    "fallbacks",
    "illegal",
    "rejected",
    "model_orders",
)


def aggregate(paths: list[Path]) -> dict[str, dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        labels = meta.get("team_labels") or {}
        for side, stats in (meta.get("agent_stats") or {}).items():
            label = str(labels.get(side) or side)
            bucket = teams.setdefault(
                label,
                {"games": 0, **{name: 0 for name in COUNTERS}},
            )
            bucket["games"] += 1
            for name in COUNTERS:
                bucket[name] += int(stats.get(name) or 0)
    for bucket in teams.values():
        steps = max(1, int(bucket["steps"]))
        bucket["constrained_share"] = round(bucket["constrained"] / steps, 4)
        bucket["generated_share"] = round(bucket["generated"] / steps, 4)
        bucket["fallback_rate"] = round(bucket["fallbacks"] / steps, 4)
        bucket["model_share"] = round(bucket["model_orders"] / steps, 4)
    return teams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-dir",
        default="../NCM_fork/artifacts/tbs/matches",
        help="directory with *.meta.json sidecars",
    )
    parser.add_argument("--batch", default="", help="substring of the match filename")
    parser.add_argument("--out", default="", help="optional JSON output path")
    args = parser.parse_args(argv)

    paths = [
        Path(match)
        for match in glob.glob(str(Path(args.match_dir) / "*.meta.json"))
        if args.batch in Path(match).name
    ]
    if not paths:
        parser.error("no match sidecars matched")
    teams = aggregate(paths)
    payload = {"games": len(paths), "batch": args.batch, "teams": teams}
    text = json.dumps(payload, indent=2)
    print(text)
    print()
    print("| Team | Games | Steps | Generated | Repaired | Constrained | Fallbacks | Fallback rate |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, row in sorted(teams.items()):
        print(
            f"| {label} | {row['games']} | {row['steps']} | {row['generated']} | "
            f"{row['repaired']} | {row['constrained']} | {row['fallbacks']} | "
            f"{row['fallback_rate']:.1%} |"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
