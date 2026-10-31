"""A deterministic scripted agent, useful as a baseline and for offline matches.

It sees only the same view an LLM sees, so it is a fair smoke test of the agent
protocol: it first executes the engine's ferry hints (a transport pickup, sail
or landing), then attacks, captures underfoot, and walks toward the nearest
enemy or uncaptured site.  It never builds and never invents coordinates.
"""

from __future__ import annotations

from typing import Any

from tbs.agents.ferry import plan_ferry
from tbs.agents.protocol import AgentPlan, TurnRequest
from tbs.engine.hexgrid import hex_distance

__all__ = ["GreedyAgent"]


class GreedyAgent:
    """The dumbest legal whole-turn policy: fight, capture, then approach."""

    name = "greedy"

    def plan(self, request: TurnRequest) -> AgentPlan:
        current = request.player
        view = request.view
        own = {
            unit["id"]: unit
            for unit in view.get("units", [])
            if unit.get("owner") == current and unit.get("pos") is not None
        }
        targets: list[tuple[int, int]] = [
            tuple(unit["pos"])
            for unit in view.get("units", [])
            if unit.get("owner") != current and unit.get("pos") is not None
        ]
        for site in view.get("sites", []):
            if site.get("owner") != current and site.get("capture_needed") is not None:
                targets.append(tuple(site["pos"]))

        ferry = plan_ferry(view)
        orders: list[dict[str, Any]] = list(ferry.orders)
        attacked: set[str] = set()
        used_destinations: set[tuple[int, int]] = set(ferry.destinations)
        claimed: set[str] = set(ferry.claimed)

        for unit_id in sorted(view.get("unit_actions", {})):
            if unit_id in claimed:
                continue
            summary = view["unit_actions"][unit_id]
            if not isinstance(summary, dict):
                continue
            if summary.get("acted") or summary.get("loaded_in") or summary.get("new_unit"):
                continue

            attacks = [
                row
                for row in summary.get("attack", [])
                if isinstance(row, list) and len(row) >= 4 and row[0] not in attacked
            ]
            if attacks:
                target, weapon, _distance, preview = max(
                    attacks, key=lambda row: (row[3], row[0], row[1])
                )
                orders.append(
                    {"action": "attack", "unit": unit_id, "target": target, "weapon": weapon}
                )
                attacked.add(str(target))
                continue

            if summary.get("capture_remaining_hp") is not None:
                orders.append({"action": "capture", "unit": unit_id})
                continue

            moves = [row for row in summary.get("move_to", []) if isinstance(row, list) and len(row) >= 2]
            position = own.get(unit_id, {}).get("pos")
            if not moves or not targets or position is None:
                continue
            current_distance = min(hex_distance(tuple(position), spot) for spot in targets)

            def score(row: list[Any]) -> tuple:
                destination = tuple(row[0])
                return (
                    min(hex_distance(destination, spot) for spot in targets),
                    row[1],
                    destination,
                )

            best = min(moves, key=score)
            destination = tuple(best[0])
            if destination in used_destinations:
                continue
            reached = min(hex_distance(destination, spot) for spot in targets)
            if reached < current_distance:
                orders.append({"action": "move", "unit": unit_id, "to": list(destination)})
                used_destinations.add(destination)

        return AgentPlan(orders=orders, commentary="scripted greedy plan")
