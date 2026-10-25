"""Scenario discovery metadata and a tiny map preview.  Pure: no pygame.

The engine only needs a scenario's rules data; a picker also wants a human
title, a one-line summary and a thumbnail of the text rows.  Keeping that work
here means the menu can be tested headlessly and ``render`` stays drawing-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tbs.engine.data import load_scenario

#: Terrain codes a preview may contain fall back to the palette's default.
PreviewRows = tuple[str, ...]


@dataclass(frozen=True)
class ScenarioInfo:
    path: Path
    title: str
    description: str
    difficulty: str
    fair: bool
    rows: PreviewRows
    width: int
    height: int
    max_turns: int
    players: int
    initial_income: int

    @property
    def summary(self) -> str:
        """One short line for the menu hint."""
        return f"{self.width}x{self.height} - {self.max_turns} turns - {self.difficulty}"

    @property
    def detail(self) -> str:
        sides = "2 players" if self.players == 2 else f"{self.players} players"
        return f"{self.description} ({sides}, {'fair' if self.fair else 'asymmetric'})"


def _fallback(path: Path, reason: str) -> ScenarioInfo:
    return ScenarioInfo(
        path=path,
        title=path.stem.replace("_", " ").title(),
        description=f"unreadable scenario: {reason}",
        difficulty="unknown",
        fair=False,
        rows=(),
        width=0,
        height=0,
        max_turns=0,
        players=0,
        initial_income=0,
    )


def describe_scenario(path: Path | str) -> ScenarioInfo:
    """Load the display metadata of one scenario, tolerating a broken file.

    A malformed map must not crash the picker: its row is shown disabled-looking
    with the parse error as the description, and choosing it surfaces the
    engine's own message when the game is built.
    """
    path = Path(path)
    try:
        data: dict[str, Any] = load_scenario(path)
        rows = tuple(str(row) for row in data.get("rows") or ())
        width = len(rows[0]) if rows else 0
        return ScenarioInfo(
            path=path,
            title=str(data.get("title") or path.stem.replace("_", " ").title()),
            description=str(data.get("description") or data.get("comment") or ""),
            difficulty=str(data.get("difficulty") or "standard"),
            fair=bool(data.get("fair", False)),
            rows=rows,
            width=width,
            height=len(rows),
            max_turns=int(data.get("max_turns", 200)),
            players=len(data.get("players") or ()),
            initial_income=int(data.get("initial_income", 1000)),
        )
    except (OSError, ValueError, KeyError, TypeError) as problem:
        return _fallback(path, str(problem))


def scenario_infos(paths: Iterable[Path | str]) -> list[ScenarioInfo]:
    """Describe every scenario, in the order given (``available_scenarios``)."""
    return [describe_scenario(path) for path in paths]


def preview_grid(
    rows: Iterable[str],
    *,
    max_width: int = 20,
    max_height: int = 12,
) -> PreviewRows:
    """A deterministic, nearest-neighbour thumbnail of text map rows.

    Small maps come back untouched; a larger one is sampled to at most
    ``max_height`` rows and ``max_width`` columns so the preview always fits.
    """
    source = [str(row) for row in rows if row]
    if not source:
        return ()
    cols = max(len(row) for row in source)
    out_height = min(len(source), max_height)
    out_width = min(cols, max_width)
    if out_height == len(source) and out_width == cols:
        return tuple(source)
    sampled: list[str] = []
    for row_index in range(out_height):
        source_row = source[row_index * len(source) // out_height]
        sampled.append(
            "".join(
                source_row[col_index * cols // out_width]
                if col_index * cols // out_width < len(source_row)
                else "."
                for col_index in range(out_width)
            )
        )
    return tuple(sampled)
