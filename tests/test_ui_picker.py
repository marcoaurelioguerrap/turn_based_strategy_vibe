"""Scenario picker metadata and preview sampling, all headless."""

from __future__ import annotations

from pathlib import Path

from tbs.engine.data import available_scenarios, default_scenario
from tbs.ui import picker


def test_every_bundled_map_has_a_title_and_a_summary():
    infos = picker.scenario_infos(available_scenarios())
    assert len(infos) >= 4
    titles = [info.title for info in infos]
    assert "Border Skirmish" in titles
    for info in infos:
        assert info.title
        assert info.description
        assert info.width > 0 and info.height > 0
        assert "turns" in info.summary
        assert info.path.exists()


def test_the_default_scenario_describes_itself():
    info = picker.describe_scenario(default_scenario())
    assert info.title == "Border Skirmish"
    assert info.width == 14 and info.height == 8
    assert info.fair is True
    assert info.players == 2
    assert info.difficulty == "intro"
    assert "intro" in info.summary


def test_preview_grid_keeps_a_small_map_untouched():
    rows = (".....", "..C..", ".....")
    assert picker.preview_grid(rows) == rows


def test_preview_grid_samples_an_oversized_map_deterministically():
    rows = tuple("".join("." if (col + row) % 2 else "C" for col in range(40)) for row in range(30))
    preview = picker.preview_grid(rows, max_width=20, max_height=12)
    assert len(preview) == 12
    assert all(len(row) == 20 for row in preview)
    assert preview == picker.preview_grid(rows, max_width=20, max_height=12)


def test_a_broken_or_missing_scenario_does_not_crash_the_picker(tmp_path: Path):
    missing = tmp_path / "gone.json"
    info = picker.describe_scenario(missing)
    assert info.title == "Gone"
    assert "unreadable scenario" in info.description
    assert info.rows == ()

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    info = picker.describe_scenario(broken)
    assert info.title == "Broken"
    assert "unreadable scenario" in info.description
