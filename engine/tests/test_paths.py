"""Rename to Framewise: old AIEditor data moves to the new folder; env vars accept both spellings."""

from __future__ import annotations

from pathlib import Path

import pytest

from aieditor_engine.paths import env, migrate_legacy


def test_migrate_legacy(tmp_path: Path) -> None:
    old, new = tmp_path / "AIEditor", tmp_path / "Framewise"
    (old / "models").mkdir(parents=True)
    (old / "models" / "m.gguf").write_text("model")
    (old / "licence.json").write_text('{"key": "x"}')
    (old / "aieditor-engine.exe").write_text("program file")  # program files stay
    new.mkdir()
    (new / "app.json").write_text('{"onboarded": true}')
    (old / "app.json").write_text('{"onboarded": false}')  # never overwrites newer data

    moved = migrate_legacy(new, old)
    assert sorted(moved) == ["licence.json", "models"]
    assert (new / "models" / "m.gguf").read_text() == "model"
    assert (new / "licence.json").exists() and not (old / "licence.json").exists()
    assert (old / "aieditor-engine.exe").exists()
    assert (new / "app.json").read_text() == '{"onboarded": true}'
    assert migrate_legacy(new, old) == []  # idempotent


def test_env_both_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRAMEWISE_HOME", raising=False)
    monkeypatch.setenv("AIEDITOR_HOME", "/old")
    assert env("HOME") == "/old"
    monkeypatch.setenv("FRAMEWISE_HOME", "/new")
    assert env("HOME") == "/new"
