from __future__ import annotations

import json
from pathlib import Path

import pytest

from aieditor_engine.core.project import SCHEMA_VERSION, ProjectKind
from aieditor_engine.core.store import (
    ProjectPaths,
    create_project,
    load_project,
    migrate,
    save_project,
)


def test_create_save_load_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    project = create_project(root, "Trip", ProjectKind.PHOTO)
    project.name = "Trip 2"
    save_project(root, project)
    loaded = load_project(root)
    assert loaded == project
    assert (root / "cache" / "proxies").is_dir()
    assert not list(root.glob("*.tmp"))


def test_create_refuses_existing(tmp_path: Path) -> None:
    create_project(tmp_path, "A")
    with pytest.raises(FileExistsError):
        create_project(tmp_path, "B")


def test_newer_schema_rejected() -> None:
    with pytest.raises(ValueError, match="newer"):
        migrate({"schema_version": SCHEMA_VERSION + 1, "name": "x"})


def test_resolve_blocks_escape(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path / "proj")
    assert paths.resolve("cache/thumbs/a.jpg").name == "a.jpg"
    with pytest.raises(ValueError):
        paths.resolve("../../secret.txt")


def test_project_file_is_plain_json(tmp_path: Path) -> None:
    create_project(tmp_path, "A")
    data = json.loads((tmp_path / "project.json").read_text())
    assert data["schema_version"] == SCHEMA_VERSION
