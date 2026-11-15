"""Project folder layout, save/load and schema migrations.

    <project dir>/
        project.json          the document (versioned)
        cache/thumbs/         derived thumbnails
        cache/proxies/        low-res preview copies of videos
        logs/trajectory.jsonl command log (training data)
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .project import SCHEMA_VERSION, Project, ProjectKind

PROJECT_FILE = "project.json"

# version -> function upgrading a raw dict from that version to version + 1
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    # v1 -> v2: layer fields, track flags and markers were added; all have defaults,
    # so old documents are valid as they are.
    1: lambda raw: raw,
}


class ProjectPaths:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def thumbs(self) -> Path:
        return self.root / "cache" / "thumbs"

    @property
    def proxies(self) -> Path:
        return self.root / "cache" / "proxies"

    @property
    def trajectory_log(self) -> Path:
        return self.root / "logs" / "trajectory.jsonl"

    def ensure(self) -> None:
        for d in (self.thumbs, self.proxies, self.trajectory_log.parent):
            d.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        """Resolve a project-relative path, refusing anything outside the project."""
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ValueError(f"path escapes project: {relative}")
        return candidate


def create_project(root: Path, name: str, kind: ProjectKind = ProjectKind.VIDEO) -> Project:
    paths = ProjectPaths(root)
    if paths.project_file.exists():
        raise FileExistsError(f"project already exists: {paths.project_file}")
    paths.ensure()
    project = Project(name=name, kind=kind)
    save_project(root, project)
    return project


def save_project(root: Path, project: Project) -> None:
    """Atomic write: never leaves a half-written project.json behind."""
    paths = ProjectPaths(root)
    paths.ensure()
    tmp = paths.project_file.with_suffix(".json.tmp")
    tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, paths.project_file)


def load_project(root: Path) -> Project:
    paths = ProjectPaths(root)
    raw = json.loads(paths.project_file.read_text(encoding="utf-8"))
    return Project.model_validate(migrate(raw))


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    version = int(raw.get("schema_version", 1))
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"project schema v{version} is newer than this app supports (v{SCHEMA_VERSION}); update the app"
        )
    while version < SCHEMA_VERSION:
        raw = MIGRATIONS[version](raw)
        version += 1
        raw["schema_version"] = version
    return raw
