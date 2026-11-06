from __future__ import annotations

import json
from pathlib import Path

import pytest

from aieditor_engine.core import builtin  # noqa: F401
from aieditor_engine.core.bus import CommandBus
from aieditor_engine.core.commands import CommandError, list_commands
from aieditor_engine.core.project import Project


def make_bus(tmp_path: Path) -> CommandBus:
    return CommandBus(Project(name="t"), log_path=tmp_path / "log.jsonl")


def add_video(bus: CommandBus, duration: float = 10.0) -> str:
    return bus.execute(
        "media.add", {"name": "a.mp4", "path": "/x/a.mp4", "kind": "video", "duration": duration}
    )["media_id"]


def test_execute_undo_redo(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    bus.execute("project.rename", {"name": "Holiday"})
    assert bus.project.name == "Holiday"
    assert bus.undo()
    assert bus.project.name == "t"
    assert bus.redo()
    assert bus.project.name == "Holiday"
    assert not bus.redo()


def test_new_command_clears_redo(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    bus.execute("project.rename", {"name": "A"})
    bus.undo()
    bus.execute("project.rename", {"name": "B"})
    assert not bus.redo()
    assert bus.project.name == "B"


def test_invalid_params_rejected_and_state_untouched(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    rev = bus.revision
    with pytest.raises(CommandError, match="invalid params"):
        bus.execute("project.rename", {"name": ""})
    with pytest.raises(CommandError, match="unknown command"):
        bus.execute("does.not_exist", {})
    assert bus.revision == rev
    assert bus.project.name == "t"


def test_failed_command_is_atomic(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    media = add_video(bus, duration=4.0)
    track = bus.execute("track.add", {})["track_id"]
    before = bus.project.model_dump()
    with pytest.raises(CommandError, match="beyond media duration"):
        bus.execute("clip.add", {"track_id": track, "media_id": media, "out_point": 9.0})
    assert bus.project.model_dump() == before


def test_timeline_commands(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    media = add_video(bus, duration=10.0)
    track = bus.execute("track.add", {})["track_id"]
    c1 = bus.execute("clip.add", {"track_id": track, "media_id": media, "out_point": 4.0})["clip_id"]
    c2 = bus.execute("clip.add", {"track_id": track, "media_id": media, "in_point": 2.0, "out_point": 5.0})["clip_id"]
    _, clip2 = bus.project.find_clip(c2)
    assert clip2.start == pytest.approx(4.0)  # appended after first clip

    bus.execute("clip.trim", {"clip_id": c1, "out_point": 3.0})
    assert bus.project.find_clip(c1)[1].duration == pytest.approx(3.0)
    with pytest.raises(CommandError):
        bus.execute("clip.trim", {"clip_id": c1, "in_point": 3.5})

    bus.execute("clip.move", {"clip_id": c1, "start": 10.0})
    assert [c.id for c in bus.project.tracks[0].clips] == [c2, c1]

    removed = bus.execute("media.remove", {"media_id": media})
    assert removed["clips_removed"] == 2
    bus.undo()
    assert len(bus.project.tracks[0].clips) == 2


def test_patch_media_applies_to_history(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    media = add_video(bus)
    bus.execute("project.rename", {"name": "x"})
    bus.patch_media(media, proxy_status="ready", proxy_path="cache/proxies/p.mp4")
    bus.undo()  # back to the state right after media.add
    assert bus.project.media[media].proxy_status == "ready"


def test_trajectory_log(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    bus.execute("project.rename", {"name": "A"}, source="ai")
    with pytest.raises(CommandError):
        bus.execute("project.rename", {"name": ""})
    lines = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [(r["source"], r["command"], r["ok"]) for r in lines] == [
        ("ai", "project.rename", True),
        ("user", "project.rename", False),
    ]
    assert lines[0]["params"] == {"name": "A"}


def test_events_emitted(tmp_path: Path) -> None:
    bus = make_bus(tmp_path)
    events: list[dict] = []
    bus.subscribe(events.append)
    bus.execute("project.rename", {"name": "A"})
    bus.undo()
    assert [e["type"] for e in events] == ["command", "undo"]


def test_command_schemas_listed() -> None:
    names = {c["name"] for c in list_commands()}
    assert {"project.rename", "media.add", "clip.add", "clip.trim"} <= names
    clip_add = next(c for c in list_commands() if c["name"] == "clip.add")
    assert "track_id" in clip_add["params"]["properties"]
