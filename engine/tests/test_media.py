from __future__ import annotations

from pathlib import Path

import av
import pytest

from aieditor_engine.core.project import MediaKind
from aieditor_engine.media.probe import ProbeError, make_thumbnail, probe
from aieditor_engine.media.proxy import make_proxy, proxy_size


def test_probe_image(image_file: Path) -> None:
    info = probe(image_file)
    assert info["kind"] == MediaKind.IMAGE
    assert (info["width"], info["height"]) == (640, 480)


def test_probe_video(video_file: Path) -> None:
    info = probe(video_file)
    assert info["kind"] == MediaKind.VIDEO
    assert (info["width"], info["height"]) == (1280, 960)
    assert info["fps"] == pytest.approx(25)
    assert info["duration"] == pytest.approx(2.0, abs=0.15)
    assert info["has_audio"]


def test_probe_rejects_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ProbeError):
        probe(bad)
    with pytest.raises(ProbeError):
        probe(tmp_path / "missing.png")


def test_thumbnails(image_file: Path, video_file: Path, tmp_path: Path) -> None:
    assert make_thumbnail(image_file, MediaKind.IMAGE, tmp_path / "t1.jpg")
    assert make_thumbnail(video_file, MediaKind.VIDEO, tmp_path / "t2.jpg")
    assert (tmp_path / "t2.jpg").stat().st_size > 0


@pytest.mark.parametrize(
    ("src", "expected"),
    [((1920, 1080), (1280, 720)), ((3840, 2160), (1280, 720)), ((1080, 1920), (406, 720)), ((640, 480), (640, 480)), ((641, 481), (640, 480))],
)
def test_proxy_size(src: tuple[int, int], expected: tuple[int, int]) -> None:
    assert proxy_size(*src) == expected


def test_make_proxy(video_file: Path, tmp_path: Path) -> None:
    progress: list[float] = []
    dst = tmp_path / "proxy.mp4"
    encoder = make_proxy(video_file, dst, progress.append)
    assert encoder
    assert progress[-1] == 1.0
    with av.open(str(dst)) as c:
        v = c.streams.video[0]
        assert (v.codec_context.width, v.codec_context.height) == (960, 720)
        assert len(c.streams.audio) == 1
        frames = sum(1 for _ in c.decode(video=0))
    assert 45 <= frames <= 55
