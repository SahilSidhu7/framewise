"""Onboarding: system check, benchmark, app settings."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aieditor_engine.api.server import create_app
from aieditor_engine.system import AppSettings, benchmark, memory_gb, system_info


def test_system_info_and_benchmark() -> None:
    info = system_info()
    assert info["cores"] >= 1 and info["disk_free_gb"] > 0 and info["cpu"]
    assert set(info["ai"]) == {"editing_model", "speech", "face_model"}
    assert info["hardware_encoder"] in (None, "nvenc")
    mem = memory_gb()
    assert mem is None or mem > 0.5
    b = benchmark(seconds=0.2)
    assert b["ms_per_frame"] > 0 and b["reel_seconds"] > 0 and b["tier"] in ("fast", "good", "basic") and b["summary"]


def test_app_settings(tmp_path) -> None:
    s = AppSettings(tmp_path)
    assert s.get() == {"onboarded": False, "focus": None}
    assert s.update({"onboarded": True, "focus": "reels"}) == {"onboarded": True, "focus": "reels"}
    assert AppSettings(tmp_path).get()["focus"] == "reels"
    for bad in ({"focus": "cooking"}, {"onboarded": "yes"}, {"secret": 1}):
        try:
            s.update(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_routes() -> None:
    c = TestClient(create_app(token="t"))
    h = {"X-Engine-Token": "t"}
    assert c.get("/system").status_code == 401
    assert c.get("/app/settings", headers=h).json()["onboarded"] is False
    assert c.post("/app/settings", headers=h, json={"onboarded": True}).json()["onboarded"] is True
    assert c.post("/app/settings", headers=h, json={"focus": "nope"}).status_code == 422
    assert c.get("/system", headers=h).json()["cores"] >= 1
    assert c.post("/system/benchmark", headers=h).json()["tier"] in ("fast", "good", "basic")
