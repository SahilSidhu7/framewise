"""Offline Pro licence: Ed25519, keys, trial, and the free-tier limits in automation."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from aieditor_engine import licence as L
from aieditor_engine.automation.jobs import JobError, JobManager, JobRequest

SEED = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
PUB = "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"


@pytest.fixture
def test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(L, "PUBLIC_KEY", PUB)


@pytest.fixture
def free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> L.LicenceStore:
    folder = tmp_path / "free_config"
    monkeypatch.setenv("FRAMEWISE_CONFIG", str(folder))
    return L.LicenceStore()


def key(**kw: object) -> str:
    payload = {"v": 1, "id": "L-test", "plan": "pro", "to": "Tester", "seats": 1, "iat": int(time.time()), "exp": None}
    payload.update(kw)
    return L.make_key(SEED, payload)


def test_ed25519_rfc8032_vectors() -> None:
    vectors = [
        ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        (SEED.hex(), PUB, "72",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ]
    for sk, pk, msg, sig in vectors:
        sk_b, pk_b, msg_b, sig_b = (bytes.fromhex(x) for x in (sk, pk, msg, sig))
        assert L.public_key(sk_b) == pk_b
        assert L.sign(sk_b, msg_b) == sig_b
        assert L.verify(pk_b, msg_b, sig_b)
        assert not L.verify(pk_b, msg_b + b"!", sig_b)
        assert not L.verify(pk_b, msg_b, sig_b[:63] + bytes([sig_b[63] ^ 1]))
        # S >= L must be rejected (signature malleability)
        s = int.from_bytes(sig_b[32:], "little") + L._L
        assert not L.verify(pk_b, msg_b, sig_b[:32] + s.to_bytes(32, "little"))


def test_shipped_public_key_is_a_valid_point() -> None:
    assert L._decompress(bytes.fromhex(L.PUBLIC_KEY)) is not None


def test_parse_keys(test_key: None) -> None:
    lic = L.parse(key(seats=3))
    assert (lic.plan, lic.to, lic.seats, lic.expires) == ("pro", "Tester", 3, None)
    k = key()
    assert L.parse(k[:40] + "\n " + k[40:]).id == "L-test"  # pasted with a line break
    body, sig = k[len(L.PREFIX):].split(".")
    forged = json.loads(L._b64d(body))
    forged["plan"] = "studio"
    with pytest.raises(L.LicenceError, match="signature"):
        L.parse(L.PREFIX + L._b64e(json.dumps(forged).encode()) + "." + sig)
    for bad in ("", "hello", "AIE1-abc", "AIE1-@@@.###"):
        with pytest.raises(L.LicenceError):
            L.parse(bad)
    with pytest.raises(L.LicenceError, match="different version"):
        L.parse(key(plan="enterprise"))
    other = L.make_key(bytes(32), {"v": 1, "id": "x", "plan": "pro", "iat": 1, "exp": None})
    with pytest.raises(L.LicenceError, match="signature"):
        L.parse(other)
    assert L.parse(key(exp=int(time.time()) - 10)).expired()


def test_store_trial_activate_env(test_key: None, free: L.LicenceStore, monkeypatch: pytest.MonkeyPatch) -> None:
    st = free.status()
    assert (st["tier"], st["pro"]) == ("free", False)
    with pytest.raises(L.LicenceRequired, match="framewise licence trial"):
        free.require("serve")

    st = free.start_trial()
    assert (st["tier"], st["trial_days_left"]) == ("trial", 14)
    assert free.start_trial()["tier"] == "trial"  # idempotent while running
    later = time.time() + 15 * 86400
    assert free.status(now=later)["tier"] == "free"
    data = json.loads(free.path.read_text())
    free.path.write_text(json.dumps({**data, "trial_started": data["trial_started"] - 15 * 86400}))
    with pytest.raises(L.LicenceError, match="already been used"):
        free.start_trial()
    free.path.write_text(json.dumps(data))

    with pytest.raises(L.LicenceError):
        free.activate("AIE1-nope.nope")
    with pytest.raises(L.LicenceError, match="expired"):
        free.activate(key(exp=int(time.time()) - 10))
    st = free.activate(key(plan="studio", to="Agency"))
    assert (st["tier"], st["licence"]["to"]) == ("studio", "Agency")
    assert free.status(now=later)["tier"] == "studio"  # a lifetime key outlives the trial
    assert free.deactivate()["tier"] == "trial"

    monkeypatch.setenv("AIEDITOR_LICENSE", key())
    assert free.status()["tier"] == "pro"
    monkeypatch.setenv("AIEDITOR_LICENSE", "garbage")
    st = free.status()
    assert st["problem"] and st["tier"] == "trial"


def test_free_tier_limits(free: L.LicenceStore, tmp_path: Path, image_file: Path, video_file: Path) -> None:
    m = JobManager(tmp_path / "home", start=False, licence=free)
    photo2 = tmp_path / "second.png"
    shutil.copy(image_file, photo2)

    with pytest.raises(JobError, match="several photos") as e:
        m.submit(JobRequest(recipe="photo_edit", inputs=[str(image_file), str(photo2)]), allow_local=True)
    assert e.value.status == 402
    with pytest.raises(JobError, match="Webhooks"):
        m.submit(JobRequest(recipe="photo_edit", inputs=[str(image_file)], webhook="http://127.0.0.1:9/h"), allow_local=True)
    assert m.submit(JobRequest(recipe="photo_edit", inputs=[str(image_file)]), allow_local=True)["notes"] == []

    job = m.submit(JobRequest(recipe="reel", inputs=[str(video_file)], options={"count": 4}), allow_local=True)
    assert job["options"]["count"] == 1 and "1 reel" in job["notes"][0]

    api = JobManager(tmp_path / "home", start=False, licence=free, pro_feature="serve")
    with pytest.raises(JobError, match="automation API") as e:
        api.submit(JobRequest(recipe="photo_edit", inputs=[str(image_file)]), allow_local=True)
    assert e.value.status == 402

    free.start_trial()
    job = m.submit(JobRequest(recipe="reel", inputs=[str(video_file)], options={"count": 4}), allow_local=True)
    assert job["options"]["count"] == 4 and job["notes"] == []
    assert api.submit(JobRequest(recipe="photo_edit", inputs=[str(image_file), str(photo2)]), allow_local=True)


def test_cli_gating(free: L.LicenceStore, tmp_path: Path, capsys: pytest.CaptureFixture[str], test_key: None) -> None:
    from aieditor_engine.cli import main

    assert main(["--home", str(tmp_path / "h"), "serve", "--port", "1"]) == 3
    assert "automation API" in capsys.readouterr().err
    assert main(["--home", str(tmp_path / "h"), "watch", str(tmp_path)]) == 3

    assert main(["licence"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "free"
    assert main(["licence", "activate", "AIE1-bad.key"]) == 1
    capsys.readouterr()
    assert main(["license", "activate", key()]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "pro"
    assert main(["licence", "deactivate"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "free"


def test_licence_routes(free: L.LicenceStore, test_key: None) -> None:
    from fastapi.testclient import TestClient

    from aieditor_engine.api.server import create_app

    c = TestClient(create_app(token="t"))
    h = {"X-Engine-Token": "t"}
    assert c.get("/licence").status_code == 401
    assert c.get("/licence", headers=h).json()["tier"] == "free"
    assert c.post("/licence/activate", headers=h, json={"key": "nope"}).status_code == 422
    assert c.post("/licence/activate", headers=h, json={"key": key()}).json()["tier"] == "pro"
    assert c.delete("/licence", headers=h).json()["tier"] == "free"
    assert c.post("/licence/trial", headers=h).json()["tier"] == "trial"
