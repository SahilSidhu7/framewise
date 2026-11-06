from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from aieditor_engine.dsl import DslError, compile_program, to_dsl
from aieditor_engine.imaging.color import (
    hsv_to_rgb,
    linear_to_srgb,
    monotone_curve,
    rgb_to_hsv,
    srgb_to_linear,
)
from aieditor_engine.imaging.lut import apply_lut3d, load_cube, write_cube
from aieditor_engine.imaging.ops import OPS, detect_horizon_angle
from aieditor_engine.imaging.recipes import expand, load_recipes, scale_params
from aieditor_engine.imaging.render import (
    analyze,
    apply_stack,
    export_image,
    load_linear,
    render_preview,
)


@pytest.fixture
def scene() -> np.ndarray:
    """Linear-light test scene: sky gradient, tilted horizon, sun, skin-tone disc."""
    h, w = 600, 900
    y = np.linspace(0, 1, h)[:, None, None]
    img = (np.array([0.35, 0.55, 0.85]) * (1 - y) + np.array([0.85, 0.8, 0.7]) * y).astype(np.float32) * np.ones((h, w, 3), np.float32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.array([[0, 380], [w, 350], [w, h], [0, h]])], 1)
    img[mask == 1] = [0.25, 0.42, 0.18]
    cv2.circle(img, (700, 130), 45, (1.0, 0.95, 0.8), -1)
    cv2.circle(img, (300, 450), 70, (0.85, 0.62, 0.5), -1)
    return srgb_to_linear(img)


def test_srgb_roundtrip() -> None:
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_hsv_roundtrip() -> None:
    rgb = np.random.default_rng(0).random((50, 50, 3), dtype=np.float32)
    assert np.allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=1e-4)


def test_monotone_curve_identity_and_monotonic() -> None:
    lut = monotone_curve([0, 1], [0, 1])
    assert np.allclose(lut, np.linspace(0, 1, len(lut)), atol=1e-6)
    lut = monotone_curve([0, 0.25, 0.75, 1], [0, 0.15, 0.85, 1])
    assert np.all(np.diff(lut) >= -1e-7)


def test_every_op_runs_with_defaults(scene: np.ndarray) -> None:
    required = {"hsl": {"band": "green"}, "lut": None, "look": None}
    for name, spec in OPS.items():
        if name in required and required[name] is None:
            continue
        out = apply_stack(scene, [{"op": name, "params": required.get(name, {})}])
        assert out.dtype == np.float32 and out.ndim == 3 and np.isfinite(out).all(), name


def test_neutral_adjust_is_identity(scene: np.ndarray) -> None:
    out = apply_stack(scene, [{"op": "adjust", "params": {}}])
    assert np.allclose(out, scene, atol=2e-3)


def test_exposure_doubles_light(scene: np.ndarray) -> None:
    out = apply_stack(scene, [{"op": "adjust", "params": {"exposure": 1}}])
    assert np.allclose(out, scene * 2, rtol=1e-3, atol=1e-4)


def test_saturation_minus_100_is_grey(scene: np.ndarray) -> None:
    d = linear_to_srgb(apply_stack(scene, [{"op": "adjust", "params": {"saturation": -100}}]))
    assert float((d.max(-1) - d.min(-1)).max()) < 0.02


def test_crop_aspect(scene: np.ndarray) -> None:
    out = apply_stack(scene, [{"op": "crop", "params": {"aspect": "4:5"}}])
    h, w = out.shape[:2]
    assert abs(w / h - 0.8) < 0.01 and h == 600


def test_rotate_90(scene: np.ndarray) -> None:
    assert apply_stack(scene, [{"op": "rotate", "params": {"angle": 90}}]).shape[:2] == (900, 600)


def test_straighten_levels_horizon(scene: np.ndarray) -> None:
    big = cv2.resize(scene, None, fx=1.5, fy=1.5)
    before = detect_horizon_angle(big)
    assert before < -1.0
    after = detect_horizon_angle(apply_stack(big, [{"op": "straighten", "params": {}}]))
    assert abs(after) < 0.3


def test_grain_deterministic(scene: np.ndarray) -> None:
    a = apply_stack(scene, [{"op": "grain", "params": {"amount": 0.5}}])
    b = apply_stack(scene, [{"op": "grain", "params": {"amount": 0.5}}])
    assert np.array_equal(a, b)


def test_disabled_op_skipped(scene: np.ndarray) -> None:
    out = apply_stack(scene, [{"op": "adjust", "params": {"exposure": 2}, "enabled": False}])
    assert np.array_equal(out, scene)


def test_lut_identity(tmp_path: Path, scene: np.ndarray) -> None:
    n = 17
    g = np.linspace(0, 1, n, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    write_cube(tmp_path / "id.cube", np.stack([r, gg, b], axis=-1))
    table = load_cube(tmp_path / "id.cube")
    d = np.clip(linear_to_srgb(scene), 0, 1)
    assert np.allclose(apply_lut3d(d, table), d, atol=1e-4)


# ----------------------------------------------------------------- looks


def test_all_looks_load_and_render(scene: np.ndarray) -> None:
    recipes = load_recipes()
    assert len(recipes) >= 15
    small = cv2.resize(scene, (300, 200))
    for name in recipes:
        out = apply_stack(small, [{"op": "look", "params": {"name": name}}])
        assert np.isfinite(out).all()
        assert not np.allclose(out, small, atol=1e-3), f"{name} does nothing"


def test_look_strength_zero_is_near_identity(scene: np.ndarray) -> None:
    small = cv2.resize(scene, (300, 200))
    out = apply_stack(small, [{"op": "look", "params": {"name": "teal_orange_cinematic", "strength": 0}}])
    assert np.allclose(out, small, atol=3e-3)


def test_strength_scaling() -> None:
    assert scale_params("grain", {"amount": 0.4, "size": 2}, 0.5) == {"amount": 0.2, "size": 2}
    assert scale_params("adjust", {"contrast": 20}, 1.5) == {"contrast": 30}
    steps = expand("ancient_sepia", 1.5)  # clamps past-limit values
    bw = dict(steps)["black_white"]
    assert bw["amount"] == 1.0


# ------------------------------------------------------------------- DSL


def test_dsl_compile_and_roundtrip() -> None:
    program = """
    # moody request -> cool shadows
    look moody_dark s=0.7 | hsl orange sat=5
    grain 0.3 size=1.5   # trailing comment
    crop 4:5
    bloom 0.2 tint=#ff3b1f
    curves 0:0,0.25:0.2,0.75:0.8,1:1
    """
    ops = compile_program(program)
    assert [o["op"] for o in ops] == ["look", "hsl", "grain", "crop", "bloom", "curves"]
    assert ops[0]["params"] == {"name": "moody_dark", "strength": 0.7}
    assert ops[1]["params"] == {"band": "orange", "sat": 5.0}
    assert ops[4]["params"]["tint"] == "#ff3b1f"
    again = compile_program(to_dsl(ops))
    assert again == ops


@pytest.mark.parametrize(
    ("program", "fragment"),
    [
        ("sparkle 3", "unknown op"),
        ("look not_a_look", "unknown look"),
        ("grain amount=5", "less than or equal"),
        ("adjust brightness=10", "has no param"),
        ("vignette 0.3 0.5 0.4 0.9", "unexpected value"),
        ('adjust contrast="ten"', "contrast"),
    ],
)
def test_dsl_errors(program: str, fragment: str) -> None:
    with pytest.raises(DslError, match=fragment):
        compile_program(program)


def test_dsl_error_names_statement() -> None:
    with pytest.raises(DslError) as exc:
        compile_program("grain 0.2\nvignette 7")
    assert exc.value.statement == 2


# -------------------------------------------------------- render / export


@pytest.fixture
def photo_path(tmp_path: Path, scene: np.ndarray) -> Path:
    from aieditor_engine.imaging.render import to_display

    path = tmp_path / "scene.png"
    Image.fromarray(to_display(scene)).save(path)
    return path


def test_preview_and_export(tmp_path: Path, photo_path: Path) -> None:
    stack = [{"op": "look", "params": {"name": "vintage_70s"}}, {"op": "crop", "params": {"aspect": "1:1"}}]
    jpeg = render_preview(photo_path, stack, max_side=400)
    assert jpeg[:2] == b"\xff\xd8"
    assert render_preview(photo_path, stack, max_side=400) is jpeg  # cached

    info = export_image(photo_path, stack, tmp_path / "out.jpg")
    assert info["width"] == info["height"] == 600
    info16 = export_image(photo_path, stack, tmp_path / "out.png", fmt="png", bit_depth=16)
    assert cv2.imread(info16["path"], cv2.IMREAD_UNCHANGED).dtype == np.uint16
    with pytest.raises(ValueError):
        export_image(photo_path, stack, tmp_path / "x.jpg", fmt="jpeg", bit_depth=16)


def test_preview_matches_export_look(tmp_path: Path, photo_path: Path) -> None:
    """Resolution-relative params: a small preview should resemble the export."""
    stack = [{"op": "look", "params": {"name": "dreamy_glow"}}]
    export_image(photo_path, stack, tmp_path / "full.png", fmt="png")
    full = load_linear(tmp_path / "full.png", max_side=300)
    preview = apply_stack(load_linear(photo_path, max_side=300), stack)
    assert float(np.abs(linear_to_srgb(full) - linear_to_srgb(preview)).mean()) < 0.03


def test_analyze_notes(scene: np.ndarray) -> None:
    dark = analyze(scene * 0.08)
    assert "underexposed" in dark["notes"]
    a = analyze(cv2.resize(scene, None, fx=1.5, fy=1.5))
    assert a["orientation"] == "landscape"
    assert any("tilted" in n for n in a["notes"])
    assert len(a["palette"]) == 5
