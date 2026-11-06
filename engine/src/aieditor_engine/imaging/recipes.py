"""Looks (recipes): named, YAML-defined op sequences with a strength knob.

Strength scales every numeric param between the op's neutral default and the
recipe's value:  value(s) = neutral + (recipe_value - neutral) * s.
So "more/less" sliders and the AI's `s=` work on any look without per-look code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .ops import OPS

RECIPE_DIR = Path(__file__).resolve().parents[1] / "recipes"

# Params where the op's "do nothing" value differs from the model default.
NEUTRAL_OVERRIDES: dict[str, dict[str, float]] = {
    "grain": {"amount": 0.0},
    "vignette": {"amount": 0.0},
    "bloom": {"amount": 0.0},
    "fade": {"amount": 0.0},
    "chromatic_aberration": {"amount": 0.0},
    "light_leak": {"amount": 0.0},
    "dust_scratches": {"amount": 0.0},
    "texture": {"amount": 0.0},
    "black_white": {"amount": 0.0},
    "sharpen": {"amount": 0.0},
    "denoise": {"strength": 0.0},
    "dehaze": {"amount": 0.0},
    "blur": {"radius": 0.0},
    "color_grade": {"shadows_sat": 0.0, "midtones_sat": 0.0, "highlights_sat": 0.0},
}
# Params that must not be interpolated (positions, hues, sizes, seeds).
NO_SCALE = {"seed", "size", "radius", "threshold", "midpoint", "feather", "shadows_hue", "midtones_hue", "highlights_hue", "balance"}


@dataclass(frozen=True)
class Recipe:
    name: str
    title: str
    description: str
    tags: tuple[str, ...]
    steps: tuple[tuple[str, dict[str, Any]], ...]

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description, "tags": list(self.tags)}


def _neutral(op_name: str, param: str) -> float | None:
    override = NEUTRAL_OVERRIDES.get(op_name, {})
    if param in override:
        return override[param]
    field = OPS[op_name].params_model.model_fields.get(param)
    default = field.default if field is not None else None
    return float(default) if isinstance(default, (int, float)) and not isinstance(default, bool) else None


def scale_params(op_name: str, params: dict[str, Any], strength: float) -> dict[str, Any]:
    out = {}
    for key, value in params.items():
        neutral = _neutral(op_name, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and neutral is not None and key not in NO_SCALE:
            out[key] = neutral + (value - neutral) * strength
        else:
            out[key] = value
    return out


def _clamp_to_model(op_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Strength > 1 may push values past limits; clamp to the schema's bounds."""
    out = dict(params)
    for key, field in OPS[op_name].params_model.model_fields.items():
        if key not in out or not isinstance(out[key], (int, float)) or isinstance(out[key], bool):
            continue
        for meta in field.metadata:
            if hasattr(meta, "ge"):
                out[key] = max(out[key], meta.ge)
            if hasattr(meta, "le"):
                out[key] = min(out[key], meta.le)
    return out


@lru_cache(maxsize=1)
def load_recipes() -> dict[str, Recipe]:
    recipes: dict[str, Recipe] = {}
    for path in sorted(RECIPE_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        steps = []
        for step in data["steps"]:
            (op_name, params), = step.items()
            if op_name not in OPS or op_name == "look":
                raise ValueError(f"{path.name}: bad op {op_name}")
            params = params or {}
            OPS[op_name].params_model.model_validate(params)  # fail fast on bad recipes
            steps.append((op_name, params))
        name = path.stem
        recipes[name] = Recipe(name, data.get("title", name), data.get("description", ""), tuple(data.get("tags", [])), tuple(steps))
    return recipes


def expand(name: str, strength: float) -> list[tuple[str, dict[str, Any]]]:
    recipe = load_recipes().get(name)
    if recipe is None:
        raise ValueError(f"unknown look: {name}")
    return [(op_name, _clamp_to_model(op_name, scale_params(op_name, params, strength))) for op_name, params in recipe.steps]
