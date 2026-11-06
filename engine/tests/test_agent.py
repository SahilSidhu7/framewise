from __future__ import annotations

import os

import pytest

from aieditor_engine.agent import planner
from aieditor_engine.agent.grammar import dsl_grammar
from aieditor_engine.agent.llm import LLM
from aieditor_engine.agent.planner import (
    Hints,
    plan,
    plan_rules,
    request_hints,
    sanitize,
    wants,
)
from aieditor_engine.dsl import compile_program

NEUTRAL = {"size": [1200, 800], "orientation": "landscape", "brightness_median": 0.5, "saturation_mean": 0.2,
           "warmth": 0.0, "palette": [{"hex": "#808080", "share": 1.0}], "notes": []}
DARK_TILTED = {**NEUTRAL, "notes": ["underexposed", "tilted horizon (-2.0 deg)"]}


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(LLM, "available", lambda: False)


def looks_in(program: str) -> list[str]:
    return [o["params"]["name"] for o in compile_program(program) if o["op"] == "look"]


@pytest.mark.parametrize(
    ("prompt", "look"),
    [
        ("make this look ancient", "ancient_sepia"),
        ("give it a cinematic movie look", "teal_orange_cinematic"),
        ("vintage retro vibe", "vintage_70s"),
        ("black and white please", "bw_classic"),
        ("aesthetic instagram feel", "aesthetic_soft"),
        ("dark and moody", "moody_dark"),
        ("neon cyberpunk", "cyberpunk"),
    ],
)
def test_rules_pick_matching_look(no_llm, prompt: str, look: str) -> None:
    result = plan(prompt, NEUTRAL)
    first = result["variations"][0]
    assert looks_in(first["program"])[0] == look
    assert len(result["variations"]) == 3
    for v in result["variations"]:
        compile_program(v["program"])  # every variation is valid DSL


def test_rules_strength_words(no_llm) -> None:
    assert "s=0.6" in plan_rules("slightly vintage", NEUTRAL)[0].program
    assert "s=1.3" in plan_rules("very vintage", NEUTRAL)[0].program


def test_rules_fix_uses_analysis(no_llm) -> None:
    v = plan_rules("fix this photo", DARK_TILTED)[0]
    ops = [o["op"] for o in compile_program(v.program)]
    assert ops[0] == "straighten" and "adjust" in ops
    assert "exposure=0.6" in v.program


def test_rules_crop_only_when_asked(no_llm) -> None:
    assert "crop" not in plan_rules("aesthetic look", NEUTRAL)[0].program
    assert "crop 9:16" in plan_rules("aesthetic look for my story", NEUTRAL)[0].program
    assert "crop 4:5" in plan_rules("vintage, crop for instagram", NEUTRAL)[0].program


def test_rules_refine_current_look(no_llm) -> None:
    v = plan_rules("less", NEUTRAL, "look vintage_70s s=1")[0]
    assert v.program == "look vintage_70s s=0.7"


def test_negation() -> None:
    assert wants(r"\bgrain\w*\b", " add some grain ")
    assert not wants(r"\bgrain\w*\b", " make it less grainy ")
    assert not wants(r"\bvignette\b", " no vignette please ")


def test_request_hints() -> None:
    h = request_hints("ancient look for a post", NEUTRAL)
    assert h.looks and h.looks[0] == "ancient_sepia" and h.aspect == "4:5"
    h = request_hints("fix it", DARK_TILTED)
    assert h.looks == () and h.fix_only and "straighten" in h.ops
    h = request_hints("warmer", NEUTRAL, "look noir s=0.8")
    assert h.looks == ("noir",) and h.strength == 0.8
    assert request_hints("edit this", NEUTRAL).looks == planner.SAFE_LOOKS


def test_sanitize_enforces_playbook() -> None:
    hints = Hints(("noir",), None, False, 1.0, ("adjust",), " moody ")
    ops = compile_program("adjust exposure=2 contrast=90\nlook noir s=1\nlook pastel\nadjust saturation=-10\nstraighten")
    out, fixes = sanitize(ops, NEUTRAL, hints)
    assert [o["op"] for o in out] == ["adjust", "look"]
    assert out[0]["params"] == {"contrast": 40, "saturation": -10}
    assert any("exposure" in f for f in fixes) and any("straighten" in f for f in fixes)


def test_grammar_constraints() -> None:
    g = dsl_grammar(("noir",), None, ("adjust",))
    assert 'lookname ::= "noir"' in g
    assert "s-crop" not in g and "s-grain" not in g
    g = dsl_grammar((), "4:5", ("adjust", "grain"))
    assert '"crop aspect=4:5"' in g and "lookname" not in g and "s-grain" in g


def test_variations_topped_up_from_rules(monkeypatch) -> None:
    monkeypatch.setattr(LLM, "available", lambda: True)
    monkeypatch.setattr(LLM, "chat", lambda *a, **k: "# vintage\nlook vintage_70s s=1")
    result = plan("vintage", NEUTRAL)
    sources = [v["source"] for v in result["variations"]]
    assert sources[0] == "llm" and len(sources) == 3 and "rules" in sources


@pytest.mark.skipif(not os.environ.get("AIEDITOR_TEST_LLM"), reason="set AIEDITOR_TEST_LLM=1 to run against the local model")
def test_live_model() -> None:
    try:
        result = plan("make it look ancient", NEUTRAL)
        assert result["llm_error"] is None
        assert result["variations"][0]["source"] == "llm"
        assert looks_in(result["variations"][0]["program"]) == ["ancient_sepia"]
    finally:
        LLM.stop()
