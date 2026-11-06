"""EditLM v0: turn a user request + photo measurements into Edit DSL programs.

Model path: local LLM (grammar-constrained), several variations at different
temperatures, lenient compile, de-duplicate.
Fallback: deterministic rule planner (keywords -> looks/adjustments), used when
no model is installed or the model returns nothing usable. The product always
answers, and the rule planner doubles as a test oracle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..dsl import compile_lenient, to_dsl
from ..imaging.ops import OPS
from ..imaging.recipes import load_recipes
from .grammar import AI_OPS, dsl_grammar
from .llm import LLM, LLMUnavailable

log = logging.getLogger(__name__)


@dataclass
class Variation:
    program: str
    rationale: str
    source: str  # "llm" | "rules"
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"program": self.program, "rationale": self.rationale, "source": self.source, "warnings": self.warnings}


# ---------------------------------------------------------------- prompting

PLAYBOOK = """Editing rules:
- Correct problems first (exposure, white balance, tilt), then apply a look.
- Prefer ONE look that matches the mood; tune it with s= (0.5 subtle, 1 normal, 1.3 strong).
- Protect highlights and skin: don't push exposure, saturation or contrast to extremes.
- Grain, vignette, bloom: subtle values (0.1-0.4) look professional.
- "old/ancient/antique" -> ancient_sepia. "vintage/retro" -> vintage_70s or polaroid. "cinematic/movie" -> teal_orange_cinematic.
- "aesthetic/instagram" -> aesthetic_soft or pastel. "dark/moody" -> moody_dark. "black and white" -> bw_classic or noir.
- Crop only if asked (square/instagram post -> crop aspect=4:5 or 1:1; story/reel -> crop aspect=9:16).
- Output 1-6 statements. Keep values inside the listed ranges."""

_NO_CROP = "crop: not requested, do not crop"
EXAMPLES = [
    (
        "make it look ancient",
        f"notes: none\nhints: allowed looks: ancient_sepia, vintage_70s; {_NO_CROP}; suggested look strength s=1",
        "# ancient request -> sepia, aged paper, dust\nlook ancient_sepia s=1",
    ),
    (
        "brighten it up and make it aesthetic",
        f"notes: underexposed\nhints: allowed looks: aesthetic_soft, pastel; {_NO_CROP}; suggested look strength s=1",
        "# dark photo -> lift exposure first, then soft airy look\nadjust exposure=0.7 shadows=25\nlook aesthetic_soft s=0.9",
    ),
    (
        "cinematic, for an instagram post",
        "notes: tilted horizon (-2.1 deg)\nhints: allowed looks: teal_orange_cinematic; crop: aspect=4:5 requested; suggested look strength s=1",
        "# level the tilt, movie grade, 4:5 post crop\nstraighten\nlook teal_orange_cinematic s=1\ncrop aspect=4:5",
    ),
    (
        "fix it",
        "notes: cool cast, flat / low contrast\nhints: no look: use adjustments/effects only; " + _NO_CROP + "; task: correction only",
        "# remove the blue cast and add some contrast\nadjust temperature=12 contrast=15 vibrance=10",
    ),
    (
        "less grain, a bit warmer",
        f"notes: none\ncurrent edit:\nlook vintage_70s s=1\nhints: allowed looks: vintage_70s; {_NO_CROP}; suggested look strength s=1",
        "# keep the vintage look but weaker (less grain), warm it\nlook vintage_70s s=0.7\nadjust temperature=12",
    ),
]


def _range(field_info) -> str:
    lo = hi = None
    for meta in field_info.metadata:
        lo = getattr(meta, "ge", lo)
        hi = getattr(meta, "le", hi)
    return f"{lo:g}..{hi:g}" if lo is not None and hi is not None else ""


@lru_cache(maxsize=1)
def op_reference() -> str:
    lines = []
    for name in AI_OPS:
        if name == "look":
            continue
        spec = OPS[name]
        params = []
        for key, f in spec.params_model.model_fields.items():
            if key == "seed":
                continue
            args = getattr(f.annotation, "__args__", None)
            if args and all(isinstance(a, str) for a in args):
                params.append(f"{key}={'|'.join(args)}")
            else:
                rng = _range(f)
                params.append(f"{key}{'(' + rng + ')' if rng else ''}")
        lines.append(f"{name} {' '.join(params)} : {spec.description}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def look_reference() -> str:
    return "\n".join(f"{r.name}: {r.description}" for r in load_recipes().values())


def system_prompt() -> str:
    return (
        "You are EditLM, the editing engine of a photo editor. Turn the user's request into an edit program "
        "in the Edit DSL. First line: '# ' + a short reason. Then one statement per line: "
        "`op key=value ...` or `look NAME s=STRENGTH`.\n\n"
        f"OPS:\n{op_reference()}\n\nLOOKS (look NAME s=0..1.5):\n{look_reference()}\n\n{PLAYBOOK}"
    )


def media_brief(analysis: dict[str, Any], current_program: str) -> str:
    notes = ", ".join(analysis.get("notes", [])) or "none"
    palette = " ".join(p["hex"] for p in analysis.get("palette", [])[:4])
    lines = [
        f"photo: {analysis['size'][0]}x{analysis['size'][1]} {analysis['orientation']}",
        f"notes: {notes}",
        f"brightness={analysis['brightness_median']} saturation={analysis['saturation_mean']} warmth={analysis['warmth']} palette={palette}",
    ]
    if current_program.strip():
        lines.append("current edit:\n" + current_program.strip())
    return "\n".join(lines)


def _messages(prompt: str, brief: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system_prompt()}]
    for user, ex_brief, answer in EXAMPLES:
        msgs.append({"role": "user", "content": f"{ex_brief}\nrequest: {user}"})
        msgs.append({"role": "assistant", "content": answer})
    msgs.append({"role": "user", "content": f"{brief}\nrequest: {prompt}"})
    return msgs


def _split_rationale(text: str) -> tuple[str, str]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    rationale = ""
    if lines and lines[0].lstrip().startswith("#"):
        rationale = lines.pop(0).lstrip("# ").strip()
    return rationale, "\n".join(lines)


# ------------------------------------------------------------------ planner


@dataclass
class Hints:
    """What the rules layer understood; constrains the model's grammar."""

    looks: tuple[str, ...] | None  # None = any look, () = no look
    aspect: str | None
    fix_only: bool
    strength: float
    ops: tuple[str, ...] = ()  # non-look ops the model may use
    text: str = ""

    def prompt_lines(self) -> str:
        parts = []
        if self.looks:
            parts.append("allowed looks: " + ", ".join(self.looks))
        elif self.looks == ():
            parts.append("no look: use adjustments/effects only")
        parts.append(f"crop: aspect={self.aspect} requested" if self.aspect else "crop: not requested, do not crop")
        if self.fix_only:
            parts.append("task: correction only")
        parts.append(f"suggested look strength s={self.strength:g}")
        return "hints: " + "; ".join(parts)


# Always available to the model; anything else only when the user names it.
SAFE_OPS = ("adjust", "sharpen", "denoise", "dehaze")
OP_WORDS = {
    "straighten": r"\b(straight\w*|level|tilt\w*|crooked)\b",
    "grain": r"\b(grain\w*|filmic|noise texture)\b",
    "vignette": r"\bvignett\w*\b",
    "bloom": r"\b(glow\w*|bloom|halation)\b",
    "fade": r"\b(fade\w*|matte|washed)\b",
    "black_white": r"\b(black and white|black & white|b&w|bw|monochrome|grayscale|greyscale)\b",
    "color_grade": r"\b(grade|grading|split.?ton\w*|tint the)\b",
    "hsl": r"\b(hsl|greens?|blues?|skies|sky|reds?|oranges?|skin tones?|purples?|yellows?)\b",
    "curves": r"\bcurves?\b",
    "blur": r"\b(blur\w*|soften the background|dreamy blur)\b",
    "rotate": r"\b(rotate\w*|turn it|sideways|upside)\b",
    "flip": r"\b(flip\w*|mirror\w*)\b",
    "chromatic_aberration": r"\b(chromatic|fringing|aberration|rgb split)\b",
    "light_leak": r"\b(light leak|leaks?)\b",
    "dust_scratches": r"\b(dust|scratch\w*)\b",
    "texture": r"\b(texture|paper|worn)\b",
}


def request_hints(prompt: str, analysis: dict[str, Any], current_program: str = "") -> Hints:
    text = " " + prompt.lower().strip() + " "
    strength = next((s for pattern, s in INTENSITY if re.search(pattern, text)), 1.0)
    aspect = next((a for pattern, a in ASPECTS if re.search(pattern, text)), None)
    looks = _match_looks(text)
    current = re.search(r"look (\w+)(?: s=([\d.]+))?", current_program)
    tweak = any(re.search(p, text) for p, _ in ADJUST_WORDS) or any(re.search(p, text) for p, _ in EXTRA_OPS)
    refine = re.search(r"\b(more|less|stronger|weaker|reduce|increase|tone down)\b", text)
    wants_fix = bool(re.search(FIX_WORDS, text))
    ops = list(SAFE_OPS) + [op for op, pattern in OP_WORDS.items() if wants(pattern, text)]
    if "tilted horizon" in " ".join(analysis.get("notes", [])) and wants_fix and "straighten" not in ops:
        ops.append("straighten")

    def h(lk: tuple[str, ...] | None, fix: bool, s: float) -> Hints:
        return Hints(lk, aspect, fix, s, tuple(ops), text)

    if looks:
        return h(tuple(looks[:4]), False, strength)
    if current and (tweak or refine):
        return h((current.group(1),), False, float(current.group(2) or 1.0))
    if wants_fix and not tweak:
        return h((), True, strength)
    if tweak or aspect:
        return h((), False, strength)
    return h(SAFE_LOOKS, False, strength)


# Crowd-pleasers for vague requests ("edit this", "make it nice").
SAFE_LOOKS = ("vivid_pop", "bright_airy", "film_portra", "golden_hour", "aesthetic_soft")


# Edit Linter for model output: playbook rules as hard checks.
SAFE_ADJUST = {"exposure": 1.2, "contrast": 40, "highlights": 60, "shadows": 60, "whites": 40, "blacks": 40,
               "temperature": 35, "tint": 25, "vibrance": 40, "saturation": 40, "clarity": 40}


def sanitize(ops: list[dict[str, Any]], analysis: dict[str, Any], hints: Hints) -> tuple[list[dict[str, Any]], list[str]]:
    notes = " ".join(analysis.get("notes", []))
    text = hints.text
    fixed: list[str] = []
    out: list[dict[str, Any]] = []
    merged_adjust: dict[str, Any] | None = None
    seen_ops: set[str] = set()
    for item in ops:
        name, params = item["op"], dict(item["params"])
        if name == "look":
            if "look" in seen_ops:
                fixed.append("dropped extra look")
                continue
            if hints.looks and params["name"] not in hints.looks:
                fixed.append(f"look {params['name']} not allowed")
                continue
        elif name == "adjust":
            for k, limit in SAFE_ADJUST.items():
                if k in params and abs(params[k]) > limit:
                    params[k] = limit if params[k] > 0 else -limit
            asked_bright = re.search(r"\b(bright\w*|light\w*|dark\w*|exposure)\b", text)
            if "exposure" in params and not asked_bright:
                ev = params["exposure"]
                if (ev > 0.25 and "underexposed" not in notes) or (ev < -0.25 and "overexposed" not in notes and "clipped" not in notes):
                    params.pop("exposure")
                    fixed.append("removed unjustified exposure change")
            if merged_adjust is not None:
                merged_adjust.update(params)
                fixed.append("merged adjust")
                continue
            if not params:
                continue
            item = {"op": name, "params": params}
            out.append(item)
            merged_adjust = item["params"]
            seen_ops.add(name)
            continue
        elif name == "straighten" and "tilted" not in notes and not re.search(OP_WORDS["straighten"], text):
            fixed.append("removed straighten (horizon already level)")
            continue
        elif name in seen_ops and name not in ("hsl",):
            fixed.append(f"dropped duplicate {name}")
            continue
        seen_ops.add(name)
        out.append({"op": name, "params": params})
    # corrections first, look next, finishing effects last
    order = {"straighten": 0, "rotate": 0, "flip": 0, "denoise": 1, "dehaze": 1, "adjust": 2, "look": 3, "crop": 9}
    out.sort(key=lambda o: order.get(o["op"], 5))
    return out, fixed


def plan(prompt: str, analysis: dict[str, Any], current_program: str = "", variations: int = 3) -> dict[str, Any]:
    brief = media_brief(analysis, current_program)
    hints = request_hints(prompt, analysis, current_program)
    results: list[Variation] = []
    llm_error = None
    if LLM.available():
        try:
            results = _plan_llm(prompt, brief, hints, analysis, variations)
        except LLMUnavailable as exc:
            llm_error = str(exc)
            log.warning("LLM unavailable, using rules: %s", exc)
    # Top up with rule-based options so the user always gets a choice.
    if len(results) < variations:
        seen = {v.program for v in results}
        for v in plan_rules(prompt, analysis, current_program, variations + 2):
            if v.program not in seen and len(results) < variations:
                results.append(v)
                seen.add(v.program)
    return {"variations": [v.to_json() for v in results], "brief": brief, "llm_error": llm_error}


def _plan_llm(prompt: str, brief: str, hints: Hints, analysis: dict[str, Any], n: int) -> list[Variation]:
    msgs = _messages(prompt, brief + "\n" + hints.prompt_lines())
    grammar = dsl_grammar(hints.looks, hints.aspect, hints.ops)
    seen: set[str] = set()
    out: list[Variation] = []
    temps = [0.2, 0.7, 1.0, 1.2][: max(1, n)]
    for i, temp in enumerate(temps):
        text = LLM.chat(msgs, grammar=grammar, temperature=temp, seed=1234 + i)
        rationale, body = _split_rationale(text)
        ops, warnings = compile_lenient(body)
        ops, fixes = sanitize(ops, analysis, hints)
        warnings += fixes
        # The user named an effect but the model skipped it: add a sensible default.
        present = {o["op"] for o in ops}
        for pattern, stmt in EXTRA_OPS:
            op_name = stmt.split()[0]
            if op_name not in present and wants(pattern, hints.text):
                ops.extend(compile_lenient(stmt)[0])
                warnings.append(f"added requested {op_name}")
        if not ops:
            continue
        program = to_dsl(ops)
        if program in seen:
            continue
        seen.add(program)
        out.append(Variation(program, rationale, "llm", warnings))
    return out


# ------------------------------------------------------------- rule planner

INTENSITY = [
    (r"\b(very|super|extremely|heavy|heavily|strong|strongly|really|lot)\b", 1.3),
    (r"\b(slight|slightly|subtle|subtly|little|bit|light|lightly|gentle|touch)\b", 0.6),
]
ASPECTS = [
    (r"\b(story|stories|reel|reels|tiktok|9:16|vertical)\b", "9:16"),
    (r"\b(square|1:1)\b", "1:1"),
    (r"\b(post|feed|portrait crop|4:5)\b", "4:5"),
    (r"\b(widescreen|16:9|youtube thumbnail|thumbnail)\b", "16:9"),
    (r"\bcrop\w*\b.*\b(instagram|insta|ig)\b|\b(instagram|insta|ig)\b.*\bcrop\w*\b", "4:5"),
]
ADJUST_WORDS = [
    (r"\b(brighter|brighten|lighter|lighten)\b", {"exposure": 0.5, "shadows": 15}),
    (r"\b(darker|darken)\b", {"exposure": -0.5}),
    (r"\b(warmer|warm it|warm up)\b", {"temperature": 20}),
    (r"\b(cooler|colder|cool it|cool down)\b", {"temperature": -20}),
    (r"\b(more contrast|punchier|punchy|pop)\b", {"contrast": 25, "clarity": 10}),
    (r"\b(less contrast|softer|flatter)\b", {"contrast": -20}),
    (r"\b(more colou?r|more saturat\w*|vibrant|colorful|colourful)\b", {"vibrance": 30}),
    (r"\b(less colou?r|desaturat\w*|muted)\b", {"saturation": -30}),
    (r"\b(sharper|sharpen|crisp)\b", None),  # handled as op
]
EXTRA_OPS = [
    (r"\b(sharper|sharpen|crisp|clearer)\b", "sharpen amount=40"),
    (r"\b(grain|grainy|filmic)\b", "grain amount=0.3"),
    (r"\bvignette\b", "vignette amount=0.35"),
    (r"\b(glow|glowy|bloom)\b", "bloom amount=0.3"),
    (r"\b(dehaze|haze|hazy|foggy)\b", "dehaze amount=30"),
    (r"\b(denoise|noisy|noise)\b", "denoise strength=40"),
    (r"\b(light leak|leak)\b", "light_leak amount=0.3"),
]
FIX_WORDS = r"\b(fix|enhance|improve|auto|better|correct|repair|clean up|make it good|nice)\b"
NEGATION = re.compile(r"\b(less|no|without|remove|reduce|fewer|lower|not|don't|dont)\b")


def wants(pattern: str, text: str) -> bool:
    """True if the text asks FOR this thing (not 'less grain' / 'no vignette')."""
    for m in re.finditer(pattern, text):
        before = text[max(0, m.start() - 25) : m.start()].split()[-2:]
        if not any(NEGATION.fullmatch(w) for w in before):
            return True
    return False


def _match_looks(text: str) -> list[str]:
    """Rank looks by keyword overlap with tags and name (titles have generic words like 'photo')."""
    scores: dict[str, float] = {}
    for r in load_recipes().values():
        score = 0.0
        for tag in r.tags:
            if re.search(rf"\b{re.escape(tag)}\b", text):
                score += 2 if " " in tag else 1.5
        if r.name.replace("_", " ") in text:
            score += 3
        if score:
            scores[r.name] = score
    return sorted(scores, key=lambda n: -scores[n])


def plan_rules(prompt: str, analysis: dict[str, Any], current_program: str = "", variations: int = 3) -> list[Variation]:
    text = " " + prompt.lower().strip() + " "
    strength = 1.0
    for pattern, s in INTENSITY:
        if re.search(pattern, text):
            strength = s
            break

    fixes: list[str] = []
    reasons: list[str] = []
    notes = " ".join(analysis.get("notes", []))
    wants_fix = bool(re.search(FIX_WORDS, text))
    if "tilted horizon" in notes and (wants_fix or "straight" in text or "level" in text):
        fixes.append("straighten")
        reasons.append("level horizon")
    elif "straight" in text or "level the" in text:
        fixes.append("straighten")
    if wants_fix:
        adj = {}
        if "underexposed" in notes:
            adj["exposure"], adj["shadows"] = 0.6, 20
        if "overexposed" in notes or "clipped highlights" in notes:
            adj["exposure"], adj["highlights"] = -0.4, -30
        if "flat / low contrast" in notes:
            adj["contrast"] = 15
        if "warm cast" in notes:
            adj["temperature"] = -12
        if "cool cast" in notes:
            adj["temperature"] = 12
        if "soft / blurry" in notes:
            fixes.append("sharpen amount=35")
        if adj:
            at = 1 if fixes and fixes[0] == "straighten" else 0  # level first, then tone
            fixes.insert(at, "adjust " + " ".join(f"{k}={v:g}" for k, v in adj.items()))
            reasons.append("correct exposure/color")

    adjust: dict[str, float] = {}
    for pattern, params in ADJUST_WORDS:
        if params and re.search(pattern, text):
            adjust.update({k: round(v * strength, 2) for k, v in params.items()})
    extras = [stmt for pattern, stmt in EXTRA_OPS if wants(pattern, text)]
    crop = next((f"crop aspect={a}" for pattern, a in ASPECTS if re.search(pattern, text)), None)

    looks = _match_looks(text)
    current_look = re.search(r"look (\w+)(?: s=([\d.]+))?", current_program)
    if not looks and current_look and not adjust and not extras and not crop:
        # refinement like "more" / "less" on the existing look
        base = float(current_look.group(2) or 1)
        if re.search(r"\b(more|stronger|increase)\b", text):
            looks, strength = [current_look.group(1)], min(1.5, base + 0.3)
        elif re.search(r"\b(less|weaker|reduce|tone down)\b", text):
            looks, strength = [current_look.group(1)], max(0.2, base - 0.3)
    if not looks and not adjust and not extras and not crop and not fixes:
        looks = ["vivid_pop"] if wants_fix else ["aesthetic_soft"]
        reasons.append("no specific style named; picked a safe default")

    def build(look: str | None, s: float) -> Variation:
        lines = list(fixes)
        if adjust:
            lines.append("adjust " + " ".join(f"{k}={v:g}" for k, v in adjust.items()))
        if look:
            lines.append(f"look {look} s={s:g}")
        lines.extend(extras)
        if crop:
            lines.append(crop)
        ops, warnings = compile_lenient("\n".join(lines))
        why = ", ".join(reasons + ([f"{load_recipes()[look].title} look"] if look else []))
        return Variation(to_dsl(ops), why or "requested adjustments", "rules", warnings)

    out: list[Variation] = []
    seen: set[str] = set()
    candidates: list[tuple[str | None, float]] = []
    if looks:
        candidates.append((looks[0], strength))
        candidates.extend((name, strength) for name in looks[1:3])
        candidates.extend([(looks[0], round(min(1.5, strength * 1.3), 2)), (looks[0], round(strength * 0.6, 2))])
    else:
        candidates.append((None, 1.0))
    for look, s in candidates:
        v = build(look, s)
        if v.program and v.program not in seen:
            seen.add(v.program)
            out.append(v)
        if len(out) >= variations:
            break
    return out
