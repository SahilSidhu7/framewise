"""Generate GBNF grammars for the Edit DSL from the op schemas.

Shape enforced (per request):
    # rationale
    [0-3 correction/other statements]
    [at most ONE look, chosen from the allowed set]
    [0-4 finishing statements]

Request hints narrow the grammar further: which looks are allowed, whether
crop is allowed (only when the user asked), and fix-only mode. Numeric ranges
and required params are enforced afterwards by the lenient compiler.
"""

from __future__ import annotations

import types
import typing
from functools import lru_cache

from ..imaging.ops import OPS
from ..imaging.recipes import load_recipes

AI_OPS = [n for n in OPS if n not in ("lut",)]
FIX_OPS = ("adjust", "white_balance", "straighten", "sharpen", "denoise", "dehaze", "curves", "hsl")


def _lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _value_rule(key: str, annotation: object) -> str:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        return _value_rule(key, inner[0]) if inner else "word"
    if origin is typing.Literal:
        return "(" + " | ".join(_lit(str(a)) for a in args) + ")"
    if origin is list:
        return "points"
    if annotation is bool:
        return '("true" | "false")'
    if annotation in (int, float):
        return "num"
    if key in ("tint", "color"):
        return "hex"
    if key == "aspect":
        return "ratio"
    return "word"


def _op_rule(op: str, aspect: str | None) -> list[str]:
    rule = f"s-{op.replace('_', '-')}"
    if op == "crop":
        return [f'{rule} ::= "crop aspect={aspect}"'] if aspect else []
    fields = OPS[op].params_model.model_fields
    keys = [k for k in fields if k != "seed"]
    if not keys:
        return [f"{rule} ::= {_lit(op)}"]
    arg = f"a-{op.replace('_', '-')}"
    alts = [f"{_lit(k + '=')} {_value_rule(k, fields[k].annotation)}" for k in keys]
    return [f"{arg} ::= " + " | ".join(alts), f'{rule} ::= {_lit(op)} (" " {arg}){{1,8}}']


@lru_cache(maxsize=64)
def dsl_grammar(
    looks: tuple[str, ...] | None = None, aspect: str | None = None, ops_allowed: tuple[str, ...] | None = None
) -> str:
    """looks=None -> any look; looks=() -> no look. ops_allowed=None -> all non-look ops."""
    all_looks = sorted(load_recipes())
    allowed_looks = all_looks if looks is None else [n for n in looks if n in all_looks]
    ops = [o for o in AI_OPS if o != "look" and (ops_allowed is None or o in ops_allowed)]
    if aspect and "crop" not in ops:
        ops.append("crop")
    if not aspect:
        ops = [o for o in ops if o != "crop"]
    if not ops:
        ops = ["adjust"]

    lines = [
        'rationale ::= "# " [^\\n]{3,160}',
        'nl ::= "\\n"',
        'num ::= "-"? [0-9]{1,3} ("." [0-9]{1,2})?',
        'hex ::= "#" [0-9a-f]{6}',
        'ratio ::= [0-9]{1,2} ":" [0-9]{1,2}',
        'points ::= num ":" num ("," num ":" num){1,7}',
        "word ::= [a-z_]{1,24}",
    ]
    stmt_rules = []
    for op in ops:
        rules = _op_rule(op, aspect)
        if rules:
            lines.extend(rules)
            stmt_rules.append(f"s-{op.replace('_', '-')}")
    lines.append("stmt ::= " + " | ".join(stmt_rules))
    if allowed_looks:
        lines.append("lookname ::= " + " | ".join(_lit(n) for n in allowed_looks))
        lines.append('s-look ::= "look " lookname " s=" num')
        root = "root ::= rationale (nl stmt){0,3} nl s-look (nl stmt){0,4} nl?" if looks else \
               "root ::= rationale ((nl stmt){0,3} nl s-look (nl stmt){0,4} | (nl stmt){1,6}) nl?"
    else:
        root = "root ::= rationale (nl stmt){1,6} nl?"
    return root + "\n" + "\n".join(lines) + "\n"
