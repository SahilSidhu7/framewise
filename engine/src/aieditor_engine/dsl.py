"""Edit DSL: the compact text language EditLM writes (and the UI can show).

    # moody request -> cool shadows, low sat
    look moody_dark s=0.7 | hsl orange sat=5
    vignette 0.3
    crop 4:5

Rules
- Statements are separated by newlines or `|`.
- `#` at the start of a statement begins a comment (the model's rationale).
- `op` then arguments: `key=value`, or bare values filled into the op's
  positional parameters in order. Values: numbers, true/false, words,
  "quoted strings", ratios like 4:5, point lists like 0:0,0.5:0.6,1:1.
- Aliases: bw, grade, ca, leak, dust, s= (look strength).

`compile_program` validates everything and returns op dicts; `to_dsl` turns
an op stack back into text (shown to users and fed back to the model).
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from pydantic import ValidationError

from .imaging.ops import OPS
from .imaging.recipes import load_recipes

ALIASES = {"bw": "black_white", "grade": "color_grade", "ca": "chromatic_aberration", "leak": "light_leak", "dust": "dust_scratches", "lut3d": "lut"}
PARAM_ALIASES = {"look": {"s": "strength"}, "hsl": {"saturation": "sat", "luminance": "lum", "lightness": "lum"}}
_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class DslError(ValueError):
    def __init__(self, message: str, statement: int | None = None, text: str | None = None) -> None:
        self.statement, self.text = statement, text
        where = f"statement {statement} `{text}`: " if statement is not None else ""
        super().__init__(where + message)


def _value(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if _NUM.match(raw):
        f = float(raw)
        return int(f) if f.is_integer() and "." not in raw and "e" not in low else f
    return raw


def _statements(text: str) -> list[str]:
    out = []
    for line in text.replace("\r", "").split("\n"):
        for part in _split_pipes(line):
            part = part.strip()
            if part and not part.startswith("#"):
                out.append(part)
    return out


def _split_pipes(line: str) -> list[str]:
    parts, buf, quote = [], [], None
    for ch in line:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == "|":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def parse_statement(stmt: str, index: int | None = None) -> tuple[str, dict[str, Any]]:
    try:
        tokens = shlex.split(stmt, posix=True)
    except ValueError as exc:
        raise DslError(str(exc), index, stmt) from exc
    # trailing inline comment: a token that starts with '#' and isn't a key=value
    for i, tok in enumerate(tokens):
        if tok.startswith("#"):
            tokens = tokens[:i]
            break
    if not tokens:
        raise DslError("empty statement", index, stmt)
    name = ALIASES.get(tokens[0].lower(), tokens[0].lower())
    spec = OPS.get(name)
    if spec is None:
        raise DslError(f"unknown op `{tokens[0]}`. Known: {', '.join(sorted(OPS))}", index, stmt)
    aliases = PARAM_ALIASES.get(name, {})
    params: dict[str, Any] = {}
    positional = list(spec.positional)
    for tok in tokens[1:]:
        if "=" in tok:
            key, raw = tok.split("=", 1)
            key = aliases.get(key.lower(), key.lower())
            if key not in spec.params_model.model_fields:
                raise DslError(f"`{name}` has no param `{key}`. Params: {', '.join(spec.params_model.model_fields)}", index, stmt)
            params[key] = _value(raw)
        else:
            if not positional:
                raise DslError(f"unexpected value `{tok}` (use key=value)", index, stmt)
            params[positional.pop(0)] = _value(tok)
    if name == "look":
        look_name = str(params.get("name", ""))
        if look_name not in load_recipes():
            raise DslError(f"unknown look `{look_name}`. Looks: {', '.join(sorted(load_recipes()))}", index, stmt)
    try:
        validated = spec.params_model.model_validate(params)
    except ValidationError as exc:
        msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise DslError(msg, index, stmt) from exc
    # keep only what was given (compact stacks; defaults come from the model)
    dumped = validated.model_dump(mode="json")
    return name, {k: dumped[k] for k in params}


def compile_program(text: str) -> list[dict[str, Any]]:
    """Parse + validate. Raises DslError on the first bad statement."""
    ops = []
    for i, stmt in enumerate(_statements(text), start=1):
        name, params = parse_statement(stmt, i)
        ops.append({"op": name, "params": params})
    return ops


def compile_lenient(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """For model output: clamp out-of-range numbers, drop statements that still
    fail, and report what was fixed. Never raises."""
    ops: list[dict[str, Any]] = []
    warnings: list[str] = []
    for i, stmt in enumerate(_statements(text), start=1):
        try:
            name, params = parse_statement(_clamp_statement(stmt), i)
            ops.append({"op": name, "params": params})
        except DslError as exc:
            warnings.append(str(exc))
    return ops, warnings


def _clamp_statement(stmt: str) -> str:
    try:
        tokens = shlex.split(stmt, posix=True)
    except ValueError:
        return stmt
    if not tokens:
        return stmt
    name = ALIASES.get(tokens[0].lower(), tokens[0].lower())
    spec = OPS.get(name)
    if spec is None:
        return stmt
    aliases = PARAM_ALIASES.get(name, {})
    out = [tokens[0]]
    for tok in tokens[1:]:
        if "=" in tok:
            key, raw = tok.split("=", 1)
            field = spec.params_model.model_fields.get(aliases.get(key.lower(), key.lower()))
            if field is not None and _NUM.match(raw):
                value = float(raw)
                for meta in field.metadata:
                    if getattr(meta, "ge", None) is not None:
                        value = max(value, meta.ge)
                    if getattr(meta, "le", None) is not None:
                        value = min(value, meta.le)
                    if getattr(meta, "gt", None) is not None and value <= meta.gt:
                        value = meta.gt + 1e-3
                raw = f"{value:g}"
            tok = f"{key}={raw}"
        out.append(shlex.quote(tok) if " " in tok else tok)
    return " ".join(out)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    if isinstance(v, list):
        return ",".join(f"{_fmt(a)}:{_fmt(b)}" for a, b in v)
    s = str(v)
    return f'"{s}"' if (" " in s or "|" in s or not s) else s


def to_dsl(ops: list[dict[str, Any]]) -> str:
    lines = []
    for item in ops:
        if not item.get("enabled", True):
            continue
        name, params = item["op"], dict(item.get("params", {}))
        spec = OPS[name]
        parts = [name]
        for key in spec.positional:
            if key in params:
                parts.append(_fmt(params.pop(key)))
            else:
                break
        for key, value in params.items():
            if value is None:
                continue
            key_out = "s" if (name == "look" and key == "strength") else key
            parts.append(f"{key_out}={_fmt(value)}")
        lines.append(" ".join(parts))
    return "\n".join(lines)
