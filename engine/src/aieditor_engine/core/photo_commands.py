"""Photo edit-stack commands. Each one is a single undo step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..dsl import DslError, compile_program
from ..imaging.ops import OPS
from ..imaging.recipes import load_recipes
from .commands import CommandError, command
from .project import EditOp, MediaKind, Project, new_id


def _image(p: Project, media_id: str) -> list[EditOp]:
    item = p.media.get(media_id)
    if item is None:
        raise CommandError(f"media not found: {media_id}")
    if item.kind != MediaKind.IMAGE:
        raise CommandError(f"{item.name} is not a photo")
    return p.edits.setdefault(media_id, [])


def _validated(op: str, params: dict[str, Any]) -> dict[str, Any]:
    spec = OPS.get(op)
    if spec is None:
        raise CommandError(f"unknown op: {op}")
    if op == "look" and params.get("name") not in load_recipes():
        raise CommandError(f"unknown look: {params.get('name')}")
    try:
        spec.params_model.model_validate(params)
    except Exception as exc:
        raise CommandError(f"invalid {op} params: {exc}") from exc
    return params


class AddOpParams(BaseModel):
    media_id: str
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    index: int | None = Field(None, description="insert position; default = end")


@command("photo.add_op", AddOpParams, "Add an operation to a photo's edit stack.", "photo")
def add_op(p: Project, a: AddOpParams) -> dict:
    stack = _image(p, a.media_id)
    item = EditOp(id=new_id("op"), op=a.op, params=_validated(a.op, a.params))
    stack.insert(len(stack) if a.index is None else a.index, item)
    return {"op_id": item.id}


class SetOpParams(BaseModel):
    media_id: str
    op_id: str
    params: dict[str, Any] | None = Field(None, description="merged into existing params")
    enabled: bool | None = None


@command("photo.set_op", SetOpParams, "Change an operation's params or toggle it.", "photo")
def set_op(p: Project, a: SetOpParams) -> None:
    stack = _image(p, a.media_id)
    item = next((o for o in stack if o.id == a.op_id), None)
    if item is None:
        raise CommandError(f"op not found: {a.op_id}")
    if a.params is not None:
        item.params = _validated(item.op, {**item.params, **a.params})
    if a.enabled is not None:
        item.enabled = a.enabled


class RemoveOpParams(BaseModel):
    media_id: str
    op_id: str


@command("photo.remove_op", RemoveOpParams, "Remove an operation from the stack.", "photo")
def remove_op(p: Project, a: RemoveOpParams) -> None:
    stack = _image(p, a.media_id)
    before = len(stack)
    stack[:] = [o for o in stack if o.id != a.op_id]
    if len(stack) == before:
        raise CommandError(f"op not found: {a.op_id}")


class ResetParams(BaseModel):
    media_id: str


@command("photo.reset", ResetParams, "Remove all edits from a photo.", "photo")
def reset(p: Project, a: ResetParams) -> None:
    _image(p, a.media_id).clear()


class ApplyLookParams(BaseModel):
    media_id: str
    name: str
    strength: float = Field(1.0, ge=0, le=1.5)


@command("photo.apply_look", ApplyLookParams, "Apply a named look; replaces any previous look on the photo.", "photo")
def apply_look(p: Project, a: ApplyLookParams) -> dict:
    stack = _image(p, a.media_id)
    params = _validated("look", {"name": a.name, "strength": a.strength})
    existing = next((o for o in stack if o.op == "look"), None)
    if existing:
        existing.params, existing.enabled = params, True
        return {"op_id": existing.id}
    item = EditOp(id=new_id("op"), op="look", params=params)
    # looks go after geometry/corrections but before finishing effects the user added later
    stack.append(item)
    return {"op_id": item.id}


class ApplyProgramParams(BaseModel):
    media_id: str
    program: str = Field(description="Edit DSL text")
    mode: Literal["replace", "append"] = "replace"


@command("photo.apply_program", ApplyProgramParams, "Apply an Edit DSL program (one undo step). Used by the AI.", "photo")
def apply_program(p: Project, a: ApplyProgramParams) -> dict:
    stack = _image(p, a.media_id)
    try:
        ops = compile_program(a.program)
    except DslError as exc:
        raise CommandError(str(exc)) from exc
    new = [EditOp(id=new_id("op"), op=o["op"], params=o["params"]) for o in ops]
    if a.mode == "replace":
        stack[:] = new
    else:
        stack.extend(new)
    return {"op_ids": [o.id for o in new]}
