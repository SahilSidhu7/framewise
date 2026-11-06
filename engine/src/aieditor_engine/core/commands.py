"""Command registry.

Every edit, whether it comes from a GUI click or from EditLM, is a named
command with a Pydantic params model. The params schema is the single
definition reused for validation, GUI forms and the AI's DSL grammar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .project import Project


class CommandError(Exception):
    """A command was rejected (unknown name, bad params, or invalid state)."""


CommandFn = Callable[[Project, Any], dict[str, Any] | None]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    params_model: type[BaseModel]
    fn: CommandFn
    description: str
    category: str

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "params": self.params_model.model_json_schema(),
        }


_REGISTRY: dict[str, CommandSpec] = {}


def command(
    name: str, params: type[BaseModel], description: str, category: str = "general"
) -> Callable[[CommandFn], CommandFn]:
    """Register a command. The function mutates the project it is given.

    The bus passes a private copy, so a command that raises leaves the real
    project untouched (commands are atomic without manual rollback).
    """

    def decorator(fn: CommandFn) -> CommandFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate command: {name}")
        _REGISTRY[name] = CommandSpec(name, params, fn, description, category)
        return fn

    return decorator


def get_command(name: str) -> CommandSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise CommandError(f"unknown command: {name}") from None


def list_commands() -> list[dict[str, Any]]:
    return [spec.schema() for spec in sorted(_REGISTRY.values(), key=lambda s: s.name)]
