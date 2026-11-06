"""Command bus: validation, atomic execution, undo/redo, trajectory log, events."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .commands import CommandError, get_command
from .project import Project, utc_now

Source = Literal["user", "ai", "system"]
EventListener = Callable[[dict[str, Any]], None]


@dataclass
class _HistoryEntry:
    label: str
    snapshot: Project  # project state *before* the command (undo) or after (redo)


class CommandBus:
    """Owns the live project. All mutations go through here.

    Thread-safe: the API server and background jobs (proxy generation) may
    call in from different threads.
    """

    def __init__(
        self,
        project: Project,
        *,
        log_path: Path | None = None,
        history_limit: int = 200,
    ) -> None:
        self._project = project
        self._undo: list[_HistoryEntry] = []
        self._redo: list[_HistoryEntry] = []
        self._history_limit = history_limit
        self._log_path = log_path
        self._listeners: list[EventListener] = []
        self._lock = threading.RLock()
        self._seq = 0
        self.revision = 0
        self.dirty = False

    # ------------------------------------------------------------------ state
    @property
    def project(self) -> Project:
        return self._project

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self.revision,
                "can_undo": bool(self._undo),
                "can_redo": bool(self._redo),
                "undo_label": self._undo[-1].label if self._undo else None,
                "redo_label": self._redo[-1].label if self._redo else None,
                "project": self._project.model_dump(mode="json"),
            }

    @property
    def lock(self) -> threading.RLock:
        """Hold while reading the project from another thread (e.g. saving)."""
        return self._lock

    def emit(self, event: dict[str, Any]) -> None:
        """Publish a non-state event (progress, notices) to listeners."""
        self._emit(event)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    # -------------------------------------------------------------- execution
    def execute(
        self, name: str, params: dict[str, Any] | None = None, *, source: Source = "user"
    ) -> dict[str, Any]:
        spec = get_command(name)
        raw = params or {}
        with self._lock:
            try:
                parsed = spec.params_model.model_validate(raw)
            except ValidationError as exc:
                self._log(source, name, raw, ok=False, error=_short_validation(exc))
                raise CommandError(f"invalid params for {name}: {_short_validation(exc)}") from exc

            working = self._project.model_copy(deep=True)
            try:
                result = spec.fn(working, parsed) or {}
            except CommandError as exc:
                self._log(source, name, raw, ok=False, error=str(exc))
                raise
            except (KeyError, ValueError) as exc:
                msg = exc.args[0] if exc.args else str(exc)
                self._log(source, name, raw, ok=False, error=str(msg))
                raise CommandError(str(msg)) from exc

            self._undo.append(_HistoryEntry(name, self._project))
            if len(self._undo) > self._history_limit:
                self._undo.pop(0)
            self._redo.clear()
            self._project = working
            self._changed()
            self._log(source, name, parsed.model_dump(mode="json"), ok=True, result=result)

        self._emit({"type": "command", "name": name, "source": source, "result": result})
        return result

    def undo(self) -> bool:
        with self._lock:
            if not self._undo:
                return False
            entry = self._undo.pop()
            self._redo.append(_HistoryEntry(entry.label, self._project))
            self._project = entry.snapshot
            self._changed()
        self._emit({"type": "undo", "name": entry.label})
        return True

    def redo(self) -> bool:
        with self._lock:
            if not self._redo:
                return False
            entry = self._redo.pop()
            self._undo.append(_HistoryEntry(entry.label, self._project))
            self._project = entry.snapshot
            self._changed()
        self._emit({"type": "redo", "name": entry.label})
        return True

    def patch_media(self, media_id: str, **fields: Any) -> None:
        """Non-undoable system update of derived media data (proxy, thumbnail).

        Applied to the live project and every history snapshot, so undo/redo
        never resurrects a stale proxy status.
        """
        with self._lock:
            states = [self._project, *(e.snapshot for e in self._undo), *(e.snapshot for e in self._redo)]
            touched = False
            for state in states:
                item = state.media.get(media_id)
                if item is not None:
                    for key, value in fields.items():
                        setattr(item, key, value)
                    touched = True
            if not touched:
                return
            self._changed()
        self._emit({"type": "media_updated", "media_id": media_id, "fields": _jsonable(fields)})

    # ---------------------------------------------------------------- helpers
    def _changed(self) -> None:
        self.revision += 1
        self.dirty = True

    def _emit(self, event: dict[str, Any]) -> None:
        event = {**event, "revision": self.revision}
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:  # a broken listener must never break editing
                pass

    def _log(self, source: Source, name: str, params: Any, *, ok: bool, **extra: Any) -> None:
        """Append to the trajectory log (JSONL). This is future EditLM training data."""
        if self._log_path is None:
            return
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": utc_now(),
            "source": source,
            "command": name,
            "params": params,
            "ok": ok,
            **_jsonable(extra),
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _short_validation(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'params'}: {err['msg']}" for err in exc.errors()
    )


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))
