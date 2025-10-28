"""Path helpers for the xnovel agent pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "DEFAULT_INPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "NovelPathConfig",
    "resolve_input_path",
    "resolve_output_path",
    "ensure_directory",
]

DEFAULT_INPUT_ROOT = Path(os.getenv("XNOVEL_INPUT_ROOT", "inputs")) / "novel"
DEFAULT_OUTPUT_ROOT = Path(os.getenv("XNOVEL_OUTPUT_ROOT", "outputs")) / "novel"


def _normalise(path: Path | str) -> Path:
    return Path(path).expanduser()


def ensure_directory(path: Path | str) -> Path:
    resolved = _normalise(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_input_path(path: Path | str | None = None, *, create: bool = False) -> Path:
    candidate = _normalise(path or DEFAULT_INPUT_ROOT)
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def resolve_output_path(path: Path | str | None = None, *, create: bool = True) -> Path:
    candidate = _normalise(path or DEFAULT_OUTPUT_ROOT)
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


@dataclass(slots=True)
class NovelPathConfig:
    """Dataclass mirroring the existing configuration style for new agents."""

    input_path: Path = DEFAULT_INPUT_ROOT
    output_path: Path = DEFAULT_OUTPUT_ROOT
    create_input: bool = False
    create_output: bool = True

    def expanded(self) -> "NovelPathConfig":
        return replace(
            self,
            input_path=_normalise(self.input_path),
            output_path=_normalise(self.output_path),
        )

    def ensure(self) -> "NovelPathConfig":
        resolved = self.expanded()
        if self.create_input:
            resolved.input_path.mkdir(parents=True, exist_ok=True)
        if self.create_output:
            resolved.output_path.mkdir(parents=True, exist_ok=True)
        return resolved

    def as_tuple(self) -> tuple[Path, Path]:
        resolved = self.expanded()
        return resolved.input_path, resolved.output_path
