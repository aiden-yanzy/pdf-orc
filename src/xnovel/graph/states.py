"""Typed state definitions for the xnovel LangGraph orchestrator."""

from __future__ import annotations

from typing import Any, TypedDict


class OutlineArtifacts(TypedDict):
    """Filesystem artefacts produced by the outline agent."""

    json_path: str
    markdown_path: str


class NovelPipelineState(TypedDict, total=False):
    """Shared mutable state passed between graph nodes."""

    # upstream analysis results
    analysis: dict[str, Any]

    # outline stage outputs
    outline: dict[str, Any]
    outline_artifacts: OutlineArtifacts

    # downstream placeholders (kept for future stages)
    metadata: dict[str, Any]
    errors: list[str]


__all__ = ["OutlineArtifacts", "NovelPipelineState"]
