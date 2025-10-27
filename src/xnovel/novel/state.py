"""Orchestrator state definitions for analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .analysis_schema import AnalysisResult


@dataclass(slots=True)
class AnalysisArtifacts:
    """Filesystem artefacts produced by the analysis agent."""

    markdown_path: Path
    json_snapshot_path: Path


@dataclass(slots=True)
class NovelOrchestratorState:
    """State container that downstream stages can rely on."""

    manuscript_path: Path
    manuscript_text: str
    analysis: Optional[AnalysisResult] = None
    analysis_artifacts: Optional[AnalysisArtifacts] = None
    cost_summary: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def with_analysis(
        self,
        *,
        analysis: AnalysisResult,
        artifacts: AnalysisArtifacts,
        cost_summary: dict[str, Any],
    ) -> "NovelOrchestratorState":
        self.analysis = analysis
        self.analysis_artifacts = artifacts
        self.cost_summary = cost_summary
        return self
