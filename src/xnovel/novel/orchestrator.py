"""Lightweight orchestrator facade around the analysis agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .analysis_agent import AnalysisAgent
from .state import NovelOrchestratorState


@dataclass(slots=True)
class AnalysisOrchestratorConfig:
    manuscript_path: Path
    output_dir: Path
    seed: Optional[int] = None


class AnalysisOrchestrator:
    """Entry point used by integration tests or CLI stubs."""

    def __init__(self, agent: AnalysisAgent) -> None:
        self._agent = agent

    def run(self, config: AnalysisOrchestratorConfig) -> NovelOrchestratorState:
        return self._agent.run(
            manuscript_path=config.manuscript_path,
            output_dir=config.output_dir,
            seed=config.seed,
        )
