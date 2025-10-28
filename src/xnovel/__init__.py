"""Scaffolding package for the novel agent pipeline."""

from .config import BudgetConfig, LLMConfig, XNovelConfig
from .io import LoadedDocument, load_input_resource
from .paths import NovelPathConfig, resolve_input_path, resolve_output_path
  
"""xnovel package exposing novel analysis utilities."""

from .novel.analysis_agent import AnalysisAgent, PromptMetadata
from .novel.orchestrator import AnalysisOrchestrator, AnalysisOrchestratorConfig
from .novel.state import AnalysisArtifacts, NovelOrchestratorState

__all__ = [
    "BudgetConfig",
    "LLMConfig",
    "XNovelConfig",
    "NovelPathConfig",
    "resolve_input_path",
    "resolve_output_path",
    "LoadedDocument",
    "load_input_resource",
    "AnalysisAgent",
    "PromptMetadata",
    "AnalysisOrchestrator",
    "AnalysisOrchestratorConfig",
    "AnalysisArtifacts",
    "NovelOrchestratorState",
]
