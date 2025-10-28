"""Novel analysis workflow components."""

from .analysis_agent import AnalysisAgent, AnalysisPromptBuilder, PromptMetadata
from .analysis_schema import (
    AnalysisResult,
    CharacterProfile,
    NarrativeRelationship,
    PlotBeat,
    StylisticNote,
)
from .orchestrator import AnalysisOrchestrator, AnalysisOrchestratorConfig
from .state import AnalysisArtifacts, NovelOrchestratorState
  
"""Novel-specific agents and helpers."""

from .outline_agent import (
    OutlineAgent,
    OutlineConfig,
    OutlineValidationError,
    generate_character_aliases,
    remap_location,
    remap_setting,
    remap_timeframe,
)

__all__ = [
    "AnalysisAgent",
    "AnalysisPromptBuilder",
    "PromptMetadata",
    "AnalysisResult",
    "CharacterProfile",
    "NarrativeRelationship",
    "PlotBeat",
    "StylisticNote",
    "AnalysisOrchestrator",
    "AnalysisOrchestratorConfig",
    "AnalysisArtifacts",
    "NovelOrchestratorState",
   "OutlineAgent",
    "OutlineConfig",
    "OutlineValidationError",
    "generate_character_aliases",
    "remap_location",
    "remap_setting",
    "remap_timeframe",
]
