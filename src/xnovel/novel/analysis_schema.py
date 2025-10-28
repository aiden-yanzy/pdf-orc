"""Structured schema definitions for novel analysis outputs."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

try:  # pragma: no cover - compatibility helper
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - pydantic v1 fallback
    ConfigDict = None  # type: ignore


class FrozenBaseModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="ignore", frozen=True)
    else:  # pragma: no cover - pydantic v1 fallback
        class Config:
            extra = "ignore"
            allow_mutation = False


class NarrativeRelationship(FrozenBaseModel):
    """Represents an inter-character relationship."""

    target: str = Field(..., description="Name of the related character.")
    nature: str = Field(..., description="Short description of how the characters relate.")
    arc: Optional[str] = Field(
        default=None,
        description="Optional notes about how the relationship evolves across the narrative.",
    )


class CharacterProfile(FrozenBaseModel):
    """Structured description of a primary or supporting character."""

    name: str = Field(..., description="Canonical name of the character.")
    role: str = Field(..., description="Narrative role, e.g., protagonist, antagonist, mentor.")
    traits: List[str] = Field(default_factory=list, description="Key personality traits or attributes.")
    goals: List[str] = Field(default_factory=list, description="Primary objectives or motivations driving the character.")
    relationships: List[NarrativeRelationship] = Field(
        default_factory=list,
        description="List of notable relationships to other characters.",
    )


class PlotBeat(FrozenBaseModel):
    """A single beat within the story structure."""

    order: int = Field(..., description="Position of the beat within the narrative arc (1-indexed).")
    title: str = Field(..., description="Concise label for the beat.")
    summary: str = Field(..., description="What happens during this beat.")
    impact: str = Field(
        ..., description="Why this beat matters for characters, stakes, or thematic progression."
    )


class StylisticNote(FrozenBaseModel):
    """Observations about tone, pacing, or other stylistic attributes."""

    aspect: str = Field(..., description="Aspect being observed, e.g., tone, pacing, narrative voice.")
    observation: str = Field(..., description="What is happening stylistically.")
    evidence: Optional[str] = Field(
        default=None,
        description="Optional supporting quote or scene reference backing the observation.",
    )


class AnalysisResult(FrozenBaseModel):
    """Aggregate structure representing the LLM-backed manuscript analysis."""

    title: str = Field(..., description="Title inferred for the manuscript.")
    overview: str = Field(..., description="High-level synopsis or critique of the manuscript.")
    plot_beats: List[PlotBeat] = Field(default_factory=list, description="Ordered list of major plot developments.")
    characters: List[CharacterProfile] = Field(
        default_factory=list,
        description="Roster of notable characters along with traits and relationships.",
    )
    stylistic_notes: List[StylisticNote] = Field(
        default_factory=list,
        description="Observations about style, tone, pacing, or authorial voice.",
    )
    themes: List[str] = Field(default_factory=list, description="Key themes reinforced throughout the narrative.")

    def sorted_plot_beats(self) -> List[PlotBeat]:
        """Return plot beats ordered by the declared position."""

        return sorted(self.plot_beats, key=lambda beat: beat.order)

    def to_markdown(self) -> str:
        """Serialize the analysis result into a Markdown document."""

        lines: list[str] = []
        lines.append(f"# Analysis Overview — {self.title}\n")
        lines.append(self.overview.strip())
        lines.append("")

        if self.themes:
            lines.append("## Core Themes")
            for theme in self.themes:
                lines.append(f"- {theme}")
            lines.append("")

        if self.plot_beats:
            lines.append("## Plot Beats")
            for beat in self.sorted_plot_beats():
                lines.append(f"### {beat.order}. {beat.title}")
                lines.append(beat.summary.strip())
                lines.append("")
                lines.append(f"_Impact_: {beat.impact.strip()}")
                lines.append("")

        if self.characters:
            lines.append("## Character Roster")
            for character in self.characters:
                lines.append(f"### {character.name}")
                lines.append(f"- Role: {character.role}")
                if character.traits:
                    lines.append(f"- Traits: {', '.join(character.traits)}")
                if character.goals:
                    lines.append(f"- Goals: {', '.join(character.goals)}")
                if character.relationships:
                    lines.append("- Relationships:")
                    for relation in character.relationships:
                        relation_line = f"  - {relation.target}: {relation.nature}"
                        if relation.arc:
                            relation_line += f" (Arc: {relation.arc})"
                        lines.append(relation_line)
                lines.append("")

        if self.stylistic_notes:
            lines.append("## Stylistic Observations")
            for note in self.stylistic_notes:
                lines.append(f"### {note.aspect}")
                lines.append(note.observation.strip())
                if note.evidence:
                    lines.append("")
                    lines.append(f"> {note.evidence.strip()}")
                lines.append("")

        return "\n".join(line.rstrip() for line in lines if line is not None).strip() + "\n"
