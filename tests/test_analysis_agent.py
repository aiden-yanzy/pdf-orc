from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from xnovel.novel.analysis_agent import AnalysisAgent, AnalysisPromptBuilder, PromptMetadata
from xnovel.novel.analysis_schema import AnalysisResult
from xnovel.novel.orchestrator import AnalysisOrchestrator, AnalysisOrchestratorConfig
from xnovel.novel.providers import LLMProvider


class MockLLMProvider(LLMProvider):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__()
        self._payload = payload

    @property
    def supports_seed(self) -> bool:
        return True

    def invoke(self, messages: Sequence[BaseMessage], *, seed: int | None = None) -> AIMessage:
        response = AIMessage(
            content=json.dumps(self._payload, ensure_ascii=False),
            usage_metadata={
                "input_tokens": 80,
                "output_tokens": 40,
                "total_tokens": 120,
                "model": "mock-model",
            },
        )
        self._record_usage(
            model="mock-model",
            prompt_tokens=80,
            completion_tokens=40,
            total_tokens=120,
            cost=0.05,
            metadata={"seed": seed},
        )
        return response


def test_prompt_builder_includes_schema():
    builder = AnalysisPromptBuilder()
    metadata = PromptMetadata(title="Test Title", estimated_word_count=123)
    manuscript_text = "# Test Title\nA brief novella synopsis."

    messages = builder.build_messages(manuscript_text, metadata)

    assert len(messages) == 2
    assert messages[0].content == builder.SYSTEM_PROMPT
    user_prompt = messages[1].content
    assert "JSON schema" in user_prompt
    assert '"plot_beats"' in user_prompt
    assert "Provide a structured critique" in user_prompt
    assert "```markdown" in user_prompt
    assert "Approximate Word Count: 123" in user_prompt


def test_analysis_agent_persists_markdown_and_tracks_cost(tmp_path: Path):
    manuscript = tmp_path / "sample.md"
    manuscript.write_text(
        "# The Lantern Road\n\nThe journey begins in a quiet village.",
        encoding="utf-8",
    )

    payload = {
        "title": "The Lantern Road",
        "overview": "A traveller seeks the source of the lanterns leading home.",
        "plot_beats": [
            {
                "order": 1,
                "title": "Departure",
                "summary": "Protagonist leaves the village seeking answers.",
                "impact": "Establishes stakes and longing.",
            },
            {
                "order": 2,
                "title": "Trials",
                "summary": "Encounters tests of courage in the haunted forest.",
                "impact": "Deepens character resilience and mythology.",
            },
            {
                "order": 3,
                "title": "Revelation",
                "summary": "Learns the lanterns were lit by loved ones guiding her home.",
                "impact": "Resolves emotional arc and theme of belonging.",
            },
        ],
        "characters": [
            {
                "name": "Mira",
                "role": "Protagonist",
                "traits": ["Curious", "Resilient"],
                "goals": ["Understand the lantern trail"],
                "relationships": [
                    {
                        "target": "Elder Rowan",
                        "nature": "Mentor offering cryptic advice",
                        "arc": "Trust grows as Mira learns to listen.",
                    }
                ],
            },
            {
                "name": "Elder Rowan",
                "role": "Mentor",
                "traits": ["Wise"],
                "goals": [],
                "relationships": [],
            },
        ],
        "stylistic_notes": [
            {
                "aspect": "Tone",
                "observation": "Gentle, folkloric narration that emphasises wonder.",
                "evidence": "Soft imagery describes lantern light trembling like fireflies.",
            }
        ],
        "themes": ["Belonging", "Homecoming"],
    }

    provider = MockLLMProvider(payload)
    agent = AnalysisAgent(provider)
    orchestrator = AnalysisOrchestrator(agent)
    config = AnalysisOrchestratorConfig(manuscript_path=manuscript, output_dir=tmp_path / "outputs", seed=11)

    state = orchestrator.run(config)

    artifacts = state.analysis_artifacts
    assert artifacts is not None
    markdown_path = artifacts.markdown_path
    json_path = artifacts.json_snapshot_path

    assert markdown_path.exists()
    markdown_content = markdown_path.read_text(encoding="utf-8")
    assert "## Plot Beats" in markdown_content
    assert "## Character Roster" in markdown_content
    assert "## Stylistic Observations" in markdown_content
    assert "> Soft imagery describes lantern light trembling like fireflies." in markdown_content

    assert json_path.exists()
    logged_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert logged_payload == payload

    assert isinstance(state.analysis, AnalysisResult)
    assert state.analysis.title == payload["title"]
    assert state.cost_summary["calls"] == 1
    assert state.cost_summary["prompt_tokens"] == 80
    assert state.cost_summary["completion_tokens"] == 40
    assert state.cost_summary["cost"] == pytest.approx(0.05, abs=0.001)
