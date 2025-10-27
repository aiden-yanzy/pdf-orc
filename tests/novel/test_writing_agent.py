from __future__ import annotations

from pathlib import Path

import pytest

from xnovel.novel.writing_agent import (
    GenerationResult,
    OutlineNode,
    SimpleCostTracker,
    WritingAgent,
    WritingAgentConfig,
    sanitize_filename,
    split_into_sections,
)


class MockProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate(self, messages, *, max_tokens: int, temperature: float) -> GenerationResult:
        if not self._responses:
            raise AssertionError("No mock responses remaining")
        content = self._responses.pop(0)
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return GenerationResult(
            content=content,
            prompt_tokens=120,
            completion_tokens=240,
            cost_usd=0.05,
            model_name="mock-model",
        )


class MockEmbedder:
    """Embedding backend that favours overlaps on specific keywords."""

    def embed_documents(self, texts: list[str]):
        return [self._vectorise(text) for text in texts]

    def embed_query(self, text: str):
        return self._vectorise(text)

    @staticmethod
    def _vectorise(text: str) -> list[float]:
        text_lower = text.lower()
        vector = [0.0, 0.0, 0.0]
        if "dragon" in text_lower:
            vector[0] += 1.0
        if "castle" in text_lower:
            vector[1] += 1.0
        vector[2] = len(text) / 1000.0
        return vector


def test_sanitize_filename_handles_special_characters():
    assert sanitize_filename("  The Emperor's Revenge?!  ") == "the-emperors-revenge"
    assert sanitize_filename("章节 01 / 引言") == "章节-01-引言"


def test_split_into_sections_respects_token_limit():
    text = ("First scene introduces stakes. " * 40) + "\n\n" + ("Second scene escalating conflict. " * 40)
    sections = split_into_sections(text, max_tokens=60)
    assert len(sections) >= 2
    assert all(section.strip() for section in sections)


def test_writing_agent_generates_ordered_chapters_and_updates_costs(tmp_path: Path):
    responses = [
        "Opening chapter content with rising dawn.",
        "Second chapter builds on the journey.",
        "Final chapter resolves the arc.",
    ]
    provider = MockProvider(responses)
    tracker = SimpleCostTracker()
    config = WritingAgentConfig(
        output_dir=tmp_path / "chapters",
        max_generation_tokens=512,
        section_token_limit=200,
        context_strategy="window",
    )
    agent = WritingAgent(provider, config=config, cost_tracker=tracker)

    outline = [
        OutlineNode(index=1, title="Opening Gambit", summary="Introduces the hero."),
        OutlineNode(index=2, title="Rising Conflict", summary="Complications emerge."),
        OutlineNode(index=3, title="Climax & Aftermath", summary="The arc concludes."),
    ]

    paths = agent.write_chapters(outline)

    assert [p.name for p in paths] == [
        "1-opening-gambit.md",
        "2-rising-conflict.md",
        "3-climax-aftermath.md",
    ]
    for path in paths:
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# ")

    assert len(provider.calls) == 3
    assert all(call["max_tokens"] == 512 for call in provider.calls)
    assert tracker.total_cost_usd == pytest.approx(0.15, rel=1e-6)
    assert len(agent.run_log) == 3
    assert [entry["chapter_index"] for entry in agent.run_log] == [1, 2, 3]


def test_retrieve_strategy_falls_back_without_embeddings(tmp_path: Path):
    provider = MockProvider([
        "First chapter text about landscapes.",
        "Second chapter referencing journeys.",
    ])
    config = WritingAgentConfig(
        output_dir=tmp_path / "chapters",
        context_strategy="retrieve",
        max_generation_tokens=256,
        section_token_limit=128,
    )
    agent = WritingAgent(provider, config=config, cost_tracker=SimpleCostTracker())

    outline = [
        OutlineNode(index=1, title="Arrival", summary=""),
        OutlineNode(index=2, title="Departure", summary=""),
    ]

    agent.write_chapters(outline)

    assert agent.run_log[1]["context_strategy"] == "window-fallback"
    second_call = provider.calls[1]
    assert "First chapter text" in second_call["messages"][1]["content"]


def test_retrieve_strategy_uses_embeddings_when_available(tmp_path: Path):
    provider = MockProvider([
        "A dragon watches from the castle walls.",
        "Allies remember the dragon's warning amid castle ruins.",
    ])
    embedder = MockEmbedder()
    config = WritingAgentConfig(
        output_dir=tmp_path / "chapters",
        context_strategy="retrieve",
        max_generation_tokens=256,
        section_token_limit=128,
    )
    agent = WritingAgent(
        provider,
        config=config,
        cost_tracker=SimpleCostTracker(),
        embedder=embedder,
    )

    outline = [
        OutlineNode(index=1, title="Dragon's Vigil", summary="The dragon watches."),
        OutlineNode(index=2, title="Council at the Castle", summary="Allies meet the dragon."),
    ]

    agent.write_chapters(outline)

    assert agent.run_log[1]["context_strategy"] == "retrieve"
    second_call = provider.calls[1]
    assert "dragon" in second_call["messages"][1]["content"].lower()
