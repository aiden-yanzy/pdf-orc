import json
import logging

import pytest

from xnovel.novel.review_agent import (
    CostTracker,
    CritiqueResult,
    ReviewAgent,
    RewriteResult,
)


class MockReviewProvider:
    def __init__(self, critiques, rewrites=None):
        self._critiques = list(critiques)
        self._rewrites = list(rewrites or [])
        self.critique_calls = []
        self.rewrite_calls = []

    def critique(self, chapter_id, chapter_text, checklist):
        idx = len(self.critique_calls)
        if idx >= len(self._critiques):
            raise AssertionError("Unexpected critique call")
        self.critique_calls.append((chapter_id, chapter_text, tuple(checklist)))
        critique = self._critiques[idx]
        return CritiqueResult(
            verdict=critique.verdict,
            summary=critique.summary,
            issues=list(critique.issues),
            cost=critique.cost,
        )

    def rewrite(self, chapter_id, chapter_text, critique):
        idx = len(self.rewrite_calls)
        if idx >= len(self._rewrites):
            raise AssertionError("Unexpected rewrite call")
        self.rewrite_calls.append((chapter_id, chapter_text))
        rewrite = self._rewrites[idx]
        return RewriteResult(
            revised_text=rewrite.revised_text,
            notes=rewrite.notes,
            cost=rewrite.cost,
        )


def test_review_agent_terminates_on_pass_without_rewrite(tmp_path):
    provider = MockReviewProvider(
        critiques=[
            CritiqueResult(verdict="pass", summary="All good", issues=[], cost=0.1),
        ]
    )
    tracker = CostTracker(total_budget=1.0)
    agent = ReviewAgent(
        provider=provider,
        cost_tracker=tracker,
        max_review_iters=3,
        logs_dir=tmp_path / "reviews",
    )

    original_text = "Original draft"
    reviewed_text = agent.review_chapter("chapter-1", original_text)

    assert reviewed_text == original_text
    assert provider.rewrite_calls == []
    assert tracker.spent == pytest.approx(0.1, rel=1e-9)
    session = agent.history["chapter-1"]
    assert len(session.iterations) == 1
    assert session.iterations[0].critique.passed is True

    log_files = list((tmp_path / "reviews").glob("*.json"))
    assert len(log_files) == 1


def test_review_agent_applies_rewrites_until_pass(tmp_path):
    provider = MockReviewProvider(
        critiques=[
            CritiqueResult(verdict="fail", summary="Outline drift", issues=["outline"], cost=0.4),
            CritiqueResult(verdict="pass", summary="Solid", issues=[], cost=0.2),
        ],
        rewrites=[
            RewriteResult(revised_text="Improved draft", notes="Tightened outline", cost=0.3),
        ],
    )

    tracker = CostTracker(total_budget=3.0)
    agent = ReviewAgent(
        provider=provider,
        cost_tracker=tracker,
        max_review_iters=3,
        logs_dir=tmp_path / "logs",
    )

    result = agent.review_chapter("chapter-2", "Working draft")

    assert result == "Improved draft"
    assert len(provider.rewrite_calls) == 1
    assert tracker.spent == pytest.approx(0.9, rel=1e-9)

    session = agent.history["chapter-2"]
    assert len(session.iterations) == 2
    assert session.iterations[0].rewrite_applied is True
    assert session.iterations[1].critique.passed is True


def test_review_agent_persists_review_log_schema(tmp_path):
    logs_dir = tmp_path / "logs" / "reviews"
    provider = MockReviewProvider(
        critiques=[
            CritiqueResult(verdict="fail", summary="Characters drift", issues=["character"], cost=0.5),
            CritiqueResult(verdict="pass", summary="Characters aligned", issues=[], cost=0.2),
        ],
        rewrites=[
            RewriteResult(revised_text="Final text", notes="Aligned arcs", cost=0.3),
        ],
    )
    tracker = CostTracker(total_budget=5.0)
    agent = ReviewAgent(
        provider=provider,
        cost_tracker=tracker,
        logs_dir=logs_dir,
    )

    chapter_id = "Chapter 1: The Beginning"
    agent.review_chapter(chapter_id, "Draft text")

    log_files = list(logs_dir.glob("*.json"))
    assert len(log_files) == 1
    data = json.loads(log_files[0].read_text(encoding="utf-8"))

    assert data["chapter_id"] == chapter_id
    assert data["initial_text"] == "Draft text"
    assert data["final_text"] == "Final text"
    assert len(data["iterations"]) == 2
    first_iter = data["iterations"][0]
    assert first_iter["critique"]["verdict"] == "fail"
    assert first_iter["rewrite_applied"] is True
    assert first_iter["revision"]["revised_text"] == "Final text"


def test_review_agent_logs_budget_exhaustion(tmp_path, caplog):
    provider = MockReviewProvider(
        critiques=[
            CritiqueResult(verdict="fail", summary="Coherence issues", issues=["coherence"], cost=0.5),
        ],
        rewrites=[
            RewriteResult(revised_text="Adjusted text", notes="", cost=0.4),
        ],
    )
    tracker = CostTracker(total_budget=0.5)
    agent = ReviewAgent(
        provider=provider,
        cost_tracker=tracker,
        logs_dir=tmp_path / "reviews",
    )

    with caplog.at_level(logging.WARNING):
        agent.review_chapter("chapter-3", "Draft")

    warnings = [record.message for record in caplog.records]
    assert any("Budget exhausted before rewrite" in message for message in warnings)
    assert len(provider.rewrite_calls) == 1
    assert tracker.spent == pytest.approx(0.5, rel=1e-9)
