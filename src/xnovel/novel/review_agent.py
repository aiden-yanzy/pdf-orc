from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

REVIEW_CHECKLIST: Tuple[str, str, str] = (
    "Outline adherence",
    "Character consistency",
    "Narrative coherence",
)


class BudgetExhausted(RuntimeError):
    """Raised when the review budget is exceeded."""


@dataclass
class CostTracker:
    """Tracks and enforces an iterative review budget."""

    total_budget: Optional[float] = None
    spent: float = 0.0

    def can_spend(self, amount: float = 0.0) -> bool:
        if self.total_budget is None:
            return True
        return self.spent + amount <= self.total_budget + 1e-9

    def spend(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Cost tracker cannot spend a negative amount.")
        if self.total_budget is not None and amount > self.remaining + 1e-9:
            raise BudgetExhausted(
                f"Budget exceeded: attempted {amount}, remaining {self.remaining}."
            )
        self.spent += amount

    @property
    def remaining(self) -> float:
        if self.total_budget is None:
            return float("inf")
        return max(self.total_budget - self.spent, 0.0)


@dataclass
class CritiqueResult:
    """Represents provider feedback for a chapter iteration."""

    verdict: str
    summary: str
    issues: Sequence[str] = field(default_factory=list)
    cost: float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict.lower() == "pass"

    def to_dict(self) -> Dict[str, object]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "issues": list(self.issues),
            "cost": self.cost,
        }


@dataclass
class RewriteResult:
    """Represents a rewritten chapter after critique."""

    revised_text: str
    notes: str = ""
    cost: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "notes": self.notes,
            "cost": self.cost,
            "revised_text": self.revised_text,
        }


@dataclass
class ReviewIterationRecord:
    iteration: int
    critique: CritiqueResult
    rewrite_applied: bool = False
    revision: Optional[RewriteResult] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "iteration": self.iteration,
            "critique": self.critique.to_dict(),
            "rewrite_applied": self.rewrite_applied,
        }
        if self.revision is not None:
            payload["revision"] = self.revision.to_dict()
        else:
            payload["revision"] = None
        return payload


@dataclass
class ReviewSession:
    chapter_id: str
    initial_text: str
    final_text: str
    iterations: List[ReviewIterationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "chapter_id": self.chapter_id,
            "initial_text": self.initial_text,
            "final_text": self.final_text,
            "iterations": [record.to_dict() for record in self.iterations],
        }


@runtime_checkable
class ReviewProvider(Protocol):
    """Abstraction over an LLM provider for chapter reviews."""

    def critique(
        self,
        chapter_id: str,
        chapter_text: str,
        checklist: Sequence[str],
    ) -> CritiqueResult:
        ...

    def rewrite(
        self,
        chapter_id: str,
        chapter_text: str,
        critique: CritiqueResult,
    ) -> RewriteResult:
        ...


def _sanitize_chapter_id(chapter_id: str) -> str:
    safe = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in chapter_id]
    sanitized = "".join(safe).strip("_")
    return sanitized or "chapter"


@dataclass
class ReviewAgent:
    provider: ReviewProvider
    cost_tracker: CostTracker = field(default_factory=CostTracker)
    max_review_iters: int = 3
    logs_dir: Path = field(default_factory=lambda: Path("logs") / "reviews")
    checklist: Sequence[str] = field(default_factory=lambda: REVIEW_CHECKLIST)

    def __post_init__(self) -> None:
        if self.max_review_iters < 1:
            raise ValueError("max_review_iters must be at least 1")
        self.logs_dir = self.logs_dir.expanduser()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._history: Dict[str, ReviewSession] = {}

    @property
    def history(self) -> Dict[str, ReviewSession]:
        return self._history

    def review_chapters(self, chapters: Dict[str, str]) -> Dict[str, str]:
        updated: Dict[str, str] = {}
        for chapter_id, text in chapters.items():
            updated[chapter_id] = self.review_chapter(chapter_id, text)
        return updated

    def review_chapter(self, chapter_id: str, chapter_text: str) -> str:
        session = ReviewSession(
            chapter_id=chapter_id,
            initial_text=chapter_text,
            final_text=chapter_text,
        )
        self._history[chapter_id] = session
        current_text = chapter_text

        for iteration in range(1, self.max_review_iters + 1):
            if self.cost_tracker.total_budget is not None and self.cost_tracker.remaining <= 0:
                logger.warning(
                    "Budget exhausted before review iteration %s for chapter '%s'",
                    iteration,
                    chapter_id,
                )
                break

            critique = self.provider.critique(chapter_id, current_text, self.checklist)

            record = ReviewIterationRecord(
                iteration=iteration,
                critique=critique,
            )

            if not self.cost_tracker.can_spend(critique.cost):
                logger.warning(
                    "Budget exhausted while accounting for critique in iteration %s for chapter '%s'",
                    iteration,
                    chapter_id,
                )
                session.iterations.append(record)
                break

            self.cost_tracker.spend(critique.cost)

            if critique.passed:
                session.iterations.append(record)
                session.final_text = current_text
                break

            rewrite_result = self.provider.rewrite(chapter_id, current_text, critique)

            if not self.cost_tracker.can_spend(rewrite_result.cost):
                logger.warning(
                    "Budget exhausted before rewrite in iteration %s for chapter '%s'",
                    iteration,
                    chapter_id,
                )
                session.iterations.append(record)
                break

            self.cost_tracker.spend(rewrite_result.cost)
            record.rewrite_applied = True
            record.revision = rewrite_result
            current_text = rewrite_result.revised_text
            session.iterations.append(record)
            session.final_text = current_text

        else:
            session.final_text = current_text

        self._history[chapter_id] = session
        self._persist_session(session)
        return session.final_text

    def _persist_session(self, session: ReviewSession) -> None:
        target = self.logs_dir / f"{_sanitize_chapter_id(session.chapter_id)}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = session.to_dict()
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


__all__ = [
    "BudgetExhausted",
    "CostTracker",
    "CritiqueResult",
    "RewriteResult",
    "ReviewIterationRecord",
    "ReviewSession",
    "ReviewAgent",
    "ReviewProvider",
    "REVIEW_CHECKLIST",
]
