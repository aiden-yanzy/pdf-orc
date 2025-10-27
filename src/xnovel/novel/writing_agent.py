"""Utilities for generating novel chapters from outline nodes.

This module provides a configurable writing agent that can iteratively
produce chapter drafts while keeping track of cost and context. Three
context management strategies are supported:

``window``
    Maintains a sliding window of prior chapter outputs capped by a token
    budget.
``summary``
    Stores rolling summaries for each generated chapter and surfaces the
    most recent ones to the language model.
``retrieve``
    Uses lightweight embeddings to retrieve relevant prior beats or
    generated sections. When embeddings are unavailable the agent falls
    back to the sequential ``window`` strategy.

The agent immediately persists each generated chapter to disk, capturing
metadata per chapter so the surrounding workflow can inspect or resume
its progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import re
from typing import Literal, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Token accounting inside the agent uses a coarse heuristic compatible
# with GPT-style tokenisation where ~4 characters ~= 1 token.
AVG_CHARS_PER_TOKEN = 4
DEFAULT_SECTION_TOKEN_LIMIT = 800
SUMMARY_CHAR_LIMIT = 220
VALID_FILENAME_PATTERN = re.compile(r"[^\w\-\s]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"[\s\-]+")


def estimate_tokens(text: str) -> int:
    """Return a rough token count for the supplied text.

    The heuristic intentionally overestimates long passages so the agent
    remains within the configured budgets when operating without a true
    tokenizer.
    """

    if not text:
        return 0
    return max(1, math.ceil(len(text) / AVG_CHARS_PER_TOKEN))


def sanitize_filename(title: str) -> str:
    """Sanitise a title for filesystem usage.

    Invalid characters are removed, whitespace is normalised to single
    hyphens and the result is lower-cased. When the input collapses to an
    empty string a default placeholder is returned.
    """

    cleaned = VALID_FILENAME_PATTERN.sub("", title).strip()
    cleaned = WHITESPACE_PATTERN.sub("-", cleaned.lower())
    cleaned = cleaned.strip("-_")
    return cleaned or "chapter"


def split_into_sections(text: str, max_tokens: int = DEFAULT_SECTION_TOKEN_LIMIT) -> list[str]:
    """Split *text* into sections constrained by *max_tokens*.

    Paragraph boundaries are preferred; when a paragraph alone exceeds
    ``max_tokens`` the function falls back to sentence level splitting.
    Consecutive separators are discarded so the returned sections are
    always non-empty.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    stripped = text.strip()
    if not stripped:
        return [""]

    if estimate_tokens(stripped) <= max_tokens:
        return [stripped]

    sections: list[str] = []
    current: list[str] = []
    current_tokens = 0

    paragraphs = [p.strip() for p in stripped.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > max_tokens:
            # Flush any accumulated context before handling the oversized
            # paragraph at sentence granularity.
            if current:
                sections.append("\n\n".join(current).strip())
                current = []
                current_tokens = 0

            sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", paragraph) if s.strip()]
            sentence_buffer: list[str] = []
            sentence_tokens = 0
            for sentence in sentences:
                tokens = estimate_tokens(sentence)
                if sentence_tokens + tokens > max_tokens and sentence_buffer:
                    sections.append(" ".join(sentence_buffer).strip())
                    sentence_buffer = [sentence]
                    sentence_tokens = tokens
                else:
                    sentence_buffer.append(sentence)
                    sentence_tokens += tokens

            if sentence_buffer:
                sections.append(" ".join(sentence_buffer).strip())
            continue

        if current_tokens + paragraph_tokens > max_tokens and current:
            sections.append("\n\n".join(current).strip())
            current = [paragraph]
            current_tokens = paragraph_tokens
        else:
            current.append(paragraph)
            current_tokens += paragraph_tokens

    if current:
        sections.append("\n\n".join(current).strip())

    return sections or [stripped]


@dataclass(slots=True)
class OutlineNode:
    """Minimal structure required to describe a chapter outline entry."""

    index: int
    title: str
    summary: str = ""
    beats: Sequence[str] = field(default_factory=list)
    metadata: dict[str, object] | None = None


BaseStrategy = Literal["window", "summary", "retrieve"]
ContextStrategy = Literal["window", "summary", "retrieve", "window-fallback"]


@dataclass
class WritingAgentConfig:
    """Runtime configuration for the writing agent."""

    temperature: float = 0.7
    chapter_start: int = 1
    chapter_end: int | None = None
    context_strategy: BaseStrategy = "window"
    max_context_tokens: int = 1600
    max_generation_tokens: int = 800
    section_token_limit: int = DEFAULT_SECTION_TOKEN_LIMIT
    embedding_top_k: int = 5
    usd_budget: float | None = None
    output_dir: Path = Path("chapters")

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir = self.output_dir.expanduser()
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.max_generation_tokens <= 0:
            raise ValueError("max_generation_tokens must be positive")
        if self.section_token_limit <= 0:
            raise ValueError("section_token_limit must be positive")


@dataclass
class GenerationResult:
    """Container for LLM responses used by the writing agent."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = 0.0
    model_name: str | None = None


class LanguageModelProtocol(Protocol):
    """Protocol expected from the LLM provider."""

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        """Return a ``GenerationResult`` for the supplied conversation."""


class CostTrackerProtocol(Protocol):
    """Protocol for updating token/cost accounting."""

    @property
    def total_cost_usd(self) -> float:
        """Return the accumulated USD cost."""

    def add_call(
        self,
        *,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Record a single LLM invocation."""


class EmbeddingBackendProtocol(Protocol):
    """Protocol for the lightweight embedding backend used in retrieval mode."""

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Embed a single query string."""


@dataclass(slots=True)
class ChapterRecord:
    """Keeps track of generated chapter artefacts for later context building."""

    index: int
    title: str
    text: str
    sections: list[str]
    summary: str
    file_path: Path


@dataclass(slots=True)
class EmbeddingRecord:
    """Association between a generated section and its embedding."""

    chapter_index: int
    vector: np.ndarray
    text: str
    tokens: int


@dataclass(slots=True)
class ContextBundle:
    """Represents the context supplied to the LLM for the next chapter."""

    snippets: list[str]
    strategy: ContextStrategy


class SimpleCostTracker(CostTrackerProtocol):
    """Light-weight cost tracker used when the caller does not supply one."""

    def __init__(self, budget_usd: float | None = None) -> None:
        self._total_cost = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self.budget_usd = budget_usd

    @property
    def total_cost_usd(self) -> float:  # pragma: no cover - simple accessors
        return self._total_cost

    def add_call(
        self,
        *,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._total_cost += cost_usd
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens


class WritingAgent:
    """Generate novel chapters from outline nodes using an LLM provider."""

    def __init__(
        self,
        provider: LanguageModelProtocol,
        *,
        config: WritingAgentConfig | None = None,
        cost_tracker: CostTrackerProtocol | None = None,
        embedder: EmbeddingBackendProtocol | None = None,
    ) -> None:
        self.config = config or WritingAgentConfig()
        self.provider = provider
        self.embedder = embedder

        if cost_tracker is None:
            cost_tracker = SimpleCostTracker(budget_usd=self.config.usd_budget)
            self._own_cost_tracker = True
        else:
            self._own_cost_tracker = False
        self.cost_tracker = cost_tracker

        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._generated_chapters: list[ChapterRecord] = []
        self._embeddings: list[EmbeddingRecord] = []
        self._summaries: dict[int, str] = {}
        self.run_log: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_chapters(self, outline: Sequence[OutlineNode]) -> list[Path]:
        """Generate chapter drafts for the supplied *outline*.

        Returns a list of file paths corresponding to generated chapters.
        The outline is processed in index order while respecting the
        configured ``chapter_start`` / ``chapter_end`` bounds and optional
        USD budget.
        """

        written_paths: list[Path] = []
        nodes = sorted(outline, key=lambda node: node.index)

        for node in nodes:
            if node.index < self.config.chapter_start:
                continue
            if self.config.chapter_end is not None and node.index > self.config.chapter_end:
                break

            if self._budget_remaining() is not None and self._budget_remaining() <= 0:
                self._log_budget_exit(node.index)
                break

            context_bundle = self._build_context_bundle(node)
            messages = self._build_messages(node, context_bundle)

            result = self.provider.generate(
                messages,
                max_tokens=self.config.max_generation_tokens,
                temperature=self.config.temperature,
            )

            sections = split_into_sections(result.content, self.config.section_token_limit)
            file_path = self._write_chapter_to_disk(node, sections)
            summary = self._summarise_text(result.content)

            record = ChapterRecord(
                index=node.index,
                title=node.title,
                text=result.content,
                sections=sections,
                summary=summary,
                file_path=file_path,
            )
            self._generated_chapters.append(record)
            self._summaries[node.index] = summary
            self._index_embeddings(node.index, sections)

            metadata = {
                "chapter_index": node.index,
                "title": node.title,
                "file_path": str(file_path),
                "context_strategy": context_bundle.strategy,
                "sections": len(sections),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cost_usd": result.cost_usd,
            }
            self.run_log.append(metadata)

            self._register_cost(result, node)

            written_paths.append(file_path)

            if self._budget_remaining() is not None and self._budget_remaining() <= 0:
                self._log_budget_exit(node.index)
                break

        return written_paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _budget_limit(self) -> float | None:
        if self.config.usd_budget is not None:
            return self.config.usd_budget
        if hasattr(self.cost_tracker, "budget_usd"):
            return getattr(self.cost_tracker, "budget_usd")  # type: ignore[return-value]
        return None

    def _current_cost(self) -> float:
        return float(getattr(self.cost_tracker, "total_cost_usd", 0.0)) if self.cost_tracker else 0.0

    def _budget_remaining(self) -> float | None:
        limit = self._budget_limit()
        if limit is None:
            return None
        return limit - self._current_cost()

    def _log_budget_exit(self, chapter_index: int) -> None:
        limit = self._budget_limit()
        if limit is None:
            return
        total_cost = self._current_cost()
        logger.warning(
            "Budget exhausted before chapter %s (spent %.4f USD of %.4f USD).",
            chapter_index,
            total_cost,
            limit,
        )
        self.run_log.append(
            {
                "event": "budget-exhausted",
                "chapter_index": chapter_index,
                "total_cost_usd": total_cost,
                "budget_usd": limit,
            }
        )

    def _register_cost(self, result: GenerationResult, node: OutlineNode) -> None:
        tracker = self.cost_tracker
        if tracker is None:
            return

        payload = {
            "model": result.model_name,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "cost_usd": result.cost_usd,
            "metadata": {
                "chapter_index": node.index,
                "chapter_title": node.title,
            },
        }

        for method_name in ("add_call", "log_call", "add"):
            method = getattr(tracker, method_name, None)
            if callable(method):
                try:
                    method(**payload)  # type: ignore[arg-type]
                except TypeError:
                    fallback_payload = dict(payload)
                    fallback_payload.pop("metadata", None)
                    method(**fallback_payload)  # type: ignore[arg-type]
                return

        if hasattr(tracker, "total_cost_usd"):
            current = getattr(tracker, "total_cost_usd")
            try:
                setattr(tracker, "total_cost_usd", current + result.cost_usd)
            except Exception:  # pragma: no cover - defensive path
                logger.debug("Cost tracker does not expose a writable total; skipping update.")

    def _build_context_bundle(self, node: OutlineNode) -> ContextBundle:
        strategy = (self.config.context_strategy or "window").lower()
        if strategy == "summary":
            return ContextBundle(self._summary_context(), "summary")
        if strategy == "retrieve":
            snippets, used_strategy = self._retrieve_context(node)
            return ContextBundle(snippets, used_strategy)
        return ContextBundle(self._window_context(), "window")

    def _window_context(self) -> list[str]:
        snippets: list[str] = []
        accumulated = 0
        for chapter in reversed(self._generated_chapters):
            chapter_tokens = estimate_tokens(chapter.text)
            if accumulated + chapter_tokens > self.config.max_context_tokens and snippets:
                break
            snippets.append(chapter.text)
            accumulated += chapter_tokens
        snippets.reverse()
        return snippets

    def _summary_context(self) -> list[str]:
        snippets: list[str] = []
        accumulated = 0
        for index in sorted(self._summaries.keys(), reverse=True):
            summary = self._summaries[index]
            tokens = estimate_tokens(summary)
            if accumulated + tokens > self.config.max_context_tokens and snippets:
                continue
            snippets.append(summary)
            accumulated += tokens
        snippets.reverse()
        return snippets

    def _retrieve_context(self, node: OutlineNode) -> tuple[list[str], ContextStrategy]:
        if not self.embedder or not self._embeddings:
            return self._window_context(), "window-fallback"

        query_text = " ".join(filter(None, [node.summary, " ".join(node.beats), node.title])).strip()
        if not query_text:
            query_text = f"Chapter {node.index}"

        try:
            query_vector = np.asarray(self.embedder.embed_query(query_text), dtype=float)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Embedding query failed: %s", exc)
            return self._window_context(), "window-fallback"

        if query_vector.size == 0:
            return self._window_context(), "window-fallback"

        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0:
            return self._window_context(), "window-fallback"

        scored: list[tuple[float, EmbeddingRecord]] = []
        for record in self._embeddings:
            denom = query_norm * float(np.linalg.norm(record.vector))
            similarity = float(np.dot(query_vector, record.vector) / denom) if denom else 0.0
            scored.append((similarity, record))

        scored.sort(key=lambda item: item[0], reverse=True)

        max_k = max(1, self.config.embedding_top_k)
        snippets: list[str] = []
        accumulated = 0
        for similarity, record in scored[:max_k]:
            if similarity <= 0 and snippets:
                break
            if accumulated + record.tokens > self.config.max_context_tokens and snippets:
                continue
            snippets.append(record.text)
            accumulated += record.tokens

        if not snippets:
            return self._window_context(), "window-fallback"
        return snippets, "retrieve"

    def _build_messages(self, node: OutlineNode, bundle: ContextBundle) -> list[dict[str, str]]:
        context_lines: list[str] = []
        if bundle.snippets:
            context_lines.append("Context from previous chapters:")
            for snippet in bundle.snippets:
                context_lines.append(snippet.strip())
                context_lines.append("")

        outline_lines = [f"Chapter {node.index}: {node.title}"]
        if node.summary:
            outline_lines.append(f"Summary: {node.summary}")
        if node.beats:
            outline_lines.append("Beats:")
            outline_lines.extend([f"- {beat}" for beat in node.beats])

        outline_lines.append("\nWrite the full chapter prose with clear scene transitions.")

        user_content = "\n".join(context_lines + outline_lines).strip()

        system_message = (
            "You are an experienced fiction ghostwriter. Maintain tone and continuity "
            "across chapters while elaborating on provided beats."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def _summarise_text(self, text: str) -> str:
        collapsed = " ".join(text.split())
        if not collapsed:
            return ""
        if len(collapsed) <= SUMMARY_CHAR_LIMIT:
            return collapsed

        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", collapsed) if s.strip()]
        summary: list[str] = []
        total_length = 0
        for sentence in sentences:
            summary.append(sentence)
            total_length += len(sentence)
            if total_length >= SUMMARY_CHAR_LIMIT:
                break
        return " ".join(summary).strip()

    def _write_chapter_to_disk(self, node: OutlineNode, sections: Sequence[str]) -> Path:
        filename = f"{node.index}-{sanitize_filename(node.title)}.md"
        file_path = self.output_dir / filename

        lines: list[str] = [f"# {node.title or f'Chapter {node.index}'}", ""]
        if len(sections) == 1:
            lines.append(sections[0].strip())
        else:
            for idx, section in enumerate(sections, start=1):
                lines.append(f"## Part {idx}")
                lines.append("")
                lines.append(section.strip())
                if idx != len(sections):
                    lines.append("")

        file_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return file_path

    def _index_embeddings(self, chapter_index: int, sections: Sequence[str]) -> None:
        if not self.embedder:
            return
        try:
            vectors = self.embedder.embed_documents(list(sections))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Embedding generation failed: %s", exc)
            return

        for text, vector in zip(sections, vectors):
            np_vector = np.asarray(vector, dtype=float)
            if np_vector.size == 0:
                continue
            self._embeddings.append(
                EmbeddingRecord(
                    chapter_index=chapter_index,
                    vector=np_vector,
                    text=text,
                    tokens=estimate_tokens(text),
                )
            )

    # ------------------------------------------------------------------
    # Convenience accessors (primarily useful in tests)
    # ------------------------------------------------------------------

    @property
    def generated_chapters(self) -> Sequence[ChapterRecord]:
        return tuple(self._generated_chapters)

    @property
    def summaries(self) -> dict[int, str]:
        return dict(self._summaries)

    def __del__(self) -> None:  # pragma: no cover - safety for file handles
        if getattr(self, "_own_cost_tracker", False):
            # Nothing to clean up explicitly, but keeps symmetrical ownership
            # semantics should the simple tracker ever need resources.
            self.cost_tracker = None


__all__ = [
    "ContextBundle",
    "CostTrackerProtocol",
    "GenerationResult",
    "LanguageModelProtocol",
    "OutlineNode",
    "SimpleCostTracker",
    "WritingAgent",
    "WritingAgentConfig",
    "sanitize_filename",
    "split_into_sections",
]
