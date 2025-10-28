"""LangGraph-powered orchestration for short novel generation.

The orchestrator chains four stages — analysis, outline, writing, review — and
exposes modular functions so each stage can be executed independently from the
CLI. It keeps track of artefacts and cost telemetry, persisting run metadata in
``run.json`` and ``logs/cost.json``. The workflow assumes that git branch
management is handled outside the tool; developers should create or switch to
an appropriate branch before invoking these commands.
"""

from __future__ import annotations

import json
import os
import random
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

"""Lightweight orchestrator facade around the analysis agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .analysis_agent import AnalysisAgent
from .state import NovelOrchestratorState

WORKFLOW_ORDER = ["analysis", "outline", "writing", "review"]
DEFAULT_FINAL_FILENAME = "novel.md"

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


class NovelWorkflowState(TypedDict, total=False):
    """State propagated through the LangGraph workflow."""

    input_path: str
    input_text: str
    analysis_text: str
    outline_text: str
    outline_items: list[str]
    draft_text: str
    review_notes: str
    final_markdown: str


@dataclass
class NovelWorkflowConfig:
    """Configuration shared across all stages of the workflow."""

    input_path: Path
    output_dir: Path
    provider: str = "mock"
    base_url: str | None = None
    api_key_env: str | None = None
    model: str = "mock-latest"
    temperature: float = 0.7
    max_tokens: int | None = None
    budget_usd: float | None = None
    max_review_iters: int = 2
    seed: int | None = None
    chapter_range: tuple[int, int] | None = None
    context_strategy: str = "sequential"
    similarity_threshold: float = 0.35
    run_id: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S")
    )

    def __post_init__(self) -> None:
        self.input_path = self.input_path.expanduser()
        self.output_dir = self.output_dir.expanduser()


@dataclass
class ProviderResult:
    """Single model invocation record."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostEntry:
    """Aggregated telemetry per stage."""

    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp,
        }


class NovelProvider(Protocol):
    """Protocol for pluggable language-model providers."""

    def run(self, stage: str, prompt: str) -> ProviderResult:  # pragma: no cover - interface
        ...


class MockNovelProvider:
    """Deterministic provider used for testing and offline development."""

    def __init__(self, config: NovelWorkflowConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed or 0)

    def run(self, stage: str, prompt: str) -> ProviderResult:
        stage_key = stage.lower().strip()
        if stage_key == "analysis":
            content = self._build_analysis(prompt)
        elif stage_key == "outline":
            content = self._build_outline(prompt)
        elif stage_key == "writing":
            content = self._build_writing(prompt)
        elif stage_key == "review":
            content = self._build_review(prompt)
        else:  # pragma: no cover - defensive branch
            content = f"Unsupported stage '{stage}'."

        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(content)
        return ProviderResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,
            model_name=self.config.model or "mock-model",
            metadata={"provider": "mock"},
        )

    def _get_excerpt(self, text: str, limit: int = 300) -> str:
        cleaned = " ".join(text.strip().split())
        return cleaned[:limit] + ("…" if len(cleaned) > limit else "")

    def _build_analysis(self, prompt: str) -> str:
        excerpt = self._get_excerpt(prompt, limit=400)
        template = textwrap.dedent(
            """
            # Analysis Snapshot

            - Key Idea: {key}
            - Protagonist Focus: A determined voice navigating shifting circumstances.
            - Antagonistic Force: External pressure challenging personal resolve.
            - Tone & Mood: Reflective with sparks of optimism.
            - Recommended Themes: Resilience · Discovery · Quiet rebellion.
            - Opportunity Notes: Clarify stakes early and foreshadow the character's choice.
            
            > Source excerpt:
            > {excerpt}
            """
        ).strip()
        key_phrase = self._select_phrase([
            "A character confronting change",
            "An emerging mystery",
            "A quiet transformation",
            "A community under strain",
        ])
        return template.format(key=key_phrase, excerpt=excerpt or "(input not supplied)")

    def _build_outline(self, prompt: str) -> str:
        beat_templates = [
            "1. Opening Image – Establish the world and hint at disruption.",
            "2. Catalyst – Present the event that forces a decision.",
            "3. Rising Tension – The protagonist tests an imperfect strategy.",
            "4. Midpoint Shift – Reveal a deeper truth that reframes the goal.",
            "5. Climax – Confront the core conflict with a decisive action.",
            "6. Resolution – Show transformed relationships and lingering questions.",
        ]
        shuffled = beat_templates[:]
        self._rng.shuffle(shuffled)
        reordered = sorted(shuffled, key=lambda line: int(line.split(".")[0]))
        return "\n".join(reordered)

    def _build_writing(self, prompt: str) -> str:
        outline_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        chapters: list[str] = []
        for idx, line in enumerate(outline_lines, start=1):
            if not line[0].isdigit():
                continue
            title = line.split("–", 1)[0].split("-", 1)[-1].strip()
            body_seed = self._select_phrase([
                "whispers of change", "shared secrets", "flickers of hope", "unanswered letters"
            ])
            paragraph = (
                f"{body_seed.capitalize()} ripple through the scene as characters make small yet"
                f" irreversible choices. Sensory details anchor the reader in the moment while"
                f" hinting at the cost of staying the same."
            )
            chapters.append(
                textwrap.dedent(
                    f"""
                    ## Chapter {idx}: {title or f"Beat {idx}"}

                    {paragraph}
                    """
                ).strip()
            )
        if not chapters:
            chapters.append(
                textwrap.dedent(
                    """
                    ## Chapter 1: New Beginnings

                    The calm surface of daily life hides a current of change. A gentle discovery
                    nudges the narrator toward a decision that cannot be ignored.
                    """
                ).strip()
            )
        return "\n\n".join(chapters)

    def _build_review(self, prompt: str) -> str:
        draft_excerpt = self._get_excerpt(prompt, limit=500)
        notes = "\n".join(
            [
                "- Maintain consistent point of view during reflective passages.",
                "- Track the emotional aftermath of the midpoint twist.",
                "- Verify continuity of supporting character motivations.",
            ]
        )
        final_addendum = self._select_phrase([
            "The last image lingers on a symbolic gesture that signals renewal.",
            "Echoes of the opening scene create a satisfying narrative loop.",
            "A quiet epilogue hints at future adventures without closing every door.",
        ])
        review = textwrap.dedent(
            f"""
            ### Review Notes
            {notes}

            ### Final Manuscript
            {draft_excerpt}

            {final_addendum}
            """
        ).strip()
        return review

    def _select_phrase(self, options: list[str]) -> str:
        return options[self._rng.randrange(len(options))]

    def _estimate_tokens(self, text: str) -> int:
        tokens = max(1, len(text.split()))
        return tokens


class ChatOpenAINovelProvider:
    """Wrapper around ``langchain-openai`` ChatOpenAI client."""

    def __init__(self, config: NovelWorkflowConfig) -> None:
        from langchain_openai import ChatOpenAI

        api_key = None
        if config.api_key_env:
            api_key = os.getenv(config.api_key_env)
        if not api_key and config.api_key_env:
            raise RuntimeError(
                f"Environment variable '{config.api_key_env}' for API key is not set."
            )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if api_key:
            kwargs["api_key"] = api_key
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens

        self._client = ChatOpenAI(**kwargs)

    def run(self, stage: str, prompt: str) -> ProviderResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        system_message = SystemMessage(
            content=(
                "You orchestrate a multi-stage fiction workflow."
                " Respond with concise Markdown tailored to the requested stage."
            )
        )
        stage_directive = HumanMessage(content=f"Stage: {stage}\n\n{prompt}")
        response = self._client.invoke([system_message, stage_directive])
        content = self._extract_content(response).strip()
        usage = getattr(response, "usage_metadata", {}) or {}
        prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion_tokens = int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        cost = float(usage.get("total_cost") or 0.0)
        model_name = getattr(response, "model", None) or getattr(response, "model_name", "")
        return ProviderResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            model_name=model_name,
            metadata=usage,
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, list):
            pieces = [segment.get("text", "") for segment in content if isinstance(segment, dict)]
            return "".join(pieces)
        return str(content or "")


class NovelOrchestrator:
    """Coordinate the four-stage LangGraph workflow for novel generation."""

    def __init__(self, config: NovelWorkflowConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.config.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._provider = self._create_provider()
        self._workflow = self._build_workflow()

        self._stages_executed: list[str] = []
        self._cost_entries: list[CostEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_all(self) -> NovelWorkflowState:
        """Execute the full LangGraph workflow."""

        self._reset_run_tracking()
        started_at = datetime.utcnow()
        initial_state = self._initial_state()
        final_state = self._workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"novel-{self.config.run_id}"}},
        )
        finished_at = datetime.utcnow()
        self._write_run_metadata(final_state, started_at, finished_at)
        self._write_cost_log()
        return final_state

    def run_until(self, final_stage: str) -> NovelWorkflowState:
        """Run stages sequentially up to and including ``final_stage``."""

        if final_stage not in WORKFLOW_ORDER:
            raise ValueError(f"Unsupported stage '{final_stage}'.")
        cutoff = WORKFLOW_ORDER.index(final_stage) + 1
        selected = WORKFLOW_ORDER[:cutoff]

        self._reset_run_tracking()
        started_at = datetime.utcnow()
        state = self._initial_state()
        for stage in selected:
            state = self._stage_function(stage)(state)
        finished_at = datetime.utcnow()
        self._write_run_metadata(state, started_at, finished_at)
        self._write_cost_log()
        return state

    # ------------------------------------------------------------------
    # LangGraph node implementations
    # ------------------------------------------------------------------
    def analysis(self, state: NovelWorkflowState) -> NovelWorkflowState:
        self._note_stage("analysis")
        input_text = state.get("input_text") or self._load_input_text()
        prompt = self._build_analysis_prompt(input_text)
        result = self._provider.run("analysis", prompt)
        analysis_text = result.content.strip()
        if not analysis_text:
            raise RuntimeError("Analysis stage returned empty content.")

        updated = dict(state)
        updated["analysis_text"] = analysis_text
        updated["outline_items"] = self._parse_outline_items(analysis_text)

        self._register_cost("analysis", result)
        self._write_stage_output("analysis.md", analysis_text)
        return updated

    def outline(self, state: NovelWorkflowState) -> NovelWorkflowState:
        self._note_stage("outline")
        analysis_text = state.get("analysis_text")
        if not analysis_text:
            # Ensure prerequisites when running standalone
            state = self.analysis(state)
            analysis_text = state.get("analysis_text", "")

        prompt = self._build_outline_prompt(analysis_text)
        result = self._provider.run("outline", prompt)
        outline_text = result.content.strip()
        if not outline_text:
            raise RuntimeError("Outline stage returned empty content.")

        updated = dict(state)
        updated["outline_text"] = outline_text
        updated["outline_items"] = self._parse_outline_items(outline_text)

        self._register_cost("outline", result)
        self._write_stage_output("outline.md", outline_text)
        return updated

    def writing(self, state: NovelWorkflowState) -> NovelWorkflowState:
        self._note_stage("writing")
        outline_text = state.get("outline_text")
        if not outline_text:
            state = self.outline(state)
            outline_text = state.get("outline_text", "")

        prompt = self._build_writing_prompt(outline_text)
        result = self._provider.run("writing", prompt)
        draft_text = result.content.strip()
        if not draft_text:
            raise RuntimeError("Writing stage returned empty content.")

        updated = dict(state)
        updated["draft_text"] = draft_text

        self._register_cost("writing", result)
        self._write_stage_output("draft.md", draft_text)
        return updated

    def review(self, state: NovelWorkflowState) -> NovelWorkflowState:
        self._note_stage("review")
        draft_text = state.get("draft_text")
        if not draft_text:
            state = self.writing(state)
            draft_text = state.get("draft_text", "")

        prompt = self._build_review_prompt(draft_text)
        result = self._provider.run("review", prompt)
        review_text = result.content.strip()
        if not review_text:
            raise RuntimeError("Review stage returned empty content.")

        review_notes, final_markdown = self._split_review_output(review_text, draft_text)
        updated = dict(state)
        updated["review_notes"] = review_notes
        updated["final_markdown"] = final_markdown

        self._register_cost("review", result)
        self._write_stage_output("review.md", review_notes)
        self._write_stage_output(DEFAULT_FINAL_FILENAME, final_markdown)
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_run_tracking(self) -> None:
        self._stages_executed = []
        self._cost_entries = []

    def _initial_state(self) -> NovelWorkflowState:
        input_text = self._load_input_text()
        return {
            "input_path": str(self.config.input_path),
            "input_text": input_text,
        }

    def _create_provider(self) -> NovelProvider:
        provider_key = (self.config.provider or "mock").lower()
        if provider_key in {"mock", "test", "stub"} or self.config.model.lower() == "test":
            return MockNovelProvider(self.config)
        return ChatOpenAINovelProvider(self.config)

    def _build_workflow(self):
        graph = StateGraph(NovelWorkflowState)
        graph.add_node("analysis", self.analysis)
        graph.add_node("outline", self.outline)
        graph.add_node("writing", self.writing)
        graph.add_node("review", self.review)

        graph.add_edge(START, "analysis")
        graph.add_edge("analysis", "outline")
        graph.add_edge("outline", "writing")
        graph.add_edge("writing", "review")
        graph.add_edge("review", END)
        return graph.compile()

    def _stage_function(self, stage: str):
        return {
            "analysis": self.analysis,
            "outline": self.outline,
            "writing": self.writing,
            "review": self.review,
        }[stage]

    def _note_stage(self, stage: str) -> None:
        if stage not in self._stages_executed:
            self._stages_executed.append(stage)

    def _register_cost(self, stage: str, result: ProviderResult) -> None:
        entry = CostEntry(
            stage=stage,
            model=result.model_name or self.config.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
        self._cost_entries.append(entry)
        if (
            self.config.budget_usd is not None
            and self.total_cost_usd > self.config.budget_usd + 1e-9
        ):
            raise RuntimeError(
                "Budget exceeded: "
                f"${self.total_cost_usd:.4f} > ${self.config.budget_usd:.4f}"
            )

    def _write_stage_output(self, filename: str, content: str) -> Path:
        path = self.config.output_dir / filename
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return path

    def _write_run_metadata(
        self,
        state: NovelWorkflowState,
        started_at: datetime,
        finished_at: datetime,
    ) -> None:
        artefacts: dict[str, str | None] = {
            "analysis": str(self.config.output_dir / "analysis.md")
            if state.get("analysis_text")
            else None,
            "outline": str(self.config.output_dir / "outline.md")
            if state.get("outline_text")
            else None,
            "draft": str(self.config.output_dir / "draft.md")
            if state.get("draft_text")
            else None,
            "review": str(self.config.output_dir / "review.md")
            if state.get("review_notes")
            else None,
            "final": str(self.config.output_dir / DEFAULT_FINAL_FILENAME)
            if state.get("final_markdown")
            else None,
        }
        metadata = {
            "input": str(self.config.input_path),
            "output_dir": str(self.config.output_dir),
            "run_id": self.config.run_id,
            "stages": self._stages_executed[:],
            "started_at": started_at.isoformat(timespec="seconds") + "Z",
            "finished_at": finished_at.isoformat(timespec="seconds") + "Z",
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "budget_usd": self.config.budget_usd,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "max_review_iters": self.config.max_review_iters,
            "chapter_range": list(self.config.chapter_range) if self.config.chapter_range else None,
            "context_strategy": self.config.context_strategy,
            "similarity_threshold": self.config.similarity_threshold,
            "branch_policy": (
                "Create or checkout the appropriate feature branch manually before "
                "running the orchestrator."
            ),
            "artefacts": artefacts,
        }
        run_path = self.config.output_dir / "run.json"
        run_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _write_cost_log(self) -> None:
        totals = {
            "prompt_tokens": sum(entry.prompt_tokens for entry in self._cost_entries),
            "completion_tokens": sum(entry.completion_tokens for entry in self._cost_entries),
            "cost_usd": round(self.total_cost_usd, 6),
        }
        payload = {
            "entries": [entry.to_dict() for entry in self._cost_entries],
            "totals": totals,
        }
        cost_path = self.logs_dir / "cost.json"
        cost_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _load_input_text(self) -> str:
        if not self.config.input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.config.input_path}. "
                "Provide --input pointing to a Markdown or text file."
            )
        if self.config.input_path.suffix.lower() not in {".md", ".markdown", ".txt"}:
            raise ValueError(
                "Unsupported input format. Please supply a .md or .txt manuscript."
            )
        return self.config.input_path.read_text(encoding="utf-8")

    def _build_analysis_prompt(self, input_text: str) -> str:
        truncated = self._truncate_text(input_text, 4000)
        chapter_info = (
            f"Focus chapters: {self.config.chapter_range[0]}-{self.config.chapter_range[1]}"
            if self.config.chapter_range
            else "Full manuscript"
        )
        return textwrap.dedent(
            f"""
            Analyse the supplied manuscript excerpt and extract key themes, character arcs,
            and opportunities for improvement. Respond with Markdown bullet points. Include
            explicit recommendations for the next stage (outline design).

            Context strategy: {self.config.context_strategy}
            Similarity threshold: {self.config.similarity_threshold}
            {chapter_info}

            Manuscript excerpt:
            """
        ).strip() + f"\n\n{truncated}"

    def _build_outline_prompt(self, analysis_text: str) -> str:
        return textwrap.dedent(
            """
            Using the prior analysis, craft a numbered outline for the novel. Each beat should
            be on its own line using the format ``<index>. <Title> – <summary>``. Limit the
            outline to 6-8 beats and make sure it progresses logically.

            Prior analysis:
            """
        ).strip() + f"\n\n{analysis_text}"

    def _build_writing_prompt(self, outline_text: str) -> str:
        focus = (
            f"Chapters {self.config.chapter_range[0]} to {self.config.chapter_range[1]}"
            if self.config.chapter_range
            else "All chapters"
        )
        return textwrap.dedent(
            f"""
            Expand the following outline into fully fledged chapters using rich prose. Maintain
            a consistent voice and embed sensory detail. Structure the response using Markdown
            headings (## Chapter N) followed by paragraphs. Focus on: {focus}.

            Outline:
            """
        ).strip() + f"\n\n{outline_text}"

    def _build_review_prompt(self, draft_text: str) -> str:
        return textwrap.dedent(
            """
            Review the draft for continuity, tone, and pacing. Respond using Markdown with two
            sections: `### Review Notes` summarising actionable feedback, followed by
            `### Final Manuscript` containing any lightly revised text ready for publication.

            Draft:
            """
        ).strip() + f"\n\n{self._truncate_text(draft_text, 6000)}"

    def _split_review_output(self, review_text: str, draft_text: str) -> tuple[str, str]:
        marker = "### Final Manuscript"
        if marker in review_text:
            notes_part, final_part = review_text.split(marker, 1)
            review_notes = notes_part.strip()
            final_markdown = marker + "\n" + final_part.strip()
        else:
            review_notes = review_text
            final_markdown = draft_text
        return review_notes, final_markdown

    def _parse_outline_items(self, text: str) -> list[str]:
        items: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                items.append(stripped)
        return items

    @property
    def total_cost_usd(self) -> float:
        return sum(entry.cost_usd for entry in self._cost_entries)

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n…\n"


# ----------------------------------------------------------------------
# Convenience entrypoints used by the CLI layer
# ----------------------------------------------------------------------

def run_all(config: NovelWorkflowConfig) -> NovelWorkflowState:
    return NovelOrchestrator(config).run_all()


def run_analysis(config: NovelWorkflowConfig) -> NovelWorkflowState:
    return NovelOrchestrator(config).run_until("analysis")


def run_outline(config: NovelWorkflowConfig) -> NovelWorkflowState:
    return NovelOrchestrator(config).run_until("outline")


def run_writing(config: NovelWorkflowConfig) -> NovelWorkflowState:
    return NovelOrchestrator(config).run_until("writing")


def run_review(config: NovelWorkflowConfig) -> NovelWorkflowState:
    return NovelOrchestrator(config).run_until("review")
