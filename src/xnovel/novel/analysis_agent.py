"""LangGraph-based analysis agent for narrative manuscripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .analysis_schema import AnalysisResult
from .io import ManuscriptIO
from .providers import LLMProvider
from .state import AnalysisArtifacts, NovelOrchestratorState


def _analysis_schema_dict() -> Dict[str, Any]:
    provider = getattr(AnalysisResult, "model_json_schema", None)
    if callable(provider):
        return provider()
    legacy_provider = getattr(AnalysisResult, "schema", None)
    if callable(legacy_provider):  # pragma: no cover - fallback for Pydantic v1
        return legacy_provider()
    raise RuntimeError("Unable to obtain schema from AnalysisResult model.")


class AnalysisWorkflowState(TypedDict, total=False):
    manuscript_path: str
    output_dir: str
    seed: Optional[int]
    manuscript_text: str
    inferred_title: str
    estimated_word_count: int
    prompt_messages: Sequence[BaseMessage]
    llm_response: AIMessage
    analysis_result: AnalysisResult
    analysis_markdown_path: str
    analysis_json_path: str


@dataclass(slots=True)
class PromptMetadata:
    title: str
    estimated_word_count: int


class AnalysisPromptBuilder:
    """Assemble deterministic prompts for the analysis agent."""

    SYSTEM_PROMPT = (
        "You are an analytical fiction editor tasked with summarising narrative structure, "
        "character dynamics, and stylistic fingerprints."
    )

    def build_messages(self, manuscript_text: str, metadata: PromptMetadata) -> List[BaseMessage]:
        schema_json = json.dumps(_analysis_schema_dict(), ensure_ascii=False, indent=2)
        manuscript_excerpt = manuscript_text.strip()
        if not manuscript_excerpt:
            raise ValueError("Manuscript text is empty; unable to build prompt.")

        prompt_lines = [
            f"Title: {metadata.title}",
            f"Approximate Word Count: {metadata.estimated_word_count}",
            "\nProvide a structured critique of the manuscript. Follow these rules strictly:",
            "1. Analyse story structure and extract major plot beats (at least three).",
            "2. Identify key characters, summarise their goals, and describe relationships between them.",
            "3. Comment on stylistic choices such as tone, pacing, and notable literary devices.",
            "4. Output must be valid JSON adhering to the provided schema. Do not include markdown fences.",
            "5. If information is absent, supply an empty array or empty string rather than hallucinating.",
            "\nJSON schema:",
            schema_json,
            "\nManuscript content:",
            "```markdown",
            manuscript_excerpt,
            "```",
        ]

        user_prompt = "\n".join(prompt_lines)
        return [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]


class AnalysisAgent:
    """Executes the analysis workflow for a given manuscript."""

    def __init__(self, provider: LLMProvider, *, io_helper: Optional[ManuscriptIO] = None) -> None:
        self._provider = provider
        self._io = io_helper or ManuscriptIO()
        self._prompt_builder = AnalysisPromptBuilder()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AnalysisWorkflowState)
        graph.add_node("load_manuscript", self._node_load_manuscript)
        graph.add_node("build_prompt", self._node_build_prompt)
        graph.add_node("invoke_model", self._node_invoke_model)
        graph.add_node("persist_outputs", self._node_persist_outputs)

        graph.add_edge(START, "load_manuscript")
        graph.add_edge("load_manuscript", "build_prompt")
        graph.add_edge("build_prompt", "invoke_model")
        graph.add_edge("invoke_model", "persist_outputs")
        graph.add_edge("persist_outputs", END)
        return graph.compile()

    def run(
        self,
        manuscript_path: Path,
        output_dir: Path,
        *,
        seed: Optional[int] = None,
    ) -> NovelOrchestratorState:
        output_dir = Path(output_dir)
        initial_state: AnalysisWorkflowState = {
            "manuscript_path": str(Path(manuscript_path)),
            "output_dir": str(output_dir),
            "seed": seed,
        }
        final_state = self._graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"analysis-{Path(manuscript_path).stem}"}},
        )

        analysis_result = final_state.get("analysis_result")
        if analysis_result is None:
            raise RuntimeError("Analysis agent did not produce a result.")

        orchestrator_state = NovelOrchestratorState(
            manuscript_path=Path(final_state["manuscript_path"]),
            manuscript_text=final_state["manuscript_text"],
            seed=seed,
        )
        artifacts = AnalysisArtifacts(
            markdown_path=Path(final_state["analysis_markdown_path"]),
            json_snapshot_path=Path(final_state["analysis_json_path"]),
        )
        cost_summary = self._provider.cost_tracker.summary()
        return orchestrator_state.with_analysis(
            analysis=analysis_result,
            artifacts=artifacts,
            cost_summary=cost_summary,
        )

    # LangGraph node implementations -------------------------------------------------

    def _node_load_manuscript(self, state: AnalysisWorkflowState) -> AnalysisWorkflowState:
        manuscript_path = Path(state["manuscript_path"])
        manuscript_text = self._io.read_text(manuscript_path)
        inferred_title = self._io.infer_title(manuscript_text, manuscript_path.stem.replace("_", " ").title())
        updated = dict(state)
        updated["manuscript_text"] = manuscript_text
        updated["inferred_title"] = inferred_title
        updated["estimated_word_count"] = max(1, len(manuscript_text.split()))
        return updated

    def _node_build_prompt(self, state: AnalysisWorkflowState) -> AnalysisWorkflowState:
        metadata = PromptMetadata(
            title=state["inferred_title"],
            estimated_word_count=state["estimated_word_count"],
        )
        messages = self._prompt_builder.build_messages(state["manuscript_text"], metadata)
        updated = dict(state)
        updated["prompt_messages"] = messages
        return updated

    def _node_invoke_model(self, state: AnalysisWorkflowState) -> AnalysisWorkflowState:
        messages = state.get("prompt_messages")
        if not messages:
            raise RuntimeError("Prompt messages missing before invoking LLM.")
        seed = state.get("seed") if self._provider.supports_seed else None
        response = self._provider.invoke(messages, seed=seed)
        analysis = self._parse_response(response)
        updated = dict(state)
        updated["llm_response"] = response
        updated["analysis_result"] = analysis
        return updated

    def _node_persist_outputs(self, state: AnalysisWorkflowState) -> AnalysisWorkflowState:
        analysis = state.get("analysis_result")
        if analysis is None:
            raise RuntimeError("Analysis result missing; cannot persist outputs.")
        output_dir = self._io.ensure_directory(Path(state["output_dir"]))
        markdown_path = output_dir / "analysis.md"
        self._io.write_markdown(markdown_path, analysis.to_markdown())

        run_log_dir = self._io.ensure_run_log_directory(output_dir)
        json_path = run_log_dir / "analysis.json"
        payload_builder = getattr(analysis, "model_dump", None)
        if callable(payload_builder):
            payload = payload_builder(mode="json")
        else:  # pragma: no cover - fallback for older pydantic
            payload = analysis.dict()
        self._io.write_json(json_path, payload)

        updated = dict(state)
        updated["analysis_markdown_path"] = str(markdown_path)
        updated["analysis_json_path"] = str(json_path)
        return updated

    # Helpers ----------------------------------------------------------------------

    def _parse_response(self, response: AIMessage) -> AnalysisResult:
        content = response.content
        if isinstance(content, list):
            text_chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_chunks.append(str(item["text"]))
                else:
                    text_chunks.append(str(item))
            content_str = "".join(text_chunks)
        else:
            content_str = str(content)

        content_str = content_str.strip()
        if not content_str:
            raise ValueError("LLM returned empty analysis content.")

        content_str = _strip_code_fence(content_str)
        try:
            payload = json.loads(content_str)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing path
            raise ValueError(f"Failed to parse LLM output as JSON: {exc}") from exc

        validator = getattr(AnalysisResult, "model_validate", None)
        if callable(validator):
            return validator(payload)
        return AnalysisResult.parse_obj(payload)  # type: ignore[return-value]  # pragma: no cover - pydantic v1 fallback


def _strip_code_fence(payload: str) -> str:
    stripped = payload.strip()
    if stripped.startswith("```json"):
        inner = stripped[len("```json") :].strip()
        if inner.endswith("```"):
            inner = inner[: -len("```")]
        return inner.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped[3:-3]
        return inner.strip()
    return stripped
