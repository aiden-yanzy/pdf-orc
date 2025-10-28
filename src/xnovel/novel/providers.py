"""LLM provider abstractions with cost tracking support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage


@dataclass(frozen=True)
class CostRecord:
    """Single invocation usage metrics."""

    model: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: Optional[float]
    metadata: Dict[str, Any]


class CostTracker:
    """Accumulates token usage and cost data across model calls."""

    def __init__(self) -> None:
        self._records: list[CostRecord] = []

    @property
    def records(self) -> Sequence[CostRecord]:  # pragma: no cover - simple accessor
        return tuple(self._records)

    def add_record(
        self,
        *,
        model: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        record = CostRecord(
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total),
            cost=cost,
            metadata=metadata or {},
        )
        self._records.append(record)

    def total_prompt_tokens(self) -> int:
        return sum(record.prompt_tokens for record in self._records)

    def total_completion_tokens(self) -> int:
        return sum(record.completion_tokens for record in self._records)

    def total_cost(self) -> float:
        return float(sum(record.cost or 0.0 for record in self._records))

    def summary(self) -> Dict[str, Any]:
        return {
            "calls": len(self._records),
            "prompt_tokens": self.total_prompt_tokens(),
            "completion_tokens": self.total_completion_tokens(),
            "total_tokens": sum(record.total_tokens for record in self._records),
            "cost": self.total_cost(),
            "by_model": self._aggregate_by_model(),
        }

    def _aggregate_by_model(self) -> Dict[str, Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        for record in self._records:
            model_key = record.model or "unknown"
            bucket = aggregated.setdefault(
                model_key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                },
            )
            bucket["calls"] += 1
            bucket["prompt_tokens"] += record.prompt_tokens
            bucket["completion_tokens"] += record.completion_tokens
            bucket["total_tokens"] += record.total_tokens
            bucket["cost"] += record.cost or 0.0
        return aggregated


class LLMProvider(ABC):
    """Abstract interface that analysis components use to communicate with an LLM."""

    def __init__(self, *, cost_tracker: Optional[CostTracker] = None) -> None:
        self._cost_tracker = cost_tracker or CostTracker()

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    @property
    def supports_seed(self) -> bool:  # pragma: no cover - interface default
        return False

    @abstractmethod
    def invoke(self, messages: Sequence[BaseMessage], *, seed: Optional[int] = None) -> AIMessage:
        """Invoke the chat model and return the generated message."""

    def _record_usage(
        self,
        *,
        model: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int],
        cost: Optional[float],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        self._cost_tracker.add_record(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            metadata=metadata,
        )


class LangChainLLMProvider(LLMProvider):
    """Adapter around a `langchain` chat model with usage extraction."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        cost_tracker: Optional[CostTracker] = None,
        name: Optional[str] = None,
        supports_seed: Optional[bool] = None,
    ) -> None:
        super().__init__(cost_tracker=cost_tracker)
        self._llm = llm
        self._name = name or getattr(llm, "model_name", llm.__class__.__name__)
        default_seed_support = hasattr(llm, "supports_seed") and bool(getattr(llm, "supports_seed"))
        self._supports_seed = supports_seed if supports_seed is not None else default_seed_support

    @property
    def supports_seed(self) -> bool:
        return self._supports_seed

    def invoke(self, messages: Sequence[BaseMessage], *, seed: Optional[int] = None) -> AIMessage:
        config: Dict[str, Any] = {}
        if seed is not None and self.supports_seed:
            config.setdefault("configurable", {})["seed"] = seed

        response: AIMessage
        if config:
            response = self._llm.invoke(messages, config=config)
        else:  # pragma: no cover - configless path is trivial
            response = self._llm.invoke(messages)

        usage, metadata = _extract_usage_metadata(response)
        self._record_usage(
            model=usage.get("model") or getattr(response, "model", self._name),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
            cost=usage.get("cost"),
            metadata=metadata,
        )
        return response


def _extract_usage_metadata(message: AIMessage) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Try to normalise usage metadata from different langchain providers."""

    usage: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    if getattr(message, "usage_metadata", None):
        usage.update(message.usage_metadata)  # type: ignore[arg-type]

    response_meta = getattr(message, "response_metadata", None) or {}
    if isinstance(response_meta, dict):
        metadata.update(response_meta)
        if "token_usage" in response_meta and not usage:
            maybe_usage = response_meta.get("token_usage")
            if isinstance(maybe_usage, dict):
                usage.update(maybe_usage)
        if "usage" in response_meta and not usage:
            maybe_usage = response_meta.get("usage")
            if isinstance(maybe_usage, dict):
                usage.update(maybe_usage)
        if "model_name" in response_meta and "model" not in usage:
            usage["model"] = response_meta.get("model_name")
        if "cost" in response_meta and "cost" not in usage:
            usage["cost"] = response_meta.get("cost")

    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    if isinstance(additional_kwargs, dict) and not usage:
        maybe_usage = additional_kwargs.get("usage") or additional_kwargs.get("token_usage")
        if isinstance(maybe_usage, dict):
            usage.update(maybe_usage)

    normalised_usage: Dict[str, Any] = {}
    normalised_usage["prompt_tokens"] = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("promptTokens")
        or 0
    )
    normalised_usage["completion_tokens"] = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("completionTokens")
        or usage.get("generated_tokens")
        or 0
    )
    total_tokens = usage.get("total_tokens") or usage.get("totalTokens")
    if total_tokens is not None:
        normalised_usage["total_tokens"] = int(total_tokens)
    cost_value = usage.get("cost") or usage.get("total_cost") or usage.get("expense")
    if cost_value is not None:
        try:
            normalised_usage["cost"] = float(cost_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    if "model" in usage:
        normalised_usage["model"] = usage.get("model")

    return normalised_usage, metadata
