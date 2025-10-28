"""Token accounting helpers and pricing models for LLM usage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Mapping

__all__ = [
    "MODEL_PRICING",
    "ModelPricing",
    "TokenUsage",
    "CostSnapshot",
    "BudgetExceededError",
    "CostTracker",
    "register_model_pricing",
]


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Price definition expressed in USD per 1K tokens."""

    prompt_per_1k: float
    completion_per_1k: float
    currency: str = "USD"

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1000) * self.prompt_per_1k
        completion_cost = (completion_tokens / 1000) * self.completion_per_1k
        return prompt_cost + completion_cost


MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(prompt_per_1k=0.00015, completion_per_1k=0.0006),
    "gpt-4o": ModelPricing(prompt_per_1k=0.005, completion_per_1k=0.015),
    "o4-mini": ModelPricing(prompt_per_1k=0.0005, completion_per_1k=0.0015),
    "claude-3-7-sonnet-20250219": ModelPricing(prompt_per_1k=0.003, completion_per_1k=0.015),
}


@dataclass(slots=True)
class TokenUsage:
    """Mutable token tally used internally by :class:`CostTracker`."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost

    def to_dict(self) -> dict[str, float | int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass(frozen=True, slots=True)
class CostSnapshot:
    """Immutable summary of a single accounting event."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    currency: str = "USD"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "currency": self.currency,
        }


class BudgetExceededError(RuntimeError):
    """Raised when recorded spend breaches a configured hard limit."""


@dataclass(slots=True)
class CostTracker:
    """Running accounting helper with optional budget enforcement."""

    pricing: Mapping[str, ModelPricing] = field(default_factory=lambda: MODEL_PRICING)
    budget_limit: float | None = None
    warn_ratio: float = 0.9
    currency: str = "USD"
    _usage: Dict[str, TokenUsage] = field(default_factory=dict, init=False, repr=False)
    _total_cost: float = field(default=0.0, init=False, repr=False)

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> CostSnapshot:
        pricing = self.pricing.get(model)
        cost = pricing.estimate_cost(prompt_tokens, completion_tokens) if pricing else 0.0
        usage = self._usage.setdefault(model, TokenUsage())
        usage.add(prompt_tokens, completion_tokens, cost)
        self._total_cost += cost

        if self.budget_limit is not None and self._total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit {self.budget_limit:.2f} {self.currency} exceeded: {self._total_cost:.2f}"
            )

        return CostSnapshot(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            currency=self.currency,
        )

    def should_warn(self) -> bool:
        if self.budget_limit is None:
            return False
        return self._total_cost >= self.budget_limit * self.warn_ratio

    def reset(self) -> None:
        self._usage.clear()
        self._total_cost = 0.0

    def usage_for(self, model: str) -> TokenUsage | None:
        return self._usage.get(model)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_cost": round(self._total_cost, 6),
            "currency": self.currency,
            "budget_limit": self.budget_limit,
            "warn_ratio": self.warn_ratio,
            "usage": {model: usage.to_dict() for model, usage in sorted(self._usage.items())},
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def remaining_budget(self) -> float | None:
        if self.budget_limit is None:
            return None
        return max(self.budget_limit - self._total_cost, 0.0)


def register_model_pricing(model: str, *, prompt_per_1k: float, completion_per_1k: float, currency: str = "USD") -> None:
    """Register or override pricing details for a model in ``MODEL_PRICING``."""

    MODEL_PRICING[model] = ModelPricing(
        prompt_per_1k=prompt_per_1k,
        completion_per_1k=completion_per_1k,
        currency=currency,
    )
