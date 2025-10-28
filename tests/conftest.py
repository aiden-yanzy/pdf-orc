"""Shared fixtures for the test suite."""
from __future__ import annotations

from typing import Any, Iterable

import pytest

from xnovel.llm.cost import CostTracker, ModelPricing

ENV_VARS = {
    "XNOVEL_MODEL",
    "OPENAI_MODEL",
    "XNOVEL_API_KEY",
    "OPENAI_API_KEY",
    "XNOVEL_BASE_URL",
    "OPENAI_BASE_URL",
    "XNOVEL_TEMPERATURE",
    "XNOVEL_MAX_TOKENS",
}


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LLM-related environment variables do not leak between tests."""

    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def dummy_chat_model(monkeypatch: pytest.MonkeyPatch):
    """Patch the LangChain chat client used by the provider abstraction."""

    from xnovel.llm import providers

    class DummyChatModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.invocations: list[tuple[str, tuple[Iterable[Any], dict[str, Any]]]] = []

        def invoke(self, messages: Iterable[Any], **kwargs: Any) -> dict[str, Any]:
            record = ("invoke", (tuple(messages), dict(kwargs)))
            self.invocations.append(record)
            return {"messages": list(messages), **kwargs}

        def stream(self, messages: Iterable[Any], **kwargs: Any):
            record = ("stream", (tuple(messages), dict(kwargs)))
            self.invocations.append(record)
            for message in messages:
                yield message

    monkeypatch.setattr(providers, "ChatOpenAI", DummyChatModel)
    return DummyChatModel


@pytest.fixture
def dummy_cost_tracker() -> CostTracker:
    """Provide a cost tracker with deterministic pricing for tests."""

    pricing = {
        "stub-model": ModelPricing(prompt_per_1k=0.001, completion_per_1k=0.002),
        "alt-model": ModelPricing(prompt_per_1k=0.01, completion_per_1k=0.02),
    }
    return CostTracker(pricing=pricing, budget_limit=5.0, warn_ratio=0.5)

