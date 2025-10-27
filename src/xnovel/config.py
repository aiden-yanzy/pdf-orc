"""Dataclass-driven configuration scaffolding for the xnovel package."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .paths import NovelPathConfig, resolve_input_path, resolve_output_path

__all__ = [
    "LLMConfig",
    "BudgetConfig",
    "XNovelConfig",
]


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:  # pragma: no cover - defensive guard
        return default


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - defensive guard
        return default


@dataclass(slots=True)
class LLMConfig:
    """Configuration for LangChain-backed chat providers."""

    model: str = os.getenv("XNOVEL_MODEL", "gpt-4o-mini")
    base_url: str | None = os.getenv("XNOVEL_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    temperature: float = _env_float("XNOVEL_TEMPERATURE", 0.0) or 0.0
    max_tokens: int | None = _env_int("XNOVEL_MAX_TOKENS")
    api_key_env: str = os.getenv("XNOVEL_API_KEY_ENV", "XNOVEL_API_KEY")
    fallback_api_key_envs: tuple[str, ...] = ("OPENAI_API_KEY",)

    def resolve_api_key(self, override: str | None = None) -> str | None:
        if override:
            return override
        env_candidates: Iterable[str | None] = (self.api_key_env, *self.fallback_api_key_envs)
        for name in env_candidates:
            if not name:
                continue
            value = os.getenv(name)
            if value:
                return value
        return None

    def provider_kwargs(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object | None]:
        return {
            "model": model or self.model,
            "base_url": base_url or self.base_url,
            "api_key": api_key if api_key is not None else self.resolve_api_key(),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }


@dataclass(slots=True)
class BudgetConfig:
    """Budget guardrails for downstream agent execution."""

    limit_usd: float | None = _env_float("XNOVEL_BUDGET_USD")
    warn_ratio: float = _env_float("XNOVEL_BUDGET_WARN_RATIO", 0.9) or 0.9
    hard_limit: bool = os.getenv("XNOVEL_BUDGET_HARD", "false").lower() == "true"

    def should_warn(self, spent: float) -> bool:
        if self.limit_usd is None:
            return False
        return spent >= self.limit_usd * self.warn_ratio

    def is_exceeded(self, spent: float) -> bool:
        if self.limit_usd is None:
            return False
        return spent >= self.limit_usd and self.hard_limit


@dataclass(slots=True)
class XNovelConfig:
    """Primary configuration entry point for the novel agent pipeline."""

    paths: NovelPathConfig = field(default_factory=NovelPathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    track_costs: bool = True

    def with_paths(self, *, input_path: Path | str | None = None, output_path: Path | str | None = None) -> "XNovelConfig":
        new_paths = replace(
            self.paths,
            input_path=resolve_input_path(input_path or self.paths.input_path, create=self.paths.create_input),
            output_path=resolve_output_path(output_path or self.paths.output_path, create=self.paths.create_output),
        )
        return replace(self, paths=new_paths)

    @property
    def input_path(self) -> Path:
        return resolve_input_path(self.paths.input_path, create=self.paths.create_input)

    @property
    def output_path(self) -> Path:
        return resolve_output_path(self.paths.output_path, create=self.paths.create_output)

    def ensure_directories(self) -> "XNovelConfig":
        self.paths = self.paths.ensure()
        return self

    def as_provider_kwargs(self, **overrides: object) -> dict[str, object | None]:
        return self.llm.provider_kwargs(**overrides)
