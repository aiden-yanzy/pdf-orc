"""LangChain chat provider abstraction for the novel agent pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple

from langchain_core.messages import BaseMessage

try:  # pragma: no cover - import guard for optional dependency
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - gracefully degrade when dependency missing
    ChatOpenAI = None  # type: ignore[assignment]

__all__ = [
    "ProviderError",
    "ProviderDependencyError",
    "ProviderSettings",
    "LangChainChatProvider",
    "build_provider",
]

MessagesLike = Sequence[BaseMessage] | Sequence[Mapping[str, Any]]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MODEL_ENVS: Tuple[str, ...] = ("XNOVEL_MODEL", "OPENAI_MODEL")
DEFAULT_API_KEY_ENVS: Tuple[str, ...] = ("XNOVEL_API_KEY", "OPENAI_API_KEY")
DEFAULT_BASE_URL_ENVS: Tuple[str, ...] = ("XNOVEL_BASE_URL", "OPENAI_BASE_URL")
DEFAULT_TEMPERATURE_ENV = "XNOVEL_TEMPERATURE"
DEFAULT_MAX_TOKEN_ENV = "XNOVEL_MAX_TOKENS"


class ProviderError(RuntimeError):
    """Base error raised when interacting with a chat provider."""


class ProviderDependencyError(ProviderError):
    """Raised when required dependencies are unavailable."""


@dataclass(slots=True)
class ProviderSettings:
    """Mutable settings bundle for a chat provider."""

    model: str = DEFAULT_MODEL
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float | None = None

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return kwargs


class LangChainChatProvider:
    """Thin wrapper around ``langchain_openai.ChatOpenAI`` with safeguards."""

    def __init__(self, settings: ProviderSettings):
        if ChatOpenAI is None:
            raise ProviderDependencyError(
                "langchain-openai is required to instantiate LangChainChatProvider"
            )
        self.settings = settings
        self._client = self._build_client(settings)

    def _build_client(self, settings: ProviderSettings):
        try:
            return ChatOpenAI(**settings.as_kwargs())  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - passthrough
            raise ProviderError(f"Failed to initialise chat model '{settings.model}': {exc}") from exc

    @property
    def model(self) -> str:
        return self.settings.model

    def invoke(self, messages: MessagesLike, **kwargs: Any) -> Any:
        try:
            return self._client.invoke(messages, **kwargs)
        except Exception as exc:  # pragma: no cover - passthrough
            raise ProviderError(f"Invocation failed for model '{self.settings.model}': {exc}") from exc

    def stream(self, messages: MessagesLike, **kwargs: Any):
        try:
            return self._client.stream(messages, **kwargs)
        except Exception as exc:  # pragma: no cover - passthrough
            raise ProviderError(f"Streaming failed for model '{self.settings.model}': {exc}") from exc

    def with_model(self, model: str, **overrides: Any) -> "LangChainChatProvider":
        new_settings = replace(self.settings, model=model)
        for key, value in overrides.items():
            if hasattr(new_settings, key):
                setattr(new_settings, key, value)
        return self.__class__(new_settings)

    def swap_model(self, model: str, **overrides: Any) -> "LangChainChatProvider":
        """Alias for :meth:`with_model` to improve readability."""

        return self.with_model(model, **overrides)


def build_provider(
    *,
    model: str | None = None,
    model_envs: Sequence[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    api_key_envs: Sequence[str] | None = None,
    base_url_envs: Sequence[str] | None = None,
) -> LangChainChatProvider:
    """Factory that mirrors CLI/env resolution for provider credentials."""

    model_env_value = _resolve_from_env(model_envs or DEFAULT_MODEL_ENVS)
    resolved_model = model or model_env_value or DEFAULT_MODEL
    resolved_base_url = base_url or _resolve_from_env(base_url_envs or DEFAULT_BASE_URL_ENVS)
    resolved_api_key = api_key or _resolve_from_env(api_key_envs or DEFAULT_API_KEY_ENVS)

    resolved_temperature = _coerce_float(temperature, os.getenv(DEFAULT_TEMPERATURE_ENV), default=0.0)
    resolved_max_tokens = _coerce_int(max_tokens, os.getenv(DEFAULT_MAX_TOKEN_ENV))

    settings = ProviderSettings(
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        timeout=timeout,
    )
    return LangChainChatProvider(settings)



def _resolve_from_env(envs: Sequence[str]) -> str | None:
    for env_name in envs:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def _coerce_float(explicit: float | None, env_value: str | None, *, default: float) -> float:
    if explicit is not None:
        return explicit
    if env_value is None:
        return default
    try:
        return float(env_value)
    except ValueError:  # pragma: no cover - defensive guard
        return default


def _coerce_int(explicit: int | None, env_value: str | None) -> int | None:
    if explicit is not None:
        return explicit
    if env_value is None:
        return None
    try:
        return int(env_value)
    except ValueError:  # pragma: no cover - defensive guard
        return None
