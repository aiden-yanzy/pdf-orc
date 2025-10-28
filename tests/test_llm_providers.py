from __future__ import annotations

import pytest

from xnovel.llm.providers import (
    LangChainChatProvider,
    ProviderDependencyError,
    ProviderSettings,
    build_provider,
)


def test_build_provider_prefers_explicit_over_env(monkeypatch: pytest.MonkeyPatch, dummy_chat_model) -> None:
    monkeypatch.setenv("XNOVEL_MODEL", "env-model")
    monkeypatch.setenv("XNOVEL_BASE_URL", "https://env.example")
    monkeypatch.setenv("XNOVEL_API_KEY", "env-key")
    monkeypatch.setenv("XNOVEL_TEMPERATURE", "0.25")
    monkeypatch.setenv("XNOVEL_MAX_TOKENS", "512")

    provider = build_provider(
        model="cli-model",
        temperature=0.9,
        max_tokens=2048,
        timeout=30.0,
    )

    assert isinstance(provider, LangChainChatProvider)
    assert provider.model == "cli-model"
    settings = provider.settings
    assert settings.base_url == "https://env.example"
    assert settings.api_key == "env-key"
    assert settings.temperature == 0.9
    assert settings.max_tokens == 2048
    assert settings.timeout == 30.0

    dummy_instance = provider._client  # type: ignore[attr-defined]
    assert dummy_instance.kwargs["model"] == "cli-model"
    assert dummy_instance.kwargs["temperature"] == 0.9


def test_build_provider_uses_env_fallbacks_when_not_overridden(monkeypatch: pytest.MonkeyPatch, dummy_chat_model) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "fallback-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fallback.example")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    monkeypatch.setenv("XNOVEL_TEMPERATURE", "0.1")

    provider = build_provider()

    assert provider.model == "fallback-model"
    settings = provider.settings
    assert settings.base_url == "https://fallback.example"
    assert settings.api_key == "fallback-key"
    assert settings.temperature == 0.1
    assert settings.max_tokens is None


def test_build_provider_raises_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from xnovel import llm

    monkeypatch.setattr(llm.providers, "ChatOpenAI", None)
    with pytest.raises(ProviderDependencyError):
        build_provider()


def test_provider_settings_as_kwargs_filters_none() -> None:
    settings = ProviderSettings(model="demo", temperature=0.5, max_tokens=None, timeout=None)
    kwargs = settings.as_kwargs()
    assert kwargs == {"model": "demo", "temperature": 0.5}


def test_langchain_chat_provider_propagates_invocation(monkeypatch: pytest.MonkeyPatch, dummy_chat_model) -> None:
    provider = build_provider(model="demo-model")
    response = provider.invoke([{"role": "user", "content": "hi"}], test=True)

    assert response["messages"][0]["content"] == "hi"
    dummy_instance = provider._client  # type: ignore[attr-defined]
    assert dummy_instance.invocations[0][0] == "invoke"

    stream_iter = provider.stream([1, 2, 3])
    assert list(stream_iter) == [1, 2, 3]
    assert dummy_instance.invocations[1][0] == "stream"
