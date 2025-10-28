from __future__ import annotations

from pathlib import Path

import pytest

from xnovel.config import LLMConfig, XNovelConfig
from xnovel.llm.cost import BudgetExceededError


def test_llm_config_resolve_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = LLMConfig(api_key_env="CUSTOM", fallback_api_key_envs=("OPENAI_API_KEY",))

    assert cfg.resolve_api_key(override="override-key") == "override-key"

    monkeypatch.setenv("CUSTOM", "primary-key")
    assert cfg.resolve_api_key() == "primary-key"

    monkeypatch.delenv("CUSTOM")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    assert cfg.resolve_api_key() == "fallback-key"


def test_llm_config_provider_kwargs_merge() -> None:
    cfg = LLMConfig(
        model="base-model",
        base_url="https://example.com",
        temperature=0.3,
        max_tokens=256,
    )
    kwargs = cfg.provider_kwargs(api_key="inline-key", temperature=0.8)

    assert kwargs["model"] == "base-model"
    assert kwargs["base_url"] == "https://example.com"
    assert kwargs["api_key"] == "inline-key"
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 256


def test_budget_config_thresholds(dummy_cost_tracker) -> None:
    tracker = dummy_cost_tracker
    assert tracker.warn_ratio == 0.5

    snapshot = tracker.record("stub-model", prompt_tokens=1000, completion_tokens=1000)
    assert snapshot.cost_usd == pytest.approx(0.003)
    assert tracker.total_cost == pytest.approx(0.003)
    assert tracker.should_warn() is False

    tracker.record("alt-model", prompt_tokens=400000, completion_tokens=0)
    assert tracker.should_warn() is True
    assert tracker.remaining_budget() == pytest.approx(5.0 - tracker.total_cost)

    with pytest.raises(BudgetExceededError):
        tracker.record("alt-model", prompt_tokens=150000, completion_tokens=0)


def test_xnovel_config_path_resolution(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    cfg = XNovelConfig().with_paths(input_path=input_dir, output_path=output_dir)
    assert cfg.paths.input_path == input_dir
    assert cfg.paths.output_path == output_dir

    cfg.ensure_directories()
    assert input_dir.exists() is False  # create_input defaults to False
    assert output_dir.exists()

    resolved_input = cfg.input_path
    resolved_output = cfg.output_path
    assert resolved_input == input_dir
    assert resolved_output == output_dir
