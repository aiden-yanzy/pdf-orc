from __future__ import annotations

import json
from pathlib import Path

import pytest

from xnovel.novel import (
    OutlineAgent,
    OutlineConfig,
    OutlineValidationError,
    generate_character_aliases,
    remap_location,
    remap_setting,
    remap_timeframe,
)


class MockProvider:
    def __init__(self, response: object):
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> object:
        self.calls.append(prompt)
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response, ensure_ascii=False)


def _analysis_fixture() -> dict:
    return {
        "title": "Shadowed Case",
        "synopsis": "Alice and Bob chase a thief through modern New York in the 2020s.",
        "characters": [
            {"name": "Alice", "role": "investigator"},
            {"name": "Bob", "role": "partner"},
        ],
        "setting": {"location": "New York City", "time_period": "2020s"},
        "themes": ["friendship", "justice"],
        "chapters": [
            {
                "title": "Trailblazing",
                "summary": "Alice and Bob discover a clue in a crowded subway.",
                "scenes": [
                    {
                        "title": "Crowded Subway",
                        "summary": "They follow the thief onto a train and almost lose them.",
                    }
                ],
            }
        ],
    }


def test_generate_character_aliases_deterministic():
    characters = ["Alice", "Bob", "Charlie"]
    mapping_one = generate_character_aliases(characters, aggressiveness=0.8, seed=42)
    mapping_two = generate_character_aliases(characters, aggressiveness=0.8, seed=42)
    mapping_three = generate_character_aliases(characters, aggressiveness=0.8, seed=99)

    assert mapping_one == mapping_two
    assert mapping_one != mapping_three
    assert len(set(mapping_one.values())) == len(characters)
    for original, alias in mapping_one.items():
        assert alias.lower() != original.lower()


def test_setting_remap_changes_time_and_place():
    location = remap_location("New York", aggressiveness=0.6, seed=3)
    timeframe = remap_timeframe("2020s", aggressiveness=0.6, seed=3)
    assert location != "New York"
    assert timeframe != "2020s"

    combined = remap_setting({"location": "Paris", "time_period": "Belle Epoque"}, aggressiveness=0.9, seed=10)
    assert combined["location"] != "Paris"
    assert combined["time_period"] != "Belle Epoque"


def test_outline_agent_generates_expected_outputs(tmp_path: Path):
    analysis = _analysis_fixture()
    config = OutlineConfig(output_dir=tmp_path, similarity_threshold=0.4, aggressiveness=0.7, seed=21)

    expected_aliases = generate_character_aliases(analysis["characters"], aggressiveness=0.7, seed=21)
    expected_setting = remap_setting(analysis["setting"], aggressiveness=0.7, seed=21)

    provider_payload = {
        "title": "Citadel Veil",
        "plot_summary": "In the aurora decades, Aria and Bastien guard the storm-crowned citadel while unraveling a rebellion.",
        "character_map": expected_aliases,
        "setting": {
            "location": expected_setting["location"],
            "time_period": expected_setting["time_period"],
            "mood": "storm-lit intrigue",
        },
        "thematic_shifts": [{"from": "friendship", "to": "collective hope"}],
        "chapters": [
            {
                "title": "Citadel on the Brink",
                "summary": "Aria senses fractures in the floating bastion's alliances.",
                "scenes": [
                    {
                        "title": "Thunderous Council",
                        "summary": "Bastien and Aria confront dissenters amid roaring winds.",
                        "beats": [
                            "Introduce the rebel envoy",
                            "Aria secures a fragile truce",
                        ],
                    }
                ],
            }
        ],
    }

    provider = MockProvider(provider_payload)
    agent = OutlineAgent(provider, config)

    state: dict[str, object] = {"analysis": analysis}
    updated_state = agent.run(state)

    assert provider.calls, "provider should be invoked"
    prompt_json = provider.calls[0]
    assert '"aggressiveness": 0.7' in prompt_json

    outline = updated_state["outline"]
    assert outline["character_map"] == expected_aliases
    assert outline["setting"]["location"] == expected_setting["location"]
    assert outline["setting"]["time_period"] == expected_setting["time_period"]

    artifacts = updated_state["outline_artifacts"]
    markdown_path = Path(artifacts["markdown_path"])
    json_path = Path(artifacts["json_path"])
    assert markdown_path.exists()
    assert json_path.exists()

    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "## Dramatis Personae" in markdown_text
    for original, alias in expected_aliases.items():
        assert alias in markdown_text
        assert original in markdown_text

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["character_map"] == expected_aliases
    assert data["chapters"][0]["scenes"][0]["title"] == "Thunderous Council"


def test_outline_agent_similarity_threshold_enforced(tmp_path: Path):
    analysis = _analysis_fixture()
    config = OutlineConfig(output_dir=tmp_path, similarity_threshold=0.2, aggressiveness=0.3, seed=8)

    expected_aliases = generate_character_aliases(analysis["characters"], aggressiveness=0.3, seed=8)
    expected_setting = remap_setting(analysis["setting"], aggressiveness=0.3, seed=8)

    # Force a response that mimics the original synopsis too closely
    provider_payload = {
        "title": "Nearly the Same",
        "plot_summary": analysis["synopsis"],
        "character_map": expected_aliases,
        "setting": expected_setting,
        "chapters": analysis["chapters"],
    }
    provider = MockProvider(provider_payload)
    agent = OutlineAgent(provider, config)

    state: dict[str, object] = {"analysis": analysis}
    with pytest.raises(OutlineValidationError) as exc:
        agent.run(state)
    assert "too similar" in str(exc.value)
