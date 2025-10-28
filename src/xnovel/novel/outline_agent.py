"""Outline generation agent for the novel pipeline.

This module consumes the upstream analysis state and produces a transformed
outline that deliberately distances itself from the source material. It writes
both Markdown and JSON artefacts while enforcing validation and similarity
constraints so the downstream stages can trust the outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

# ---------------------------------------------------------------------------
# Exceptions & configuration
# ---------------------------------------------------------------------------


class OutlineValidationError(ValueError):
    """Raised when the generated outline fails hard validation checks."""


@dataclass
class OutlineConfig:
    """Configuration for the outline generator."""

    output_dir: Path
    similarity_threshold: float = 0.35
    aggressiveness: float = 0.5
    seed: int | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1")
        if not (0.0 <= self.aggressiveness <= 1.0):
            raise ValueError("aggressiveness must be between 0 and 1")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def rng(self) -> random.Random:
        return random.Random(self.seed)


# ---------------------------------------------------------------------------
# Pure helper functions (deterministic with seed) for unit testing.
# ---------------------------------------------------------------------------


def _normalise_character_input(character: str | Mapping[str, Any]) -> str:
    if isinstance(character, str):
        return character
    for key in ("alias", "name", "label", "id"):
        if key in character and isinstance(character[key], str):
            return character[key]
    raise ValueError("Character input must contain a name-like field")


def generate_character_aliases(
    characters: Iterable[str | Mapping[str, Any]],
    *,
    aggressiveness: float = 0.5,
    seed: int | None = None,
) -> dict[str, str]:
    """Return a deterministic mapping of original character names to aliases.

    The aggressiveness parameter influences how adventurous the generated names
    become. Values close to 1 introduce more dramatic syllable blends, while
    0 keeps the aliases closer to a curated list of grounded names.
    """

    names = [_normalise_character_input(item).strip() for item in characters]
    if not names:
        return {}

    rng = random.Random(seed)
    exotic_fragments = [
        "ae", "ia", "on", "ar", "el", "is", "or", "ul", "yn", "va",
        "ra", "th", "li", "ka", "ze", "qu", "fi", "mo", "sa", "chi",
    ]
    grounded_pool = [
        "Aria", "Bastien", "Calla", "Dorian", "Elowen", "Fen", "Galen",
        "Helia", "Isolde", "Joran", "Kaelis", "Liora", "Marek", "Neris",
        "Orin", "Phaedra", "Quinn", "Rowan", "Sylas", "Tamsin", "Ulric",
        "Vesper", "Wren", "Xavian", "Yara", "Zephir",
    ]

    aliases: dict[str, str] = {}
    used_aliases: set[str] = set()
    for idx, original in enumerate(names):
        base_choice = grounded_pool[idx % len(grounded_pool)]
        if aggressiveness < 0.2:
            alias = base_choice
        else:
            additions = max(1, math.ceil(aggressiveness * 3))
            prefix = ""
            suffix = ""
            for _ in range(additions):
                prefix += rng.choice(exotic_fragments).capitalize()
                suffix += rng.choice(exotic_fragments)
            alias = f"{prefix}{base_choice}{suffix.capitalize()}"
        alias = alias.strip()
        if alias.lower() == original.lower():
            alias = f"{alias} {rng.choice(['Prime', 'Nova', 'Rex'])}"
        while alias in used_aliases:
            alias = f"{alias}{rng.randint(2, 9)}"
        aliases[original] = alias
        used_aliases.add(alias)
    return aliases


def remap_timeframe(
    original_timeframe: str,
    *,
    aggressiveness: float = 0.5,
    seed: int | None = None,
) -> str:
    """Produce a remapped timeframe description."""

    cleaned = (original_timeframe or "").strip()
    rng = random.Random(seed)
    eras = [
        "mythic antiquity", "a far-future diaspora", "the luminous age of airships",
        "a fragmented post-solar renaissance", "the clockwork rebellion era",
        "a storm-lashed century between empires", "the aurora decades",
        "the gilded dusk before the flood", "the ember age of fractured moons",
    ]
    if aggressiveness < 0.3 and cleaned:
        return f"an alternate take on {cleaned}"
    idx = rng.randrange(len(eras)) if eras else 0
    return eras[idx]


def remap_location(
    original_location: str,
    *,
    aggressiveness: float = 0.5,
    seed: int | None = None,
) -> str:
    """Produce a remapped location description."""

    cleaned = (original_location or "").strip()
    rng = random.Random(seed)
    locations = [
        "the floating bastions of Skyrind", "a submerged megacity beneath sapphire ice",
        "the canyon citadels of Auric Rim", "an itinerant forest that migrates each equinox",
        "the storm-fortified cliffs of Meridian Wraith", "a desert archipelago stitched by sky-bridges",
        "the bioluminescent caverns of Netherglow", "a ringed metropolis orbiting a gas giant",
        "the crystalline tundra of Everglass", "the helical towers of Verdant Spire",
    ]
    if aggressiveness < 0.3 and cleaned:
        descriptor = rng.choice(["hidden", "parallel", "reimagined", "shadow"])
        return f"a {descriptor} rendition of {cleaned}"
    idx = rng.randrange(len(locations)) if locations else 0
    return locations[idx]


def remap_setting(
    setting: Mapping[str, Any] | None,
    *,
    aggressiveness: float = 0.5,
    seed: int | None = None,
) -> dict[str, str]:
    """Return a deterministic remapped setting dictionary."""

    original = setting or {}
    location = original.get("location", "") if isinstance(original, Mapping) else ""
    timeframe = original.get("time_period", "") if isinstance(original, Mapping) else ""
    rng_seed = seed if seed is not None else None
    return {
        "location": remap_location(location, aggressiveness=aggressiveness, seed=rng_seed),
        "time_period": remap_timeframe(timeframe, aggressiveness=aggressiveness, seed=rng_seed + 1 if rng_seed is not None else None),
    }


# ---------------------------------------------------------------------------
# Outline agent implementation.
# ---------------------------------------------------------------------------


def _call_provider(provider: Any, prompt: str) -> Any:
    if provider is None:
        raise RuntimeError("No outline provider configured")
    if hasattr(provider, "invoke") and callable(provider.invoke):  # type: ignore[attr-defined]
        return provider.invoke(prompt)
    if hasattr(provider, "generate") and callable(provider.generate):  # type: ignore[attr-defined]
        return provider.generate(prompt)
    if callable(provider):
        return provider(prompt)
    raise TypeError("Unsupported provider type; expected callable or invoke/generate method")


class OutlineAgent:
    """Agent responsible for generating the transformed outline."""

    def __init__(self, provider: Any, config: OutlineConfig):
        self.provider = provider
        self.config = config

    # Public API -------------------------------------------------------------

    def run(self, state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        analysis = state.get("analysis")
        if not isinstance(analysis, Mapping):
            raise OutlineValidationError("Analysis state missing or invalid; cannot build outline")

        characters = analysis.get("characters", [])
        synopsis = (analysis.get("synopsis") or "").strip()
        setting = analysis.get("setting")
        chapters = analysis.get("chapters", [])

        alias_map = generate_character_aliases(
            characters,
            aggressiveness=self.config.aggressiveness,
            seed=self.config.seed,
        )
        remapped_setting = remap_setting(
            setting,
            aggressiveness=self.config.aggressiveness,
            seed=self.config.seed,
        )

        prompt_payload = {
            "task": "Generate a transformed outline with new aliases, setting, and tone",
            "aggressiveness": self.config.aggressiveness,
            "similarity_threshold": self.config.similarity_threshold,
            "characters": characters,
            "character_map": alias_map,
            "setting": remapped_setting,
            "themes": analysis.get("themes", []),
            "chapters": chapters,
            "synopsis": synopsis,
        }
        prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        response = _call_provider(self.provider, prompt)
        outline_data = self._parse_response(response)

        existing_map = outline_data.get("character_map")
        if isinstance(existing_map, Mapping):
            merged_map = dict(existing_map)
        else:
            merged_map = {}
        merged_map.update(alias_map)
        outline_data["character_map"] = merged_map

        existing_setting = outline_data.get("setting")
        if isinstance(existing_setting, Mapping):
            merged_setting = dict(existing_setting)
        else:
            merged_setting = {}
        merged_setting.update(remapped_setting)
        outline_data["setting"] = merged_setting

        thematic_shifts = outline_data.get("thematic_shifts")
        if not isinstance(thematic_shifts, list):
            outline_data["thematic_shifts"] = self._default_thematic_shifts(analysis)
        elif not thematic_shifts:
            outline_data["thematic_shifts"] = self._default_thematic_shifts(analysis)

        outline_data.setdefault("plot_summary", outline_data.get("summary", ""))

        self._validate_outline(outline_data, characters, setting)
        self._enforce_similarity(synopsis, chapters, outline_data)

        markdown_text = self._render_markdown(outline_data)
        json_path, markdown_path = self._write_outputs(outline_data, markdown_text)

        state["outline"] = outline_data
        state["outline_artifacts"] = {
            "json_path": str(json_path),
            "markdown_path": str(markdown_path),
        }
        return state

    # Internal helpers ------------------------------------------------------

    def _parse_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, Mapping):
            return dict(response)
        if hasattr(response, "content"):
            response = getattr(response, "content")
        if isinstance(response, str):
            stripped = response.strip()
            if stripped.startswith("```"):
                stripped = self._strip_code_fence(stripped)
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise OutlineValidationError("Model response was not valid JSON") from exc
        raise OutlineValidationError("Unsupported model response type")

    def _strip_code_fence(self, text: str) -> str:
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            return "\n".join(lines[1:-1])
        return text

    def _default_thematic_shifts(self, analysis: Mapping[str, Any]) -> list[dict[str, str]]:
        themes = analysis.get("themes") or []
        if not themes:
            return []
        rng = self.config.rng
        pivot_words = [
            "transcendence", "resilience", "catharsis", "collective hope", "reckoning",
            "memory", "rebirth", "solidarity", "reclamation", "self-authorship",
        ]
        shifts = []
        for theme in themes:
            target = rng.choice(pivot_words)
            if isinstance(theme, Mapping):
                source = theme.get("name") or theme.get("theme") or json.dumps(theme, ensure_ascii=False)
            else:
                source = str(theme)
            if source == target:
                target = f"renewed {target}"
            shifts.append({"from": source, "to": target})
        return shifts

    def _validate_outline(
        self,
        outline: Mapping[str, Any],
        characters: Iterable[Any],
        original_setting: Mapping[str, Any] | None,
    ) -> None:
        alias_map = outline.get("character_map")
        if not isinstance(alias_map, Mapping) or not alias_map:
            raise OutlineValidationError("Outline is missing character remapping data")
        original_names = [_normalise_character_input(item).strip() for item in characters]
        missing = [name for name in original_names if name and name not in alias_map]
        if missing:
            raise OutlineValidationError(f"Missing aliases for characters: {', '.join(missing)}")
        duplicates = len(set(alias_map.values())) != len(alias_map.values())
        if duplicates:
            raise OutlineValidationError("Character aliases must be unique")
        for name, alias in alias_map.items():
            if isinstance(name, str) and isinstance(alias, str) and alias.strip().lower() == name.strip().lower():
                raise OutlineValidationError(f"Character '{name}' was not renamed")

        setting = outline.get("setting")
        if not isinstance(setting, Mapping):
            raise OutlineValidationError("Outline is missing setting information")
        original_location = ""
        original_time = ""
        if isinstance(original_setting, Mapping):
            original_location = str(original_setting.get("location", ""))
            original_time = str(original_setting.get("time_period", ""))
        new_location = str(setting.get("location", ""))
        new_time = str(setting.get("time_period", ""))
        if new_location.strip().lower() == original_location.strip().lower():
            raise OutlineValidationError("Setting location was not transformed")
        if new_time.strip().lower() == original_time.strip().lower():
            raise OutlineValidationError("Setting timeframe was not transformed")

        chapters = outline.get("chapters")
        if not isinstance(chapters, list) or not chapters:
            raise OutlineValidationError("Outline must contain chapter details")
        for idx, chapter in enumerate(chapters, start=1):
            if not isinstance(chapter, Mapping):
                raise OutlineValidationError(f"Chapter {idx} has invalid structure")
            if not chapter.get("title"):
                raise OutlineValidationError(f"Chapter {idx} is missing a title")
            scenes = chapter.get("scenes")
            if scenes and not isinstance(scenes, list):
                raise OutlineValidationError(f"Chapter {idx} scenes must be a list")

    def _enforce_similarity(
        self,
        synopsis: str,
        chapters: Iterable[Mapping[str, Any]],
        outline: Mapping[str, Any],
    ) -> None:
        original_parts = [synopsis]
        for chapter in chapters or []:
            summary = chapter.get("summary") if isinstance(chapter, Mapping) else None
            if summary:
                original_parts.append(str(summary))
        original_text = " ".join(part.strip() for part in original_parts if part).strip()
        if not original_text:
            return  # nothing to compare against

        transformed_parts: list[str] = []
        if outline.get("plot_summary"):
            transformed_parts.append(str(outline["plot_summary"]))
        for chapter in outline.get("chapters", []) or []:
            if isinstance(chapter, Mapping):
                if chapter.get("summary"):
                    transformed_parts.append(str(chapter["summary"]))
                for scene in chapter.get("scenes", []) or []:
                    if isinstance(scene, Mapping) and scene.get("summary"):
                        transformed_parts.append(str(scene["summary"]))
        transformed_text = " ".join(part.strip() for part in transformed_parts if part).strip()
        if not transformed_text:
            return

        ratio = SequenceMatcher(a=original_text.lower(), b=transformed_text.lower()).ratio()
        if ratio > self.config.similarity_threshold:
            raise OutlineValidationError(
                "Transformed outline remains too similar to the source material "
                f"(similarity {ratio:.2f} > threshold {self.config.similarity_threshold:.2f})."
            )

    def _render_markdown(self, outline: Mapping[str, Any]) -> str:
        lines: list[str] = []
        title = outline.get("title") or "Transformed Outline"
        lines.append(f"# {title}")
        lines.append("")

        char_map = outline.get("character_map", {})
        if char_map:
            lines.append("## Dramatis Personae")
            for original, alias in char_map.items():
                lines.append(f"- **{alias}** (formerly {original})")
            lines.append("")

        setting = outline.get("setting", {})
        if setting:
            lines.append("## Reimagined Setting")
            location = setting.get("location")
            time_period = setting.get("time_period")
            if location:
                lines.append(f"- Location: {location}")
            if time_period:
                lines.append(f"- Era: {time_period}")
            for key, value in setting.items():
                if key in {"location", "time_period"}:
                    continue
                lines.append(f"- {key.replace('_', ' ').title()}: {value}")
            lines.append("")

        thematic = outline.get("thematic_shifts", [])
        if thematic:
            lines.append("## Thematic Shifts")
            for shift in thematic:
                if isinstance(shift, Mapping):
                    origin = shift.get("from", "original theme")
                    target = shift.get("to", "new direction")
                    lines.append(f"- {origin} → {target}")
                else:
                    lines.append(f"- {shift}")
            lines.append("")

        chapters = outline.get("chapters", []) or []
        for c_idx, chapter in enumerate(chapters, start=1):
            if not isinstance(chapter, Mapping):
                continue
            chap_title = chapter.get("title") or f"Chapter {c_idx}"
            lines.append(f"## Chapter {c_idx}: {chap_title}")
            if chapter.get("summary"):
                lines.append(str(chapter["summary"]))
                lines.append("")
            scenes = chapter.get("scenes") or []
            for s_idx, scene in enumerate(scenes, start=1):
                if not isinstance(scene, Mapping):
                    continue
                scene_title = scene.get("title") or scene.get("scene_title") or f"Scene {c_idx}.{s_idx}"
                lines.append(f"### Scene {c_idx}.{s_idx}: {scene_title}")
                if scene.get("summary"):
                    lines.append(str(scene["summary"]))
                beats = scene.get("beats")
                if isinstance(beats, list) and beats:
                    for beat in beats:
                        lines.append(f"- {beat}")
                lines.append("")
        markdown_text = "\n".join(line.rstrip() for line in lines).strip() + "\n"
        return markdown_text

    def _write_outputs(self, outline: Mapping[str, Any], markdown_text: str) -> tuple[Path, Path]:
        json_path = self.config.output_dir / "outline.json"
        markdown_path = self.config.output_dir / "outline.md"
        json_path.write_text(json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(markdown_text, encoding="utf-8")
        return json_path, markdown_path


__all__ = [
    "OutlineAgent",
    "OutlineConfig",
    "OutlineValidationError",
    "generate_character_aliases",
    "remap_location",
    "remap_timeframe",
    "remap_setting",
]
