"""Novel-specific agents and helpers."""

from .outline_agent import (
    OutlineAgent,
    OutlineConfig,
    OutlineValidationError,
    generate_character_aliases,
    remap_location,
    remap_setting,
    remap_timeframe,
)

__all__ = [
    "OutlineAgent",
    "OutlineConfig",
    "OutlineValidationError",
    "generate_character_aliases",
    "remap_location",
    "remap_setting",
    "remap_timeframe",
]
