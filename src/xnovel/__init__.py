"""Scaffolding package for the novel agent pipeline."""

from .config import BudgetConfig, LLMConfig, XNovelConfig
from .io import LoadedDocument, load_input_resource
from .paths import NovelPathConfig, resolve_input_path, resolve_output_path

__all__ = [
    "BudgetConfig",
    "LLMConfig",
    "XNovelConfig",
    "NovelPathConfig",
    "resolve_input_path",
    "resolve_output_path",
    "LoadedDocument",
    "load_input_resource",
]
