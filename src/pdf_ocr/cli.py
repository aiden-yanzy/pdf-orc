"""Command-line entrypoint for the standalone PDF OCR pipeline."""

from __future__ import annotations

from .processor import main, run_from_cli

__all__ = ["main", "run_from_cli"]
