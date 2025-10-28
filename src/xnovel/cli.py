"""Command line interface for the xNovel orchestration workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from .novel import (
    NovelWorkflowConfig,
    run_all,
    run_analysis,
    run_outline,
    run_review,
    run_writing,
)

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xnovel",
        description=(
            "Tools for orchestrating the short novel pipeline. Ensure you create or "
            "switch to the appropriate git feature branch manually before running "
            "these commands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command")

    novel_parser = subparsers.add_parser(
        "novel",
        help="Coordinate analysis → outline → writing → review stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        description=(
            "Novel development workflow coordinating analysis, outline, writing, and "
            "review stages. Branch creation and git operations remain manual to "
            "respect repository policies."
        ),
    )
    novel_sub = novel_parser.add_subparsers(dest="novel_command", required=True)

    for name, help_text in [
        ("analyze", "Generate an analysis snapshot for the manuscript."),
        ("outline", "Create a numbered outline based on the manuscript analysis."),
        ("write", "Expand the outline into draft chapters."),
        ("review", "Produce review notes and a polished manuscript."),
        ("run-all", "Execute the entire workflow (analysis → review)."),
    ]:
        sub = novel_sub.add_parser(
            name,
            help=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            allow_abbrev=False,
        )
        _register_shared_arguments(sub)

    return parser


def _register_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source manuscript (.md or .txt).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where artefacts and logs will be written.",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        help="LLM provider to use (mock, openai, etc.).",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Optional base URL for API-compatible providers.",
    )
    parser.add_argument(
        "--api-key-env",
        dest="api_key_env",
        default=None,
        help="Environment variable containing the provider API key.",
    )
    parser.add_argument(
        "--model",
        default="mock-latest",
        help="Model name or identifier to target.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generative providers.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for provider responses.",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Optional spend budget in USD before the run aborts.",
    )
    parser.add_argument(
        "--max-review-iters",
        type=int,
        default=2,
        help="Maximum number of review iterations (informational).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic mock outputs.",
    )
    parser.add_argument(
        "--chapter-range",
        type=_chapter_range_type,
        default=None,
        help="Optional chapter range to emphasise (format: start:end or single number).",
    )
    parser.add_argument(
        "--context-strategy",
        default="sequential",
        help="Context retrieval strategy identifier used by downstream stages.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Similarity threshold applied when retrieving context.",
    )


def _chapter_range_type(value: str) -> tuple[int, int]:
    parts = value.split(":", 1)
    if len(parts) == 1:
        start = end = _positive_int(parts[0])
    else:
        left, right = parts
        start = _positive_int(left) if left else 1
        end = _positive_int(right) if right else start
    if end < start:
        raise argparse.ArgumentTypeError("chapter-range end must not be earlier than start")
    return (start, end)


def _positive_int(token: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:  # pragma: no cover - argparse formatting
        raise argparse.ArgumentTypeError(f"Invalid integer value: {token}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("Values must be positive integers")
    return value


def _build_config(args: argparse.Namespace) -> NovelWorkflowConfig:
    return NovelWorkflowConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        provider=args.provider,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        budget_usd=args.budget_usd,
        max_review_iters=args.max_review_iters,
        seed=args.seed,
        chapter_range=args.chapter_range,
        context_strategy=args.context_strategy,
        similarity_threshold=args.similarity_threshold,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "novel":
        parser.print_help()
        return 0

    command_map: dict[str, Callable[[NovelWorkflowConfig], object]] = {
        "analyze": run_analysis,
        "outline": run_outline,
        "write": run_writing,
        "review": run_review,
        "run-all": run_all,
    }

    runner = command_map.get(args.novel_command)
    if runner is None:  # pragma: no cover - safety net
        parser.print_help()
        return 1

    config = _build_config(args)

    try:
        runner(config)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
