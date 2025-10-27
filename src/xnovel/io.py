"""Input loading utilities for the xnovel agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pdf_ocr.extractors import PDFExtractionOptions, extract_pdf_to_text

__all__ = ["LoadedDocument", "load_input_resource"]

Logger = Callable[[str], None]

SUPPORTED_MARKDOWN_SUFFIXES = {".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}


@dataclass(slots=True)
class LoadedDocument:
    """Container for structured input content and metadata."""

    content: str
    source: Path
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source": str(self.source),
            "metadata": self.metadata,
        }


def load_input_resource(
    source: Path | str,
    *,
    extraction_options: PDFExtractionOptions | None = None,
    assets_dir: Path | str | None = None,
    logger: Logger | None = None,
    encoding: str = "utf-8",
) -> LoadedDocument:
    """Load Markdown or PDF content and return text with metadata."""

    source_path = Path(source).expanduser()
    if not source_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"资源不存在: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix in SUPPORTED_MARKDOWN_SUFFIXES:
        text = source_path.read_text(encoding=encoding)
        metadata = {
            "kind": "markdown",
            "length": len(text),
            "path": str(source_path),
        }
        return LoadedDocument(content=text, source=source_path, metadata=metadata)

    if suffix in PDF_SUFFIXES:
        options = extraction_options or PDFExtractionOptions()
        result = extract_pdf_to_text(source_path, assets_dir=assets_dir, options=options, logger=logger)
        metadata = {
            "kind": "pdf",
            "pages": result.page_count,
            "media_assets": result.media_assets,
            "path": str(source_path),
        }
        if assets_dir is not None:
            metadata["assets_dir"] = str(Path(assets_dir).expanduser())
        return LoadedDocument(content=result.combined_text, source=source_path, metadata=metadata)

    raise ValueError(f"Unsupported input format for {source_path}")
