"""Utility helpers for extracting structured text from PDF documents.

This module centralises the low-level logic that powers the legacy
``PDFMarkdownConverter`` so that it can be reused by newer agent
pipelines without pulling in the full OCR-to-Markdown workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Callable

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

__all__ = [
    "PDFExtractionOptions",
    "PDFExtractionResult",
    "extract_pdf_to_text",
    "extract_pdf_to_markdown",
]

Logger = Callable[[str], None]


@dataclass(slots=True)
class PDFExtractionOptions:
    """Configuration bundle controlling PDF text extraction behaviour."""

    ocr_langs: str = "chi_sim+eng"
    ocr_min_text_threshold: int = 40
    page_dpi: int = 300
    extract_images: bool = True
    enable_formula_detection: bool = True
    output_dir: Path | None = None
    tesseract_available: bool | None = None


@dataclass(slots=True)
class PDFExtractionResult:
    """Structured result returned by :func:`extract_pdf_to_text`."""

    page_texts: list[str]
    combined_text: str
    media_assets: list[dict[str, object]]

    @property
    def page_count(self) -> int:
        return len(self.page_texts)

    @property
    def markdown_text(self) -> str:
        return self.combined_text

    def as_dict(self) -> dict[str, object]:
        return {
            "page_texts": self.page_texts,
            "combined_text": self.combined_text,
            "media_assets": self.media_assets,
        }


def extract_pdf_to_text(
    pdf_path: Path | str,
    *,
    assets_dir: Path | str | None = None,
    options: PDFExtractionOptions | None = None,
    logger: Logger | None = None,
) -> PDFExtractionResult:
    """Extract page-wise text (with optional OCR) from a PDF document.

    Parameters
    ----------
    pdf_path:
        Target PDF file to process.
    assets_dir:
        Optional directory where extracted image assets will be written.
        When omitted, image placeholders are still returned but no files
        are created.
    options:
        Extraction behaviour overrides. Any field left ``None`` will be
        inferred from the environment or sensible defaults.
    logger:
        Optional callback used for diagnostic messages. Supply ``print``
        when you wish to replicate the legacy CLI output, or ``None`` to
        silence informational logs.
    """

    pdf_path = Path(pdf_path).expanduser()
    if not pdf_path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    opts = options or PDFExtractionOptions()
    tesseract_available = (
        opts.tesseract_available if opts.tesseract_available is not None else which("tesseract") is not None
    )

    assets_dir_path = Path(assets_dir).expanduser() if assets_dir is not None else None
    output_dir = opts.output_dir.expanduser() if opts.output_dir is not None else None

    if opts.extract_images and assets_dir_path is not None:
        assets_dir_path.mkdir(parents=True, exist_ok=True)

    relative_assets = None
    if opts.extract_images and assets_dir_path is not None:
        try:
            relative_assets = assets_dir_path.relative_to(output_dir) if output_dir is not None else assets_dir_path
        except ValueError:
            relative_assets = assets_dir_path

    page_texts: list[str] = []
    media_assets: list[dict[str, object]] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            page_parts: list[str] = []
            image_counter = 0
            formula_counter = 0

            for block in page_dict.get("blocks", []):
                block_type = block.get("type")
                if block_type == 0:
                    block_lines: list[str] = []
                    for line in block.get("lines", []):
                        line_spans: list[str] = []
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            if not span_text.strip():
                                continue
                            if opts.enable_formula_detection and _looks_like_formula(span_text):
                                formula_counter += 1
                                placeholder = _format_formula_placeholder(
                                    pdf_path.stem, page_index, formula_counter, span_text.strip()
                                )
                                line_spans.append(placeholder)
                                media_assets.append(
                                    {
                                        "type": "formula",
                                        "label": f"{page_index}-{formula_counter}",
                                        "page": page_index,
                                        "placeholder": placeholder,
                                        "latex": span_text.strip(),
                                    }
                                )
                            else:
                                line_spans.append(span_text)
                        if line_spans:
                            block_lines.append("".join(line_spans).strip())
                    if block_lines:
                        page_parts.append("\n".join(block_lines))
                elif block_type == 1 and opts.extract_images and assets_dir_path is not None:
                    xref = block.get("xref")
                    if not xref:
                        continue
                    image_counter += 1
                    saved = _save_image_asset(
                        doc=doc,
                        xref=xref,
                        assets_dir=assets_dir_path,
                        relative_base=relative_assets,
                        stem=pdf_path.stem,
                        page_index=page_index,
                        image_index=image_counter,
                        logger=logger,
                    )
                    if saved:
                        placeholder, asset_meta = saved
                        page_parts.append(placeholder)
                        media_assets.append(asset_meta)

            text_from_blocks = "\n\n".join(part for part in page_parts if part).strip()
            if len(text_from_blocks) >= opts.ocr_min_text_threshold:
                if logger:
                    logger(f"📥 第 {page_index} 页直接提取文本，长度 {len(text_from_blocks)} 字符。")
                page_texts.append(text_from_blocks)
                continue

            base_text = text_from_blocks
            if not tesseract_available:
                page_texts.append(base_text)
                if logger:
                    logger(f"⚠️ 第 {page_index} 页文本过短且未配置 OCR，保留原始内容。")
                continue

            ocr_text = _ocr_page(page, dpi=opts.page_dpi, ocr_langs=opts.ocr_langs)
            combined = (base_text + "\n" + ocr_text).strip() if base_text else ocr_text.strip()
            page_texts.append(combined)
            if logger:
                logger(f"🖼️ 第 {page_index} 页使用 OCR 补充内容，合并后长度 {len(combined)} 字符。")

    combined_text = "\n\n".join(chunk for chunk in page_texts if chunk)
    return PDFExtractionResult(page_texts=page_texts, combined_text=combined_text, media_assets=media_assets)


def extract_pdf_to_markdown(
    pdf_path: Path | str,
    *,
    assets_dir: Path | str | None = None,
    options: PDFExtractionOptions | None = None,
    logger: Logger | None = None,
) -> tuple[str, list[dict[str, object]]]:
    """Return Markdown text assembled from direct PDF extraction.

    The function returns a two-tuple ``(markdown_text, media_assets)`` to
    keep parity with legacy expectations.
    """

    result = extract_pdf_to_text(pdf_path, assets_dir=assets_dir, options=options, logger=logger)
    return result.combined_text, result.media_assets


def _ocr_page(page: fitz.Page, *, dpi: int, ocr_langs: str) -> str:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if pix.alpha:
        image = image.convert("RGB")

    try:
        return pytesseract.image_to_string(image, lang=ocr_langs)
    except pytesseract.TesseractNotFoundError as err:  # pragma: no cover - environment specific
        raise RuntimeError("pytesseract 未找到 Tesseract 可执行文件。") from err
    except pytesseract.TesseractError as err:  # pragma: no cover - environment specific
        raise RuntimeError(f"Tesseract OCR 执行失败: {err}") from err


def _save_image_asset(
    *,
    doc: fitz.Document,
    xref: int,
    assets_dir: Path,
    relative_base: Path | None,
    stem: str,
    page_index: int,
    image_index: int,
    logger: Logger | None,
) -> tuple[str, dict[str, object]] | None:
    try:
        image_info = doc.extract_image(xref)
    except Exception as exc:  # pragma: no cover - passthrough logging
        if logger:
            logger(f"⚠️ 提取图像失败 (page {page_index}, xref {xref}): {exc}")
        return None

    image_bytes = image_info.get("image")
    image_ext = image_info.get("ext", "png")
    if not image_bytes:
        return None

    filename = f"{stem}_p{page_index:03d}_img{image_index:02d}.{image_ext}"
    filepath = assets_dir / filename
    try:
        filepath.write_bytes(image_bytes)
    except Exception as exc:  # pragma: no cover - filesystem-dependent
        if logger:
            logger(f"⚠️ 保存图像失败: {filepath} ({exc})")
        return None

    relative_path = filepath if relative_base is None else (relative_base / filename)
    placeholder = f"![图像 {page_index}-{image_index}]({relative_path.as_posix()})"
    asset_meta: dict[str, object] = {
        "type": "image",
        "label": f"{page_index}-{image_index}",
        "page": page_index,
        "placeholder": placeholder,
        "relative_path": relative_path.as_posix(),
    }
    return placeholder, asset_meta


def _format_formula_placeholder(stem: str, page_index: int, formula_index: int, latex: str) -> str:
    sanitized = latex.strip()
    return f"[公式 {page_index}-{formula_index}]\n$$\n{sanitized}\n$$"


def _looks_like_formula(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 4:
        return False

    math_tokens = ["\\frac", "\\sum", "\\int", "\\sqrt", "\\alpha", "\\beta", "\\gamma", "\\pi"]
    if any(token in stripped for token in math_tokens):
        return True

    math_chars = set("=+-*/<>∑∫√≈≤≥^_{}[]|\\")
    total_chars = len(stripped)
    math_count = sum(1 for ch in stripped if ch in math_chars)
    digit_count = sum(ch.isdigit() for ch in stripped)
    if total_chars == 0:
        return False
    return (math_count + digit_count) >= total_chars * 0.4
