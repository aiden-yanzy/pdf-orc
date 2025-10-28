from __future__ import annotations

from pathlib import Path

import pytest

from pdf_ocr import extractors


def test_pdf_extraction_result_helpers() -> None:
    result = extractors.PDFExtractionResult(
        page_texts=["one", "two"], combined_text="one\ntwo", media_assets=[{"type": "image"}]
    )
    assert result.page_count == 2
    assert result.markdown_text == "one\ntwo"
    payload = result.as_dict()
    assert payload["combined_text"] == "one\ntwo"


def test_extract_pdf_to_markdown_uses_extract_to_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_extract(pdf_path: Path, *, assets_dir=None, options=None, logger=None):
        captured["pdf_path"] = pdf_path
        captured["assets_dir"] = assets_dir
        captured["logger"] = logger
        return extractors.PDFExtractionResult(page_texts=["ok"], combined_text="ok", media_assets=[])

    monkeypatch.setattr(extractors, "extract_pdf_to_text", fake_extract)

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    markdown, assets = extractors.extract_pdf_to_markdown(pdf_path, assets_dir=tmp_path / "assets")
    assert markdown == "ok"
    assert assets == []
    assert captured["pdf_path"] == pdf_path


@pytest.mark.parametrize(
    "candidate, expected",
    [
        ("x = y + 1", True),
        ("Plain text", False),
        ("\\frac{a}{b}", True),
        ("12 + 34 = 46", True),
    ],
)
def test_formula_detection_heuristic(candidate: str, expected: bool) -> None:
    assert extractors._looks_like_formula(candidate) is expected


def test_format_formula_placeholder() -> None:
    placeholder = extractors._format_formula_placeholder("sample", 2, 4, "x^2")
    assert "[公式 2-4]" in placeholder
    assert "x^2" in placeholder
