from __future__ import annotations

from pathlib import Path

import pytest

from xnovel.io import LoadedDocument, load_input_resource


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text("# Title\n\nSome **content**.", encoding="utf-8")
    return path


def test_load_input_resource_markdown(markdown_file: Path) -> None:
    document = load_input_resource(markdown_file)
    assert isinstance(document, LoadedDocument)
    assert document.metadata["kind"] == "markdown"
    assert document.metadata["length"] == len(document.content)
    assert document.metadata["path"].endswith("sample.md")


def test_load_input_resource_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from pdf_ocr.extractors import PDFExtractionResult
    from xnovel import io as io_module

    result = PDFExtractionResult(page_texts=["hello"], combined_text="hello", media_assets=[])

    def fake_extract(pdf_path: Path, *, assets_dir=None, options=None, logger=None):
        return result

    monkeypatch.setattr(io_module, "extract_pdf_to_text", fake_extract)

    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% stub content")

    document = load_input_resource(pdf_path, assets_dir=tmp_path / "assets")
    assert document.metadata["kind"] == "pdf"
    assert document.metadata["pages"] == 1
    assert document.metadata["media_assets"] == []
    assert document.metadata["assets_dir"].endswith("assets")


def test_load_input_resource_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.txt"
    path.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError):
        load_input_resource(path)
