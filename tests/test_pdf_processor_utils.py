from __future__ import annotations

import numpy as np
import pytest

from pdf_ocr.processor import PDFMarkdownConverter, PDFProcessingConfig


def make_converter(**config_overrides):
    cfg = PDFProcessingConfig(**config_overrides)
    converter = object.__new__(PDFMarkdownConverter)
    converter.config = cfg
    return converter


def test_split_into_chunks_respects_overlap() -> None:
    converter = make_converter(max_chunk_chars=10, chunk_overlap=4)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = converter._split_into_chunks(text)

    assert chunks[0].startswith("abcdef")
    assert len(chunks) > 1
    overlap = converter.config.chunk_overlap
    assert chunks[0][-overlap:] == chunks[1][:overlap]


def test_split_into_chunks_handles_short_text() -> None:
    converter = make_converter(max_chunk_chars=100)
    text = "short text"
    assert converter._split_into_chunks(text) == ["short text"]


def test_retrieve_similar_indices_filters_by_threshold() -> None:
    converter = make_converter(rag_top_k=3, similarity_threshold=0.3)

    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 0.1, 0.9],
        ]
    )

    indices = converter._retrieve_similar_indices(matrix, target_idx=0)
    assert indices == [1]

    converter.config.similarity_threshold = 0.0
    indices = converter._retrieve_similar_indices(matrix, target_idx=0)
    assert indices[:2] == [1, 2]


def test_retrieve_similar_indices_handles_small_matrix() -> None:
    converter = make_converter(rag_top_k=2)
    matrix = np.array([[0.0, 0.0, 0.0]])
    assert converter._retrieve_similar_indices(matrix, target_idx=0) == []

    converter.config.rag_top_k = 0
    matrix = np.array([[1.0, 0.0, 0.0], [0.2, 0.9, 0.0]])
    assert converter._retrieve_similar_indices(matrix, target_idx=0) == []
