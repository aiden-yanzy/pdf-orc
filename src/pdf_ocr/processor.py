"""
Standalone PDF OCR → Markdown pipeline.

This module is extracted from the zhihu_short_novel project and can be used
independently under ~/ai-lab/pdf_ocr.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Iterable, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

from pdf_ocr.extractors import PDFExtractionOptions, extract_pdf_to_text

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older installations
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        print(
            "⚠️ 检测到旧版 langchain-community 的 HuggingFaceEmbeddings，"
            "建议 `pip install -U langchain-huggingface` 并更新配置以消除弃用警告。"
        )
    except ImportError:
        HuggingFaceEmbeddings = None  # type: ignore

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_INPUT_DIR = Path("inputs")
DEFAULT_OUTPUT_DIR = Path("outputs") / "pdf_markdown"


class PDFProcessingState(TypedDict, total=False):
    """LangGraph 状态结构。"""

    pdf_path: str
    assets_dir: str
    page_texts: list[str]
    combined_text: str
    chunks: list[str]
    chunk_embeddings: list[list[float]]
    embed_available: bool
    refined_chunks: list[str]
    final_markdown: str
    media_assets: list[dict]


@dataclass
class PDFProcessingConfig:
    """运行配置参数。"""

    input_dir: Path = DEFAULT_INPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    overwrite: bool = False
    ocr_langs: str = os.getenv("OCR_LANGS", "chi_sim+eng")
    ocr_min_text_threshold: int = 40  # 页面文字不足时触发 OCR
    page_dpi: int = 300
    max_chunk_chars: int = 4000
    chunk_overlap: int = 200
    rag_top_k: int = 3
    similarity_threshold: float = 0.35
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "local")
    local_embed_model: str = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    local_embed_device: str = os.getenv("LOCAL_EMBED_DEVICE", "cpu")
    extract_images: bool = os.getenv("EXTRACT_IMAGES", "true").lower() == "true"
    enable_formula_detection: bool = os.getenv("ENABLE_FORMULA_DETECTION", "true").lower() == "true"


class PDFMarkdownConverter:
    """将 PDF 转换为 Markdown，并通过 LLM 修正文本。"""

    def __init__(self, config: PDFProcessingConfig):
        load_dotenv()

        self.config = config
        self.config.input_dir = config.input_dir.expanduser()
        self.config.output_dir = config.output_dir.expanduser()
        self.verbose = os.getenv("VERBOSE", "True").lower() == "true"

        self._ensure_directories()

        self._tesseract_available = which("tesseract") is not None
        if not self._tesseract_available:
            print("⚠️ 未检测到 tesseract 可执行文件，无法进行 OCR，将仅尝试直接提取文本。")

        self.llm = self._create_llm()
        self.embedder = self._create_embedder()
        self.workflow = self._build_workflow()

    def _ensure_directories(self) -> None:
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.config.input_dir}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_extraction_options(self) -> PDFExtractionOptions:
        return PDFExtractionOptions(
            ocr_langs=self.config.ocr_langs,
            ocr_min_text_threshold=self.config.ocr_min_text_threshold,
            page_dpi=self.config.page_dpi,
            extract_images=self.config.extract_images,
            enable_formula_detection=self.config.enable_formula_detection,
            output_dir=self.config.output_dir,
            tesseract_available=self._tesseract_available,
        )

    def _create_llm(self) -> ChatOpenAI:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:3000/v1")
        model_name = os.getenv("OPENAI_MODEL", "claude-3-7-sonnet-20250219")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        if not api_key and base_url:
            print("⚠️ 未设置 OPENAI_API_KEY，将尝试使用 base_url 代理。")

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key or None,
            temperature=temperature,
        )

    def _create_embedder(self):
        backend = (self.config.embedding_backend or "openai").lower()

        if backend in {"none", "disabled", "off"}:
            print("ℹ️ 已禁用向量嵌入，RAG 功能关闭。")
            return None

        if backend == "local":
            if HuggingFaceEmbeddings is None:
                print("⚠️ 缺少 langchain-huggingface 依赖，无法加载本地嵌入模型，将跳过 RAG。")
                return None

            model_name = self.config.local_embed_model
            device = self.config.local_embed_device
            try:
                embedder = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True},
                )
                if self.verbose:
                    print(f"🔧 Local Embedding Model: {model_name} (device={device})")
                return embedder
            except Exception as exc:  # pragma: no cover
                print(f"⚠️ 初始化本地嵌入模型失败，将跳过 RAG 相似度检索: {exc}")
                return None

        api_key = os.getenv("OPENAI_API_KEY", "")
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        embed_base_url = os.getenv("OPENAI_EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        if not api_key and not embed_base_url:
            print("⚠️ 未检测到嵌入模型配置，将跳过 RAG 相似度检索。")
            return None

        try:
            embedder = OpenAIEmbeddings(
                model=embed_model,
                openai_api_key=api_key or None,
                openai_api_base=embed_base_url,
            )
            if self.verbose:
                print("🔧 Embedding Model:", embed_model)
            return embedder
        except Exception as exc:  # pragma: no cover
            print(f"⚠️ 初始化嵌入模型失败，将跳过 RAG 相似度检索: {exc}")
            return None

    def _build_workflow(self):
        graph = StateGraph(PDFProcessingState)
        graph.add_node("extract_text", self._node_extract_text)
        graph.add_node("chunk_text", self._node_chunk_text)
        graph.add_node("embed_chunks", self._node_embed_chunks)
        graph.add_node("refine_chunks", self._node_refine_chunks)
        graph.add_node("assemble_markdown", self._node_assemble_markdown)

        graph.add_edge(START, "extract_text")
        graph.add_edge("extract_text", "chunk_text")
        graph.add_edge("chunk_text", "embed_chunks")
        graph.add_edge("embed_chunks", "refine_chunks")
        graph.add_edge("refine_chunks", "assemble_markdown")
        graph.add_edge("assemble_markdown", END)

        return graph.compile()

    def _node_extract_text(self, state: PDFProcessingState) -> PDFProcessingState:
        pdf_path = Path(state["pdf_path"])
        assets_dir = Path(state.get("assets_dir", self.config.output_dir / f"{pdf_path.stem}_assets"))
        page_texts, combined_text, media_assets = self._extract_pdf_content(pdf_path, assets_dir)

        updated = dict(state)
        updated["page_texts"] = page_texts
        updated["combined_text"] = combined_text
        updated["media_assets"] = media_assets
        print(f"📚 已提取 {len(page_texts)} 页文本，总长度 {len(combined_text)} 字符。")
        return updated

    def _node_chunk_text(self, state: PDFProcessingState) -> PDFProcessingState:
        combined_text = state.get("combined_text", "")
        chunks = self._split_into_chunks(combined_text)
        updated = dict(state)
        updated["chunks"] = chunks
        avg = (sum(len(c) for c in chunks) // len(chunks)) if chunks else 0
        print(f"🧩 文本拆分为 {len(chunks)} 段，平均长度约 {avg} 字符。")
        return updated

    def _node_embed_chunks(self, state: PDFProcessingState) -> PDFProcessingState:
        chunks = state.get("chunks", [])
        updated = dict(state)
        if not chunks:
            updated["chunk_embeddings"] = []
            updated["embed_available"] = False
            print("⚠️ 无文本分段，可跳过嵌入计算。")
            return updated

        if not self.embedder:
            updated["chunk_embeddings"] = []
            updated["embed_available"] = False
            print("⚠️ 未配置嵌入模型，跳过相似度检索。")
            return updated

        try:
            embeddings = self.embedder.embed_documents(chunks)  # type: ignore[union-attr]
            updated["chunk_embeddings"] = embeddings
            updated["embed_available"] = True
            print(f"📊 已生成 {len(embeddings)} 个向量嵌入，用于 RAG 检索。")
        except Exception as exc:  # pragma: no cover
            updated["chunk_embeddings"] = []
            updated["embed_available"] = False
            print(f"⚠️ 向量嵌入生成失败，将退化为顺序上下文：{exc}")
            self.embedder = None
        return updated

    def _node_refine_chunks(self, state: PDFProcessingState) -> PDFProcessingState:
        chunks = state.get("chunks", [])
        if not chunks:
            raise ValueError("未获取到任何文本分段，无法进行 LLM 修复。")

        embeddings = state.get("chunk_embeddings") or []
        embed_available = state.get("embed_available", False) and len(embeddings) == len(chunks)
        vector_matrix = np.array(embeddings) if embed_available else None

        results: list[str] = []
        for idx, chunk in enumerate(chunks, start=0):
            print(f"🤖 正在修复段 {idx + 1}/{len(chunks)}，长度 {len(chunk)} 字符。")
            context_indices: list[int] = []
            if idx > 0:
                context_indices.append(idx - 1)

            if embed_available and vector_matrix is not None and vector_matrix.size > 0:
                similar_indices = self._retrieve_similar_indices(vector_matrix, idx)
                context_indices.extend(similar_indices)

            context_indices = [i for i in dict.fromkeys(context_indices) if i != idx]
            if context_indices:
                print(f"   │  使用上下文段索引: {context_indices}")
            else:
                print("   │  未检索到额外上下文，使用当前段落独立修复。")

            context_snippets = [chunks[i] for i in context_indices]
            context_text = "\n\n---\n\n".join(context_snippets) if context_snippets else "无"

            system_prompt = (
                "你是一名专业的文本编辑，需要将 OCR 结果整理成结构化的 Markdown 文档。"
                "请纠正明显的识别错误，保留原文语气，合并断裂的句子，并在必要时加入小节标题。"
                "如果检测到错别字、乱码或语句不通顺，请在合理范围内修正。"
                "对提供的上下文进行适度参照，但不要重复上下文内容。"
                "输出应只包含修正后的正文，不要加入修复说明、提示语或任何与正文无关的解释。"
                "若文本中已存在图片或公式占位符，请原样保留，无需额外说明。"
            )
            user_prompt = (
                f"文件名: {Path(state['pdf_path']).name}\n"
                f"当前段落序号: {idx + 1}/{len(chunks)}\n"
                "以下上下文仅供理解，请勿在输出中重复：\n"
                "```text\n"
                f"{context_text}\n"
                "```\n\n"
                "以下是需要修复的 OCR 文本，请输出 Markdown：\n"
                "```text\n"
                f"{chunk}\n"
                "```"
            )

            response = self.llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )

            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(
                    chunk_piece.get("text", "")
                    for chunk_piece in content
                    if isinstance(chunk_piece, dict)
                )
            if not content.strip():
                raise RuntimeError("LLM 返回内容为空。")

            results.append(content.strip())

        updated = dict(state)
        updated["refined_chunks"] = results
        return updated

    def _node_assemble_markdown(self, state: PDFProcessingState) -> PDFProcessingState:
        refined = state.get("refined_chunks") or []
        if not refined:
            refined = state.get("chunks", [])
        final_markdown = "\n\n".join(refined).strip()
        print(f"🧾 Markdown 聚合完成，最终长度 {len(final_markdown)} 字符。")
        updated = dict(state)
        updated["final_markdown"] = final_markdown
        return updated

    def process_all(self) -> list[Path]:
        pdf_files = sorted(self.config.input_dir.glob("*.pdf"))
        if not pdf_files:
            print("📂 未在输入目录中找到 PDF 文件。")
            return []

        print(f"📑 共检测到 {len(pdf_files)} 个 PDF，将逐一处理。")
        generated_files: list[Path] = []
        for pdf_path in pdf_files:
            try:
                output_path = self._process_single(pdf_path)
                generated_files.append(output_path)
                print(f"✅ 已生成: {output_path}")
            except Exception as exc:  # pragma: no cover
                print(f"❌ 处理 {pdf_path.name} 时失败: {exc}")
        return generated_files

    def _process_single(self, pdf_path: Path) -> Path:
        print(f"📄 开始处理: {pdf_path.name}")
        output_path = self.config.output_dir / (pdf_path.stem + ".md")
        assets_dir = self.config.output_dir / f"{pdf_path.stem}_assets"
        if self.config.extract_images:
            assets_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not self.config.overwrite:
            print(f"⏭️ 跳过已有文件: {output_path}")
            return output_path

        final_state = self.workflow.invoke(
            {"pdf_path": str(pdf_path), "assets_dir": str(assets_dir)},
            config={"configurable": {"thread_id": f"pdf-{pdf_path.stem}"}},
        )

        markdown_text = (final_state.get("final_markdown") or "").strip()
        if not markdown_text:
            raise ValueError("LLM 修复后内容为空，可能是上游步骤失败。")

        output_path.write_text(markdown_text, encoding="utf-8")
        return output_path

    def _extract_pdf_content(self, pdf_path: Path, assets_dir: Path) -> tuple[list[str], str, list[dict]]:
        options = self._build_extraction_options()
        logger = print
        assets_target: Path | None = assets_dir if self.config.extract_images else None
        result = extract_pdf_to_text(
            pdf_path,
            assets_dir=assets_target,
            options=options,
            logger=logger,
        )
        return result.page_texts, result.combined_text, result.media_assets

    def _split_into_chunks(self, text: str) -> list[str]:
        max_chars = max(1000, self.config.max_chunk_chars)
        overlap = max(0, min(self.config.chunk_overlap, max_chars // 2))
        cleaned = text.strip()
        if len(cleaned) <= max_chars:
            return [cleaned]

        chunks: list[str] = []
        start = 0
        text_length = len(cleaned)
        while start < text_length:
            end = min(text_length, start + max_chars)
            chunk = cleaned[start:end]
            chunks.append(chunk.strip())
            if end == text_length:
                break
            start = max(0, end - overlap)

        return [chunk for chunk in chunks if chunk]

    def _retrieve_similar_indices(self, matrix: np.ndarray, target_idx: int) -> list[int]:
        if len(matrix) <= 1:
            return []

        top_k = max(0, self.config.rag_top_k)
        if top_k == 0:
            return []

        target_vec = matrix[target_idx]
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return []

        norms = np.linalg.norm(matrix, axis=1) + 1e-10
        similarities = (matrix @ target_vec) / (norms * target_norm + 1e-10)
        similarities[target_idx] = -1.0

        sorted_indices = np.argsort(similarities)[::-1]
        selected: list[int] = []
        for idx in sorted_indices:
            if len(selected) >= top_k:
                break
            score = similarities[idx]
            if score < self.config.similarity_threshold:
                continue
            selected.append(int(idx))

        return selected



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 inputs 目录中的 PDF 进行 OCR 并生成 Markdown。")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="PDF 输入目录。")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Markdown 输出目录。",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 Markdown 文件。")
    parser.add_argument(
        "--ocr_langs",
        type=str,
        default=os.getenv("OCR_LANGS", "chi_sim+eng"),
        help="pytesseract 语言包参数，例如 'chi_sim+eng'。",
    )
    parser.add_argument(
        "--ocr_min_text_threshold",
        type=int,
        default=40,
        help="当页面直接提取的字符数低于该值时触发 OCR。",
    )
    parser.add_argument("--page_dpi", type=int, default=300, help="OCR 渲染使用的 DPI。")
    parser.add_argument(
        "--max_chunk_chars",
        type=int,
        default=4000,
        help="传入 LLM 前单段最大字符数，超过会拆分。",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="分段时的字符重叠，用于保持上下文连贯。",
    )
    parser.add_argument(
        "--rag_top_k",
        type=int,
        default=3,
        help="RAG 检索时返回的相似段落数量。",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.35,
        help="当相似度低于该阈值时忽略检索结果。",
    )
    parser.add_argument(
        "--embedding_backend",
        type=str,
        default=os.getenv("EMBEDDING_BACKEND", "local"),
        choices=["local", "openai", "none"],
        help="向量嵌入后端，可选 local/openai/none。",
    )
    parser.add_argument(
        "--local_embed_model",
        type=str,
        default=os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="本地嵌入模型名称或目录（需要提前下载）。",
    )
    parser.add_argument(
        "--local_embed_device",
        type=str,
        default=os.getenv("LOCAL_EMBED_DEVICE", "cpu"),
        help="本地嵌入模型加载设备，例如 cpu、cuda。",
    )
    parser.add_argument(
        "--extract_images",
        action="store_true",
        default=os.getenv("EXTRACT_IMAGES", "true").lower() == "true",
        help="启用图片提取并写入 Markdown。",
    )
    parser.add_argument(
        "--no-extract_images",
        dest="extract_images",
        action="store_false",
        help="禁用图片提取。",
    )
    parser.add_argument(
        "--enable_formula_detection",
        action="store_true",
        default=os.getenv("ENABLE_FORMULA_DETECTION", "true").lower() == "true",
        help="启用公式检测并插入 LaTeX 占位符。",
    )
    parser.add_argument(
        "--disable_formula_detection",
        dest="enable_formula_detection",
        action="store_false",
        help="禁用公式检测。",
    )
    return parser


def run_from_cli(args: Iterable[str] | None = None) -> list[Path]:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    config = PDFProcessingConfig(
        input_dir=parsed.input_dir,
        output_dir=parsed.output_dir,
        overwrite=parsed.overwrite,
        ocr_langs=parsed.ocr_langs,
        ocr_min_text_threshold=parsed.ocr_min_text_threshold,
        page_dpi=parsed.page_dpi,
        max_chunk_chars=parsed.max_chunk_chars,
        chunk_overlap=parsed.chunk_overlap,
        rag_top_k=parsed.rag_top_k,
        similarity_threshold=parsed.similarity_threshold,
        embedding_backend=parsed.embedding_backend,
        local_embed_model=parsed.local_embed_model,
        local_embed_device=parsed.local_embed_device,
        extract_images=parsed.extract_images,
        enable_formula_detection=parsed.enable_formula_detection,
    )

    processor = PDFMarkdownConverter(config)
    return processor.process_all()


def main():
    """Console entrypoint."""
    run_from_cli()


if __name__ == "__main__":  # pragma: no cover
    main()
