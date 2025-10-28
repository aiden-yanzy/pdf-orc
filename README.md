# PDF OCR Pipeline

独立的 PDF → Markdown 转换工具，支持：

- PyMuPDF 文本提取 + Tesseract OCR 补齐
- 图片资源导出并写入 Markdown 占位符
- 简单公式检测（生成 LaTeX 片段）
- LangGraph 工作流编排 + LLM 修正
- RAG 方式的相似度检索（本地或 OpenAI 嵌入）

## 快速开始

```bash
cd ~/ai-lab/pdf_ocr
pip install -e .

# 默认从 inputs 读取 PDF，输出至 outputs/pdf_markdown
python -m pdf_ocr.cli
```

### 常用参数

```bash
# 指定输入/输出目录
python -m pdf_ocr.cli --input_dir ./inputs --output_dir ./outputs

# 调整分段与 RAG
python -m pdf_ocr.cli --max_chunk_chars 3500 --chunk_overlap 200 --rag_top_k 4

# 使用本地嵌入模型
python -m pdf_ocr.cli --embedding_backend local --local_embed_model /path/to/model

# 关闭图片提取或公式检测
python -m pdf_ocr.cli --no-extract_images --disable_formula_detection
```

需要先安装 Tesseract OCR（含中文语言包）以及可访问的 LLM/嵌入服务。

环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `EMBEDDING_BACKEND` 等与原项目保持一致。详细说明见 `pdf_ocr/processor.py` 中的注释。

## 更多文档

- Novel pipeline usage guide: [docs/USAGE-novel.md](docs/USAGE-novel.md) (English)
- 更新记录参见 [docs/CHANGELOG.md](docs/CHANGELOG.md)。
