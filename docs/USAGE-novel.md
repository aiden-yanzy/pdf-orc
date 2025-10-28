# Novel Pipeline Usage Guide

> **Scope**: This document focuses on the experimental long-form fiction pipeline that lives under the `xnovel` package. It does not replace the existing PDF OCR guidance; refer to `README.md` for details about the legacy pipeline.

## Overview

The novel pipeline ingests Markdown manuscripts or lightweight PDFs, passes them through LangGraph-powered refinement stages, and emits structured chapters under `outputs/novel`. It shares the same dependency set as the PDF OCR tooling (LangChain, OpenAI-compatible chat models, and optional embeddings) while adding cost/budget guardrails tailored for long-form editing sessions.

Default paths are resolved through [`NovelPathConfig`](../src/xnovel/paths.py):

```
inputs/novel/        # default staging area for manuscripts and PDFs
outputs/novel/       # enriched markdown, assets, and run metadata
```

## Installation and Environment Setup

```bash
# Inside the repository root
pip install -e .

# (Optional) load secrets and provider overrides
cp .env.example .env  # if you keep environment variables under version control
source .env
```

- Python 3.10 or newer is required.
- Ensure you have access to an OpenAI-compatible chat provider. The pipeline resolves API keys from `XNOVEL_API_KEY` (falling back to `OPENAI_API_KEY`).
- HuggingFace embeddings are optional; only enable them if you need retrieval-augmented context during refinement.

## Sample Assets

Lightweight samples are provided for smoke-testing:

- `samples/novel_sample.md` – Markdown excerpt demonstrating heading and list structure.
- `samples/novel_sample.pdf` – One-page PDF created specifically for this repository.

Copy either file into your input directory to verify the end-to-end flow:

```bash
mkdir -p inputs/novel
cp samples/novel_sample.* inputs/novel/
```

All sample copy is original to this repository and can be redistributed under the repository's license.

## Running the CLI

The CLI module exposes a `novel` command group. The most common entrypoint processes every supported artifact inside the input directory:

```bash
python -m xnovel.cli novel run-all \
  --input inputs/novel \
  --output outputs/novel \
  --model gpt-4o-mini \
  --temperature 0.1 \
  --track-costs
```

Use `python -m xnovel.cli novel --help` to inspect the full set of subcommands and options. The CLI mirrors the dataclasses in `xnovel.config`, so flags are deliberately named after configuration attributes.

### Common Flags

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--input`, `-i` | Override the resolved input directory. | `inputs/novel` |
| `--output`, `-o` | Override the resolved output directory. | `outputs/novel` |
| `--model` | Target chat model for LangChain (OpenAI-compatible). | `gpt-4o-mini` |
| `--base-url` | Custom API base for OpenAI-compatible services. | `XNOVEL_BASE_URL` / `OPENAI_BASE_URL` |
| `--api-key` | Explicit API key for the chat provider. | `XNOVEL_API_KEY` / fallback |
| `--temperature` | Sampling temperature forwarded to the provider. | `0.0` |
| `--max-tokens` | Hard cap for LLM responses (if supported). | Unset |
| `--budget-usd` | Session budget in USD; 0 or unset disables hard limits. | Unset |
| `--budget-warn-ratio` | Threshold (0-1.0) for emitting warnings. | `0.9` |
| `--budget-hard` | Treat budget as a hard stop instead of soft warning. | `false` |
| `--track-costs/--no-track-costs` | Toggle accumulation of provider spend. | Enabled |

> **Tip**: If you prefer environment-driven configuration, set the variables listed below instead of passing flags explicitly.

## Environment Variables

| Variable | Purpose | Notes |
| -------- | ------- | ----- |
| `XNOVEL_INPUT_ROOT` | Default root for manuscript ingestion. | Mirrors `--input`.
| `XNOVEL_OUTPUT_ROOT` | Default root for enriched output. | Mirrors `--output`.
| `XNOVEL_MODEL` / `OPENAI_MODEL` | Preferred chat model name. | CLI `--model` takes precedence.
| `XNOVEL_BASE_URL` / `OPENAI_BASE_URL` | Base URL for OpenAI-compatible endpoints. | Supports self-hosted gateways.
| `XNOVEL_API_KEY` / `OPENAI_API_KEY` | Primary / fallback API keys. | You may also set `XNOVEL_API_KEY_ENV` to point to a custom variable.
| `XNOVEL_TEMPERATURE` | Default sampling temperature. | Float.
| `XNOVEL_MAX_TOKENS` | Maximum completion tokens. | Integer.
| `XNOVEL_BUDGET_USD` | Session budget in USD. | Float.
| `XNOVEL_BUDGET_WARN_RATIO` | Warn when spent >= ratio * budget. | Float; defaults to `0.9`.
| `XNOVEL_BUDGET_HARD` | Set to `true` to abort when budget is exceeded. | Defaults to `false`.

The helper methods in [`LLMConfig`](../src/xnovel/config.py) resolve these environment variables in the order listed above. Provider overrides apply equally to OpenAI-hosted accounts, Azure OpenAI resources, or third-party proxies (e.g., `https://api.fireworks.ai/v1`).

## Provider Configuration and Base URL Overrides

The pipeline constructs a LangChain `ChatOpenAI` client via `build_provider` in [`xnovel.llm.providers`](../src/xnovel/llm/providers.py). Key behaviors:

1. `XNOVEL_MODEL` -> `OPENAI_MODEL` -> built-in default (`gpt-4o-mini`).
2. `XNOVEL_BASE_URL` allows you to target OpenAI-compatible services (Bento, LM Studio, Fireworks, etc.).
3. API keys resolve from `XNOVEL_API_KEY`, falling back to `OPENAI_API_KEY`. Set `XNOVEL_API_KEY_ENV` when you want the loader to fetch credentials from a custom variable.
4. If you need custom network behavior (timeouts, retries), extend `LangChainChatProvider` or supply overrides when wiring up `XNovelConfig.as_provider_kwargs()`.

## Cost and Budget Tracking

`BudgetConfig` reads `XNOVEL_BUDGET_USD`, `XNOVEL_BUDGET_WARN_RATIO`, and `XNOVEL_BUDGET_HARD` to determine whether to warn or halt once an estimated spend crosses your threshold. The `--track-costs` CLI option controls whether token usage is recorded for post-run summaries. When `--budget-hard` (or `XNOVEL_BUDGET_HARD=true`) is set, the agent will raise an error rather than finishing the current task once the limit is breached.

## Optional Embeddings

The novel pipeline reuses the embedding utilities from the PDF OCR workflow. If you supply `EMBEDDING_BACKEND=local` (plus `LOCAL_EMBED_MODEL` and optional `LOCAL_EMBED_DEVICE`), retrieved context will be attached to each refinement call. Omitting embedding settings simply disables retrieval without failing the run.

## Output Structure

After a successful `run-all` invocation the output directory contains:

```
outputs/novel/
  ├── <manuscript-stem>.md        # Final, agent-refined Markdown
  ├── <manuscript-stem>_assets/   # Optional images or extracted figures (PDF input)
  └── run.json                    # (If cost tracking enabled) spend + metadata summary
```

Depending on your configuration, additional telemetry (chunk diagnostics, LangGraph state) may be persisted alongside the markdown files.

## Next Steps

- Inspect `src/xnovel/config.py` for programmatic access to the configuration dataclasses.
- Extend the CLI with project-specific prompts or guardrails before pointing it at larger manuscripts.
- Keep PDF OCR-specific instructions separate by continuing to consult the top-level `README.md`.
