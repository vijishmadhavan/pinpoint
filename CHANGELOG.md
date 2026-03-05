# Changelog

All notable changes to Pinpoint are documented here.

## [Unreleased]
- 230 integration tests, 66% coverage
- GitHub Actions CI (lint + test)
- Pre-commit hook (ruff + pytest)
- Move path consistency (video_embeddings, photo_classifications)

## 2025-12-15 — Software Maturity
- Linting: ruff (Python), eslint + prettier (JS)
- Security: `_check_safe()` path validation on all file endpoints
- Split `api.py` (3802 LOC) into 9 routers
- Split `bot/index.js` (3759 LOC) into 3 modules
- Python type hints on all function signatures (19 files)
- 98 initial pytest integration tests

## 2025-12-01 — Photo Cull & Group
- Photo auto-cull: Gemini vision scores photos /100 (tech + aesthetic)
- Photo auto-group: classify photos into user-defined categories
- DB cache for photo classifications (mtime-based)
- `suggest_categories`: sample 20 photos → Gemini suggests 4-8 groups
- Batch API (50% cheaper) for cull, context caching for group
- Structured outputs (`response_json_schema`) in all Gemini calls
- Cancellation (stop flag) + retry (429/503 backoff)

## 2025-11-15 — Native Gemini Video/Audio + Process Prompt
- Native Gemini video search (no FFmpeg extraction)
- `transcribe_audio` / `search_audio` tools
- OCR priority flip: Gemini first, Tesseract fallback
- Process-based system prompt (GATHER → ACT → ANSWER)
- Result sufficiency `_hint` fields across all endpoints
- Tool result summaries + cost circuit breaker ($0.10/message)
- Action ledger for structural truth enforcement
- MAX_ROUNDS 25 → 12, history 20 → 50 messages

## 2025-11-01 — Ollama + Intent Grouping
- Ollama local LLM adapter (Qwen 3.5-9B as Gemini alternative)
- Intent-based tool grouping: 60 tools → 17-28 per call

## 2025-10-15 — Memory & Intelligence
- Persistent memory (SQLite, /memory on|off)
- Web search (Jina Reader API)
- Video search (SigLIP2 + FFmpeg)
- Token optimization (TempStore refs, 500-char history cap)
- Gemini CLI patterns (loop detection, max rounds, efficiency prompt)
- Supermemory patterns (conflict detection, static/dynamic priority)
- Access control (/allow, /revoke), reminders, timezone (IST)
- Smarter memory (Mem0-inspired: LLM dedup, merge)
- Document chunking (Chonkie: RecursiveChunker, section-level search)
- Smart tools (result hints, DataFrame LRU cache, multi-sheet Excel)
- Face memory (known_faces table, remember_face/forget_face)
- Tool calling intelligence (SkillRL/ReCall patterns)

## 2025-09-15 — Core Features
- PDF, DOCX, XLSX, PPTX, EPUB, TXT, CSV extractors
- SQLite FTS5 database + BM25 search
- FastAPI server (port 5123)
- WhatsApp bot (Baileys) + file sending
- Gemini AI + tool calling (64 tools)
- Conversation memory (last 20 msgs, 60min idle reset)
- Face recognition (InsightFace)
- SKILL.MD system + 23 skill files
- Extended powers (write, chart, Excel, PDF, image, zip, download)
- Query expansion + RRF (Gemini-powered)
- Visual image search (SigLIP2 ONNX, text-to-image similarity)
- WhatsApp UX polish (markdown→WA, read receipts, debounce)

## 2025-09-01 — Initial Release
- First commit with full source code
