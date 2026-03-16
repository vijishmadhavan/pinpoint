# Pinpoint

[![Tests](https://github.com/vijishmadhavan/pinpoint/actions/workflows/test.yml/badge.svg)](https://github.com/vijishmadhavan/pinpoint/actions/workflows/test.yml)
[![Lint](https://github.com/vijishmadhavan/pinpoint/actions/workflows/lint.yml/badge.svg)](https://github.com/vijishmadhavan/pinpoint/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Search all your local files from WhatsApp. No cloud uploads. No tracking. Everything stays on your machine.

Pinpoint indexes your documents, PDFs, spreadsheets, images, and media into a local SQLite database, then lets you search and work with them through natural language — either via WhatsApp or direct API calls.

## What You Can Do

- **"Find the Sharma invoice from last month"** — searches across all your indexed documents instantly
- **"What's in that Excel in Downloads?"** — reads and analyzes spreadsheets, CSVs, PDFs on the fly
- **"Find everyone over 30 in the contacts spreadsheet"** — search, filter, group, sort within Excel/CSV files
- **"Create an Excel with these expense totals"** — generates spreadsheets, text files, charts from conversation
- **Send a purchase order image, ask "turn this into Excel"** — extracts tables from images/PDFs into spreadsheets
- **"Merge these 3 PDFs into one"** — merge, split PDFs, convert images to PDF and back
- **"Move all receipts to the Tax folder"** — batch file operations through conversation
- **"Group my wedding photos by category"** — AI classifies photos into groups and sorts them into folders
- **"Cull my camera roll — keep the best 80%"** — AI scores every photo for quality and separates rejects
- **"Who is this person?"** — remembers faces, recognizes them across photos later
- **"OCR this scanned document"** — extracts text from images and scanned PDFs
- **"Send me that PDF"** — sends files directly to your WhatsApp
- **Send a photo/file to the bot** — saves it to your PC, renames it, organizes into folders you choose. Important WhatsApp images and documents never get lost
- **"Create a folder called Tax 2025 and move all receipts there"** — creates folders and organizes files through conversation
- **"Find Sharma's phone number from that Excel"** — searches inside spreadsheets with smart phone/ID normalization (finds "920-889-6630" when you type "9208896630")
- **"Where's that chart I made yesterday?"** — searches files Pinpoint itself created in past conversations
- **"Send an email to john@company.com with the Q1 report"** — Gmail send, Calendar create, Drive upload (requires [gws CLI](https://github.com/googleworkspace/cli) + Google auth)
- **"Remind me to call the dentist at 3pm"** — persistent reminders that survive restarts and reconnects
- **"Remember that my car insurance expires in March"** — persistent memory across conversations
- **"Watch my Documents folder"** — auto-indexes new files as they appear
- **Voice messages** — transcribes and responds to audio messages

## How It Works

```
Your Files ──> Indexer ──> SQLite/FTS5 ──> Search API ──> WhatsApp Bot
   (local)     (extract)    (local DB)     (FastAPI)      (Gemini AI)
```

1. **Index** — Pinpoint extracts text from PDFs, Office docs, images (OCR), spreadsheets, and plain text files
2. **Search** — FTS5 full-text search with metadata-aware ranking and smart fallbacks
3. **Clarify** — When results are ambiguous (multiple similar matches), Pinpoint asks "which one do you mean?" instead of guessing wrong
4. **Act** — The WhatsApp bot turns your questions into tool calls: search, read, move, analyze, transform

Everything runs locally. The only external calls are to Gemini (for the AI layer) and optionally to web search providers.

**Important:** Search only finds files that have been indexed. Files get indexed when you:
- Explicitly index a file or folder (`/index-file`, `/index`)
- Watch a folder — new files are picked up every 60 minutes
- Read or analyze a file — auto-indexes in the background for future searches

Pinpoint does not scan your entire computer automatically. You control what gets indexed.

## Gets Smarter Over Time

Every interaction builds a local cache that makes future operations faster and cheaper:

- **Documents** — text, chunks, and embeddings stored after first index. Re-search is instant, no re-extraction.
- **Images** — embeddings cached after first search or group. Next time you search or group the same folder, cached images are free.
- **Videos** — frame embeddings stored per video. Searching the same video again costs nothing.
- **Photo scores** — culling scores cached by file mtime. Re-running cull on the same folder skips already-scored photos.
- **Photo classifications** — grouping results cached. Re-grouping reuses existing classifications.
- **Faces** — detected face data cached per image. Recognition on already-scanned photos is instant.
- **Facts** — extracted key facts stored per document. Fact search never re-extracts.
- **Search queries** — query expansion and reranking results cached. Repeated searches are free.

If you cancel a long job halfway (like embedding 1000 photos), the work already done is saved. Next run picks up where it left off.

## Memory System

Pinpoint has a 4-layer memory system that learns from everyday use:

**Conversation memory** — Keeps the last 50 messages per session. In the bot flow, long conversations are compacted instead of simply truncated, so important outcomes can survive even when older turns are compressed. Idle chats reset after 60 minutes.

**Persistent personal memory** — "Remember that my passport number is X12345." Stored permanently in SQLite, searchable with FTS5, and survives restarts. When you save a new fact, Gemini can decide to add it, update an existing memory, merge complementary details, ignore duplicates, or supersede a contradiction with an audit trail. You can also forget by description — "forget my old address" — without needing an internal ID.

**Document fact extraction** — When a file is indexed, Gemini extracts key facts such as names, dates, amounts, and topics, then stores them separately from the raw document text. You can search facts directly without reopening the full file.

**Face memory** — "Remember this is John." Saves face embeddings persistently so future face detection runs can recognize the same person across photos.

## Quick Start

### Backend only (search + file APIs)

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate pinpoint

# 2. Configure
cp .env.example .env
# Edit .env — add GEMINI_API_KEY for AI features (optional for basic search)

# 3. Start
python run_api.py
```

The API runs at `http://localhost:5123`. Try `http://localhost:5123/docs` for interactive API docs.

### With WhatsApp bot

```bash
# Install bot deps
cd bot && npm install && cd ..

# Start everything
./start.sh
```

Scan the QR code with WhatsApp to pair. Then just message your files.

## Common Things You Can Do

| Ask this | Pinpoint does this |
|---|---|
| "Find invoice 4821" | Searches indexed documents by content and filename |
| "Read the quarterly report PDF" | Extracts and returns the text |
| "Search for Sharma across all files" | Full-text search with ranked results |
| "Analyze the sales spreadsheet" | Loads Excel/CSV into pandas, runs queries |
| "Filter rows where amount > 5000" | Search, filter, group, sort within spreadsheets |
| "Create an Excel summary of Q1 expenses" | Generates new spreadsheets from conversation |
| (send purchase order image) "make this an Excel" | Extracts tables from images/PDFs into spreadsheets |
| "Merge invoice_1.pdf and invoice_2.pdf" | Merge, split PDFs |
| "Convert these images to a single PDF" | Images to PDF, PDF to images |
| "Make a bar chart of sales by month" | Generates charts from data |
| "Move old files to archive" | Batch file operations |
| "Watch my Downloads folder" | Auto-indexes new files every 60 minutes |
| "Find photos of the beach" | Visual image search across your photos |
| "OCR this scanned receipt" | Extracts text from images/scanned PDFs |
| "Group wedding photos by category" | AI classifies and sorts photos into folders |
| "Cull my camera roll, keep best 80%" | AI scores photos, separates rejects into a folder |
| "Who is this person?" | Face detection + recognition across photos |
| "Send me that report" | Sends the file to your WhatsApp chat |
| (send a photo/file to bot) | Saves to PC, renames, puts in your chosen folder |
| "Make a folder called Invoices 2025" | Creates folders on your computer |
| "Find Sharma's number from contacts.xlsx" | Searches inside spreadsheets, normalizes phone/ID formats |
| "Where's that chart I made yesterday?" | Searches files Pinpoint created in past conversations |
| "Email john@company.com the Q1 report" | Gmail send with attachment (needs gws CLI setup) |
| "Remind me at 5pm to call bank" | Persistent reminders — survive restarts |
| "Remember my passport number is X" | Stores in persistent memory |
| (voice message) | Transcribes audio and responds |

## What's Stable vs Optional

**Stable core** — works without any API keys:
- Document search (FTS5)
- File read/list/move/rename/delete
- Auto-indexing on file access
- Watch folders
- Background job tracking
- Data analysis (Excel, CSV)

**Optional** — needs Gemini API key or extra setup:
- WhatsApp bot (needs Gemini + WhatsApp pairing)
- OCR, captioning, fact extraction (Gemini-powered)
- Photo culling/scoring (Gemini Flash — vision judges quality, needs to "see" each photo)
- Photo grouping by category (Gemini Embedding 2 — cheap, classifies by similarity not vision)
- Visual image/video search (Gemini Embedding 2 — text-to-image similarity)
- Face recognition (needs insightface + GPU)
- Google Workspace — Gmail, Calendar, Drive (needs [gws CLI](https://github.com/googleworkspace/cli): `npm install -g @googleworkspace/cli && gws auth login`)
- Web search (needs Jina or LangSearch API key)

Note: Gemini Embedding 2 is used for image/video/photo features. Document text search uses FTS5 by default — embedding-based document search exists but is not the default path.

## Architecture

```
pinpoint/
  run_api.py              # Backend entrypoint (port 5123)
  api/                    # FastAPI routers
    core.py               #   health, indexing
    search.py             #   document search, facts, web read
    files.py              #   file ops, watch folders, background jobs
    data.py               #   Excel/CSV analysis
    media.py              #   image/video/audio search, OCR
    photos.py             #   photo scoring, culling, grouping
    faces.py              #   face detection and recognition
    transform.py          #   file/image/PDF transforms
    memory.py             #   conversation memory
    google.py             #   Google Workspace integration
  search_pipeline.py      # Search: FTS5, ranking, ambiguity detection
  indexing_service.py     # Shared index/chunk/embed pipeline
  job_service.py          # Persistent background job lifecycle
  database.py             # SQLite schema and helpers
  extractors.py           # Text extraction (PDF, Office, images, OCR)
  bot/
    index.js              # WhatsApp bot entrypoint
    src/tools.js          #   Gemini tool declarations
    src/llm.js            #   LLM loop (Gemini / Ollama)
    src/skills.js         #   Skill system for tool routing
```

For deeper details: [docs/architecture.md](docs/architecture.md)

## Repo Layout

The main product surface is intentionally small:

- `api/` — FastAPI routers and API-facing behavior
- `pinpoint/` — Python package and CLI entry points
- `bot/` — WhatsApp bot package
- `skills/` — skill markdowns shipped with the product
- `benchmarks/` — search evaluation datasets, reports, and benchmark scripts
- `tests/` — regression and packaging coverage
- `docs/` — product docs, troubleshooting, and release notes

Internal planning notes and downloaded comparison repos are kept out of the GitHub-facing product surface.

## Environment Variables

Copy `.env.example` to `.env`. Key variables:

| Variable | Required? | What it does |
|---|---|---|
| `GEMINI_API_KEY` | For AI features | Enables bot, OCR, media, photo workflows |
| `API_SECRET` | No | If set, all API requests need `X-API-Secret` header |
| `GEMINI_MODEL` | No | Defaults to `gemini-3.1-flash-lite-preview` |
| `OLLAMA_MODEL` | No | Use local Ollama instead of Gemini for bot |
| `JINA_API_KEY` | No | Enables web search via Jina |

## Tests

```bash
conda run -n pinpoint python -m pytest tests/ -q
```

The suite covers search, indexing, file operations, jobs, packaging, security, and API contracts.

## Benchmarks

Pinpoint includes offline search evaluation and load testing:

```bash
# Search quality
python evaluate_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed

# Concurrent load
python load_test_search.py --corpus benchmarks/corpus_v4_mixed --rounds 10 --concurrency 8
```

Current results on the mixed-domain offline benchmark: **94.4% success@1, perfect recall, ~4.3ms average query latency**. In a concurrent load test (concurrency=2, rounds=1): **~216 QPS, ~9.2ms average wall latency**.

See [benchmarks/README.md](benchmarks/README.md) for details.

## Docs

- [Architecture](docs/architecture.md) — how the system is built
- [Benchmark Summary](docs/benchmark-summary.md) — what Pinpoint has measured so far
- [Stability Policy](docs/stability-policy.md) — what Pinpoint treats as stable, optional, or experimental
- [Troubleshooting](docs/troubleshooting.md) — common issues and fixes
- [Release Checklist](docs/release-checklist.md) — what to check before pushing
- [Contributing](CONTRIBUTING.md) — how to contribute

## License

[MIT](LICENSE)
