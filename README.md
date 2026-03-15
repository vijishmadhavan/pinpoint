# Pinpoint

[![Tests](https://github.com/vijishmadhavan/pinpoint/actions/workflows/test.yml/badge.svg)](https://github.com/vijishmadhavan/pinpoint/actions/workflows/test.yml)
[![Lint](https://github.com/vijishmadhavan/pinpoint/actions/workflows/lint.yml/badge.svg)](https://github.com/vijishmadhavan/pinpoint/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Search all your local files from WhatsApp. No cloud uploads. No tracking. Everything stays on your machine.

Pinpoint indexes your documents, PDFs, spreadsheets, images, and media into a local SQLite database, then lets you search and work with them through natural language — either via WhatsApp or direct API calls.

## What You Can Do

- **"Find the Sharma invoice from last month"** — searches across all your indexed documents instantly
- **"What's in that Excel in Downloads?"** — reads and analyzes spreadsheets, CSVs, PDFs on the fly
- **"Move all receipts to the Tax folder"** — batch file operations through conversation
- **"Group my wedding photos by category"** — AI-powered photo organization and culling
- **"Who is this person?"** — remembers faces, recognizes them across photos later
- **"OCR this scanned document"** — extracts text from images and scanned PDFs
- **"Send me that PDF"** — sends files directly to your WhatsApp
- **Send a photo/file to the bot** — saves it to your PC, renames it, organizes into folders you choose. Important WhatsApp images and documents never get lost
- **"Create a folder called Tax 2025 and move all receipts there"** — creates folders and organizes files through conversation
- **"Remind me to call the dentist at 3pm"** — sets one-time or recurring reminders
- **"Remember that my car insurance expires in March"** — persistent memory across conversations
- **"Watch my Documents folder"** — auto-indexes new files as they appear
- **Voice messages** — transcribes and responds to audio messages

## How It Works

```
Your Files ──> Indexer ──> SQLite/FTS5 ──> Search API ──> WhatsApp Bot
   (local)     (extract)    (local DB)     (FastAPI)      (Gemini AI)
```

1. **Index** — Pinpoint extracts text from PDFs, Office docs, images (OCR), spreadsheets, and plain text files
2. **Search** — FTS5 full-text search with metadata-aware ranking, ambiguity detection, and smart fallbacks
3. **Act** — The WhatsApp bot turns your questions into tool calls: search, read, move, analyze, transform

Everything runs locally. The only external calls are to Gemini (for the AI layer) and optionally to web search providers.

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
| "Move old files to archive" | Batch file operations |
| "Watch my Downloads folder" | Auto-indexes new files every 60 minutes |
| "Find photos of the beach" | Visual image search across your photos |
| "OCR this scanned receipt" | Extracts text from images/scanned PDFs |
| "Group wedding photos by category" | AI classifies and sorts photos into folders |
| "Who is this person?" | Face detection + recognition across photos |
| "Send me that report" | Sends the file to your WhatsApp chat |
| (send a photo/file to bot) | Saves to PC, renames, puts in your chosen folder |
| "Make a folder called Invoices 2025" | Creates folders on your computer |
| "Remind me at 5pm to call bank" | Sets a reminder, delivers via WhatsApp |
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
- Photo scoring, culling, grouping (Gemini vision)
- Image/video/audio search
- Face recognition (needs insightface + GPU)
- Web search (needs Jina or LangSearch API key)

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

277+ tests covering search, indexing, file operations, jobs, security, and API contracts.

## Benchmarks

Pinpoint includes offline search evaluation and load testing:

```bash
# Search quality
python evaluate_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed

# Concurrent load
python load_test_search.py --corpus benchmarks/corpus_v4_mixed --rounds 10 --concurrency 8
```

Current results on mixed-domain benchmark: **94% success@1, perfect recall, 4ms latency, 216 QPS**.

See [benchmarks/README.md](benchmarks/README.md) for details.

## Docs

- [Architecture](docs/architecture.md) — how the system is built
- [Troubleshooting](docs/troubleshooting.md) — common issues and fixes
- [Release Checklist](docs/release-checklist.md) — what to check before pushing
- [Contributing](CONTRIBUTING.md) — how to contribute

## License

[MIT](LICENSE)
