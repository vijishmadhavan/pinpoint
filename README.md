# Pinpoint

Pinpoint is a local-first search and file-ops tool with a FastAPI backend and an optional WhatsApp bot frontend.

It indexes documents, images, spreadsheets, PDFs, and other local files into SQLite, then exposes search, read, move, transform, and media-analysis APIs. The bot layer uses WhatsApp plus Gemini to turn natural-language requests into tool calls.

## What It Does

- Search local documents with lexical-first retrieval over SQLite FTS5.
- Read files directly from disk, including text, PDF, Office docs, Excel, and images.
- Auto-index on file reads and data analysis requests.
- Watch folders for ongoing indexing.
- Track background work with persistent jobs and cancellation.
- Search and process images, video, audio, OCR, and photo workflows.
- Expose everything as HTTP APIs and optional WhatsApp tools.

## Stability Guide

### Stable Core

These are the most reliable and most product-defining parts of the current codebase:

- FastAPI backend
- lexical-first document search over SQLite FTS5
- metadata-aware ranking and ambiguity hints
- direct file read/list/move/search operations
- shared indexing pipeline
- persistent background jobs and cancellation

### Optional Integrations

These work, but depend on extra environment setup or external services:

- WhatsApp bot
- Gemini-backed media, OCR, captioning, fact extraction, and photo workflows
- Ollama bot fallback
- web search provider integrations
- Google Workspace integration via `gws` CLI
- face-related workflows that need heavier optional dependencies

### Experimental or Non-default

These exist in the codebase but are not the default production path:

- semantic query expansion / embedding / rerank stages in search
- benchmark-only BM25L comparison paths
- some heavier multimodal and media flows that are more environment-sensitive than core search/file use

## Current Search Design

The default production path is lexical-first:

- FTS5 is the default search path.
- Metadata-aware lexical boosts are applied for filenames, titles, paths, and identifier-like matches.
- Ambiguous result sets return clarification hints instead of pretending the top hit is certain.
- Semantic stages still exist in the codebase, but they are not the default path.

This repo also includes offline benchmark and load-test harnesses under [benchmarks/README.md](benchmarks/README.md).

## Architecture

- Python 3.11 backend: FastAPI + SQLite/FTS5
- Node.js 20 bot: Baileys + Google GenAI SDK
- Shared `.env` file at repo root
- FastAPI entrypoint: [run_api.py](run_api.py)
- Bot entrypoint: [bot/index.js](bot/index.js)
- Combined starter: [start.sh](start.sh)

Important backend modules:

- [api/__init__.py](api/__init__.py): app setup and auth middleware
- [api/core.py](api/core.py): health, status, folder indexing, single-file indexing
- [api/search.py](api/search.py): document search, facts search, document fetch, web read/search
- [api/files.py](api/files.py): file listing, reading, moving, watch folders, background jobs
- [api/media.py](api/media.py): image/video/audio search and OCR
- [api/photos.py](api/photos.py): photo scoring, culling, grouping
- [search_pipeline.py](search_pipeline.py): search pipeline, ambiguity detection, timing
- [indexing_service.py](indexing_service.py): shared file indexing pipeline
- [job_service.py](job_service.py): persistent background job lifecycle

Supporting docs:

- [docs/architecture.md](docs/architecture.md)
- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/release-checklist.md](docs/release-checklist.md)

## Requirements

- Python 3.11
- Node.js 20
- A working Python environment with the repo dependencies installed
- A working Node install for the bot dependencies in [bot/package.json](bot/package.json)

Important note:

- [start.sh](start.sh) currently expects a Conda environment named `pinpoint`.
- The repo now includes [environment.yml](environment.yml) as the baseline local development environment.

## Environment Variables

Both the Python backend and Node bot load the repo root `.env`.

Start from:

```bash
cp .env.example .env
```

Common variables:

- `API_SECRET`: optional API auth secret. If set, requests must send `X-API-Secret`.
- `GEMINI_API_KEY`: enables Gemini-powered features such as optional semantic indexing, OCR fallback, captions, audio/video/photo workflows, and bot reasoning.
- `GEMINI_MODEL`: defaults to `gemini-3.1-flash-lite-preview`.

Optional web search variables:

- `LANGSEARCH_API_KEY`
- `JINA_API_KEY`

Optional bot variable:

- `OLLAMA_MODEL`: use Ollama instead of Gemini for the bot LLM loop.

## Running the Backend

If you already have the project’s Conda env:

```bash
conda run -n pinpoint python run_api.py
```

To create it from scratch:

```bash
conda env create -f environment.yml
conda activate pinpoint
```

The API starts on `http://localhost:5123`.

Interactive docs:

- `http://localhost:5123/docs`

Health check:

```bash
curl http://localhost:5123/ping
```

## Running the WhatsApp Bot

Install bot dependencies:

```bash
cd bot
npm install
```

Then start the full stack from repo root:

```bash
./start.sh
```

What `start.sh` does:

- activates Node 20 via `nvm`
- starts the FastAPI backend with `conda run -n pinpoint python run_api.py`
- starts the WhatsApp bot in the foreground

## API Highlights

Search and read:

- `GET /search`
- `GET /search-facts`
- `GET /document/{id}`
- `GET /web-read`
- `POST /read_file`

Indexing and status:

- `GET /ping`
- `GET /status`
- `POST /index`
- `GET /indexing/status`
- `POST /index-file`

File and folder operations:

- `GET /list_files`
- `GET /file_info`
- `POST /grep`
- move / batch move / rename / delete endpoints in [api/files.py](api/files.py)

Watch folders and jobs:

- `POST /watch-folder`
- `POST /unwatch-folder`
- `GET /watched-folders`
- `GET /background-jobs`
- `POST /background-jobs/{job_id}/cancel`

Media and photo workflows:

- `POST /search-images-visual`
- `POST /search-video`
- `POST /search-audio`
- `POST /ocr`
- `POST /score-photo`
- `POST /cull-photos`
- `POST /suggest-categories`
- `POST /group-photos`

## Background Jobs

Long-running operations are persisted in SQLite-backed background jobs.

Job records include:

- job type
- target path
- status
- current stage
- item counts
- structured details
- timestamps
- failure reason

Current job-backed flows include:

- large folder indexing
- watched-folder initial indexing
- watched-folder auto-indexing
- path registry scans
- large image embedding/search prep

These job-backed flows are part of the current stable backend surface.

## Benchmarks and Load Testing

Offline search evaluation:

```bash
python evaluate_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --output benchmarks/v4_mixed_report.json
```

Concurrent lexical-first load testing:

```bash
python load_test_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --rounds 10 --concurrency 8 --output benchmarks/v4_mixed_load_report.json
```

More benchmark details live in [benchmarks/README.md](benchmarks/README.md).

## Tests

Run the focused backend suite in the project env:

```bash
conda run -n pinpoint python -m pytest tests/ -q
```

Useful targeted runs:

```bash
conda run -n pinpoint python -m pytest tests/test_search.py -q
conda run -n pinpoint python -m pytest tests/test_evaluate_search.py tests/test_load_test_search.py -q
```

## Open-Source Status

What is in relatively good shape:

- lexical-first search path
- background job persistence and cancellation
- benchmark and load-test harnesses
- a real test suite for the main backend flows
- a clearer stable/optional split than before

What still needs hardening for broader external use:

- deployment docs beyond the local Conda-based flow
- cleanup of some optional Gemini-heavy features and their dependencies
- more polished end-user docs for every tool surface

## License

This repo now includes an MIT license in [LICENSE](LICENSE).
