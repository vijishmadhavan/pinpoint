# Architecture

This document describes the current shape of Pinpoint as implemented in the repo.

## Overview

Pinpoint has two main runtime pieces:

- a Python FastAPI backend
- a Node.js WhatsApp bot

The backend owns indexing, search, file operations, data analysis, media workflows, and background jobs. The bot turns natural-language messages into tool calls against the backend.

## Stability Classification

### Stable Core

- FastAPI backend and router structure
- lexical-first search path
- shared indexing pipeline
- watch folders and persistent background jobs
- direct file operations and data analysis endpoints

### Optional Integrations

- WhatsApp bot
- Gemini-backed OCR, media, captioning, fact extraction, and photo workflows
- Ollama-based bot fallback
- Google Workspace integration
- face-recognition paths with heavier dependencies

### Experimental / Non-default

- semantic search stages that are present but not the default path
- benchmark-only lexical comparison paths like BM25L experiments

## Main Components

### FastAPI backend

Entrypoints:

- [run_api.py](../run_api.py)
- [api/__init__.py](../api/__init__.py)

The backend starts a FastAPI app on port `5123` and loads `.env` at startup. If `API_SECRET` is set, requests must include `X-API-Secret`.

Main router modules:

- [api/core.py](../api/core.py): health, status, folder indexing, single-file indexing
- [api/search.py](../api/search.py): search, facts, document fetch, web read/search
- [api/files.py](../api/files.py): file listing, reading, moving, path registry, watch folders, background jobs
- [api/data.py](../api/data.py): calculations, Excel reads, pandas analysis
- [api/media.py](../api/media.py): image/video/audio search and OCR
- [api/photos.py](../api/photos.py): photo scoring, culling, grouping
- [api/transform.py](../api/transform.py): file/image/PDF transforms
- [api/faces.py](../api/faces.py): face detection, recognition, memory
- [api/memory.py](../api/memory.py): conversation and memory features
- [api/google.py](../api/google.py): Google Drive/Gmail integration

### WhatsApp bot

Entrypoint:

- [bot/index.js](../bot/index.js)

The bot uses Baileys for WhatsApp connectivity and Google GenAI or optional Ollama for the LLM loop. It reads the same repo-root `.env` and calls the backend on `http://localhost:5123`.

The bot’s tool surface and summaries live mainly in:

- [bot/src/tools.js](../bot/src/tools.js)
- [bot/src/llm.js](../bot/src/llm.js)
- [bot/src/skills.js](../bot/src/skills.js)

## Data Model

Primary storage is SQLite.

Key responsibilities in [database.py](../database.py):

- document metadata
- content storage
- chunks and chunk embeddings
- facts
- path registry
- generated files
- watched folders
- background jobs

Search is built on SQLite FTS5 plus optional semantic and rerank stages that are not the default production path.

## Indexing Pipeline

Shared indexing logic lives in:

- [indexing_service.py](../indexing_service.py)

Current indexing flow:

1. check whether the file is unchanged
2. extract text/content
3. upsert the document row
4. chunk the text
5. optionally embed chunks
6. optionally extract facts

This shared service is used by:

- single-file indexing
- folder indexing
- auto-index-on-read
- auto-index-on-analyze
- watched-folder indexing

## Search Pipeline

Search logic lives in:

- [search_pipeline.py](../search_pipeline.py)

Current default behavior:

- lexical-first search over FTS5
- metadata-aware lexical scoring
- ambiguity detection and clarification hints
- stage timing in responses

Important point:

- semantic expansion/embedding/rerank paths still exist
- they are not the default production path
- they are benchmarked and treated as optional/experimental

## Background Jobs

Persistent job tracking lives in:

- [job_service.py](../job_service.py)

Job-backed flows currently include:

- large folder indexing
- watched-folder initial indexing
- watched-folder auto-indexing
- path registry scans
- large image embedding prep

Job endpoints live in:

- [api/files.py](../api/files.py)

Important endpoints:

- `GET /background-jobs`
- `POST /background-jobs/{job_id}/cancel`

## Watch Folders

Watch-folder registration is handled in:

- [api/files.py](../api/files.py)

Current design:

- watched folders are persisted in SQLite
- initial indexing runs in the background
- periodic scans index eligible files
- unwatch requests cancel relevant in-flight jobs

## Evaluation and Load Testing

Search evaluation lives in:

- [evaluate_search.py](../evaluate_search.py)
- [benchmarks/README.md](../benchmarks/README.md)

Concurrent search load testing lives in:

- [load_test_search.py](../load_test_search.py)

These are used to validate search quality and latency before changing retrieval behavior.

## Current Product Shape

Pinpoint is currently best understood as:

- a local-first file search and automation backend
- with an optional WhatsApp assistant frontend

It is not yet packaged like a one-click consumer application. The current setup is still developer-oriented, even though the product surface is becoming cleaner.
