# Troubleshooting

This guide covers the most common current failure modes in Pinpoint.

## API does not start

Symptoms:

- `run_api.py` exits immediately
- `http://localhost:5123/ping` does not respond

Check:

1. The `pinpoint` environment exists and has the backend dependencies.
2. You are starting the backend with:

```bash
conda run -n pinpoint python run_api.py
```

3. Port `5123` is free.

Useful check:

```bash
curl http://localhost:5123/ping
```

Expected:

- `{"status":"ok"}`

## Bot does not connect

Symptoms:

- `start.sh` stops before the bot starts
- QR pairing does not appear
- bot cannot reach the backend

Check:

1. Node.js 20 is available through `nvm`.
2. Bot dependencies are installed in `bot/`:

```bash
cd bot
npm install
```

3. The backend is reachable at `http://localhost:5123`.
4. If `API_SECRET` is set, the bot and backend are reading the same `.env`.

## Gemini features do not work

Symptoms:

- search falls back to lexical-only behavior
- OCR/captioning/media features return Gemini-related errors
- bot warns that `GEMINI_API_KEY` is missing

Check:

1. `.env` exists at the repo root.
2. `GEMINI_API_KEY` is set.
3. `GEMINI_MODEL` is valid.

Important:

- many features degrade gracefully without Gemini
- core lexical document search still works without it
- semantic search is not the default production path, so a missing Gemini key should not break ordinary document search

## Search returns no useful results

Check:

1. The file was indexed.
2. The file path is allowed and readable.
3. The file type is supported by the current extractors.
4. You are searching with terms that actually appear in the document or metadata.

Useful endpoints:

- `POST /index-file`
- `GET /status`
- `GET /search`
- `GET /search-facts`

If a result set is ambiguous, the backend may intentionally return a clarification hint instead of pretending the top hit is certain.

## Watch folder is not indexing

Check:

1. The folder exists.
2. The path is not blocked by path-safety rules.
3. The folder is listed in:

- `GET /watched-folders`

4. The initial or follow-up background job is visible in:

- `GET /background-jobs`

5. The path scan status is visible in:

- `GET /path-scan-status`

Useful actions:

- re-add the watched folder
- inspect job status
- cancel a stuck job and retry

## Background job looks stuck

Check:

1. `GET /background-jobs`
2. the job `status`
3. the job `current_stage`
4. the job `details`
5. `total_items` vs `completed_items`

You can request cancellation with:

- `POST /background-jobs/{job_id}/cancel`

Common causes:

- very large folder walks
- expensive media/Gemini work
- environment/dependency mismatch

## Image / OCR / media features fail

Some features need more than the base Python environment.

Examples:

- Gemini API key for Gemini-backed OCR, captions, video/audio understanding, and photo workflows
- `tesseract` for local OCR
- `ffmpeg` for video/audio workflows
- optional heavy dependencies for face-related paths

If the core API works but media features do not, verify those optional dependencies separately.

This is expected to some degree because media, face, and Gemini-heavy flows are more optional and environment-sensitive than the stable core search/file APIs.

## Tests fail in a different environment

Use the documented environment:

```bash
conda env create -f environment.yml
conda activate pinpoint
conda run -n pinpoint python -m pytest tests/ -q
```

If tests hang or behave differently, first check:

- wrong Python environment
- missing FastAPI/test dependencies
- stale local `.env`
- external/background startup behavior

## Search quality changed after a code change

Do not trust intuition alone.

Run:

```bash
python evaluate_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --output benchmarks/v4_mixed_report.json
python load_test_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --rounds 10 --concurrency 8 --output benchmarks/v4_mixed_load_report.json
```

If the benchmark got worse, the change is probably not a real improvement.
