# Contributing

## Scope

Pinpoint is a local-first file search and automation tool with:

- a FastAPI backend
- a Node.js WhatsApp bot
- SQLite-backed indexing/search
- optional Gemini-powered features

Contributions should prefer correctness, clear behavior, and measured tradeoffs over adding more complexity.

## Local Setup

Create the documented Conda environment:

```bash
conda env create -f environment.yml
conda activate pinpoint
```

Install bot dependencies:

```bash
cd bot
npm install
```

Copy the environment template:

```bash
cp .env.example .env
```

If you want Gemini-backed features, fill in `GEMINI_API_KEY` in `.env`.

## Running

Backend:

```bash
conda run -n pinpoint python run_api.py
```

Bot + backend together:

```bash
./start.sh
```

## Tests

Run the backend suite:

```bash
conda run -n pinpoint python -m pytest tests/ -q
```

Useful targeted runs:

```bash
conda run -n pinpoint python -m pytest tests/test_search.py -q
conda run -n pinpoint python -m pytest tests/test_files.py tests/test_core.py tests/test_media.py -q
conda run -n pinpoint python -m pytest tests/test_evaluate_search.py tests/test_load_test_search.py -q
```

## Benchmarks

Search changes should be checked with the offline benchmark harness before being treated as improvements.

Examples:

```bash
python evaluate_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --output benchmarks/v4_mixed_report.json
python load_test_search.py --dataset benchmarks/search_relevance_v4_mixed.json --corpus benchmarks/corpus_v4_mixed --rounds 10 --concurrency 8 --output benchmarks/v4_mixed_load_report.json
```

## Contribution Rules

- Prefer lexical-first search changes unless benchmark data clearly justifies something heavier.
- Do not add silent `except Exception: pass` behavior to indexing/search/job flows.
- Keep background work visible through job status instead of ad hoc threads.
- Keep setup/docs aligned with actual code, not local assumptions.
- Avoid adding more repo-root clutter; put new docs in a deliberate location.

## Pull Request Expectations

Before opening a PR:

- tests should pass
- docs should be updated if behavior changed
- `.env.example` should be updated if config changed
- benchmark notes should be updated if search behavior changed

## Experimental Features

Some features are still optional or experimental, especially Gemini-heavy retrieval paths. If you change them, document:

- when they run
- how they fail
- what benchmark evidence supports them
