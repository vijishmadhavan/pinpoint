# Release Checklist

Use this before pushing a public-facing checkpoint or tagging a release.

## Code and Tests

- backend tests pass
- new features have regression coverage where practical
- no known failing focused suite is being ignored

Suggested command:

```bash
conda run -n pinpoint python -m pytest tests/ -q
```

## Setup and Config

- `README.md` matches the current startup flow
- `.env.example` matches the actual config keys used by the code
- `environment.yml` still reflects the intended baseline environment
- `CONTRIBUTING.md` still matches the real contributor workflow

## Search Changes

If search behavior changed:

- benchmark output was checked with `evaluate_search.py`
- load behavior was checked with `load_test_search.py` when relevant
- docs were updated if the default search behavior changed

## Background Jobs and Long-Running Work

If indexing/watch/media job behavior changed:

- `GET /background-jobs` still reflects the new behavior
- cancellation behavior still makes sense
- job progress fields remain coherent

## Open-Source Surface

- `LICENSE` exists
- root README is product-facing, not a dev scratchpad
- new internal notes are not dumped into the repo root unnecessarily

## Final Sanity

- no accidental secrets in tracked files
- no machine-specific instructions leaked into docs unless intentionally documented
- no stale benchmark/report claims that no longer match the code
