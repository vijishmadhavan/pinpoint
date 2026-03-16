# Stability Policy

This document defines how Pinpoint classifies features and interfaces.

## Stable

Stable means the feature is part of the expected day-to-day product surface and changes should be conservative.

Examples:

- lexical-first document search
- file read/list/move/rename/delete flows
- watch folders
- background job inspection and cancellation
- spreadsheet/data analysis endpoints
- packaged Python CLI surface

Expectations:

- covered by normal regression testing
- reflected in the README and troubleshooting docs
- changes should preserve behavior unless there is a clear reason to break it

## Optional

Optional means the feature is supported, but depends on extra setup, API keys, or heavier runtime dependencies.

Examples:

- WhatsApp bot
- Gemini-backed OCR/captioning/fact extraction
- Google Workspace integration
- face recognition
- media-heavy workflows

Expectations:

- should fail clearly when prerequisites are missing
- should be documented as requiring extra setup
- should not make the stable core unusable when unavailable

## Experimental

Experimental means the feature exists for evaluation or limited use, but is not the default production path.

Examples:

- non-default semantic document retrieval stages
- benchmark-only lexical comparison variants
- feature paths that are still being validated primarily through benchmarks

Expectations:

- may change or be removed based on evidence
- should not be presented as the recommended default path
- should be measured before being promoted into the stable core

## Release Guidance

When changing a feature:

- if it affects a stable feature, update tests and public docs
- if it affects an optional feature, update setup and troubleshooting docs when needed
- if it affects an experimental feature, update benchmark/evaluation notes before changing public claims
