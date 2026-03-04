# Photo Grouping

## Can
- Auto-discover categories from sample photos (Gemini suggests 4-8 groups)
- Auto-group ALL photos in a folder via Gemini vision classification
- Classifications cached in DB — re-runs skip already-classified photos (free)
- Handle 500-50,000 photos (10 concurrent Gemini calls, background job)
- Generate an HTML gallery report grouped by category (click opens original)

## Cannot
- Delete photos (always moved to subfolders, never deleted)
- Group non-image files (video, PDF, etc.)

## Tools
| Tool | What it does |
|---|---|
| suggest_categories | Sample ~20 photos, Gemini suggests 4-8 category names |
| group_photos | Classify ALL photos in folder into categories — background job |
| group_status | Poll progress of a running group job |

## Workflow
1. `list_files` — survey folder, count images
2. `suggest_categories(folder)` — Gemini auto-discovers categories
3. Confirm with user: "Found N photos. Group into [categories]? Unmatched go to `_uncategorized`."
4. `group_photos(folder, categories)` — starts background job (cached = free re-runs)
5. `group_status(folder)` — poll every 5s until done
6. Report: group counts per category, send HTML report

## Cost
~$0.045 per 1000 images (flash-lite, LOW resolution). Re-runs: $0 (cached).
