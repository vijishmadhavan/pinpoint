# Photo Cull

## Can
- Score any photo on technical + aesthetic quality (/100 via Gemini vision)
- Auto-cull a folder: score all photos, keep top N%, move rejects to `_rejects`
- Generate an HTML gallery report with thumbnails (click opens original)
- Handle 500-50,000 photos efficiently (10 concurrent scores, cached)

## Cannot
- Delete photos (rejects are always moved, never deleted)
- Score non-image files (PDF, video, etc.)

## Tools
| Tool | What it does |
|---|---|
| score_photo | Score one photo — returns 7 sub-scores + total/100 + reasoning |
| cull_photos | Score all photos in folder, move bottom rejects — background job |
| cull_status | Poll progress of a running cull job |

## Scoring Rubric
| Category | Sub-score | Max |
|---|---|---|
| Technical | Sharpness | 15 |
| Technical | Exposure | 15 |
| Technical | Composition | 10 |
| Technical | Quality | 10 |
| Aesthetic | Emotion | 20 |
| Aesthetic | Interest | 15 |
| Aesthetic | Keeper | 15 |
| **Total** | | **100** |

## Workflow
1. `list_files` — survey folder, count images
2. Confirm with user: "Found N photos. Keep top 80%? Rejects go to `_rejects`."
3. `cull_photos(folder, keep_pct)` — starts background job
4. `cull_status(folder)` — poll every 5s until done
5. Report: kept/rejected counts, avg scores, threshold, send HTML report

## Post-Cull Follow-Ups
- If the user asks for the "best photo", "best bride and groom shot", "keeper", or similar AFTER culling, stay in the same folder and prefer `search_images_visual` to find the right visual match.
- Do NOT start with `search_documents(file_type="image")` for best-photo selection after culling. Caption search may find a matching photo, but it does not reflect the cull workflow or quality ranking.
- After finding the right match, send the image directly.

## Cost
~$0.045 per 1000 images (flash-lite, LOW resolution). 50K photos ~ $2.25.
