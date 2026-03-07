"""
Pinpoint — Photo Auto-Cull (Photographer Power)

Gemini vision scores photos on technical + aesthetic quality (/100).
Rejects are MOVED to a separate folder, never deleted.

Scoring rubric:
  Technical (50): Sharpness/15, Exposure/15, Composition/10, Quality/10
  Aesthetic (50): Emotion/20, Interest/15, Keeper/15
  Total: /100 + reasoning

HTML report: thumbnail gallery (base64 tiny JPEGs), click opens original.
"""

from __future__ import annotations

import html as html_mod
import io
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from database import DB_PATH, get_db
from extractors import IMAGE_EXTENSIONS, _get_gemini, _preprocess_image, gemini_call_with_retry

# --- SQLite cache ---

_db_lock = threading.RLock()  # RLock: reentrant — _get_conn() + callers both acquire


def _init_table(conn: Any) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photo_scores (
            path TEXT PRIMARY KEY,
            mtime REAL,
            sharpness INTEGER, exposure INTEGER, composition INTEGER, quality INTEGER,
            emotion INTEGER, interest INTEGER, keeper INTEGER,
            total INTEGER,
            reasoning TEXT,
            scored_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photo_classifications (
            path TEXT PRIMARY KEY,
            mtime REAL,
            category TEXT,
            classified_at TEXT
        )
    """)
    conn.commit()


_db_conn = None


def _get_conn() -> Any:
    global _db_conn
    with _db_lock:
        if _db_conn is None:
            _db_conn = get_db(DB_PATH)
            _init_table(_db_conn)
        return _db_conn


_gemini_call_with_retry = gemini_call_with_retry  # alias for backward compat


def _cached_score(conn: Any, path: str, mtime: float) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM photo_scores WHERE path = ? AND mtime = ?", (path, mtime)).fetchone()
    if row:
        return dict(row)
    return None


# --- Gemini scoring ---

_SCORE_PROMPT = """Rate this photo's quality.

Scoring:
- Technical (50): Sharpness/15 (focus, detail), Exposure/15 (lighting, dynamic range), Composition/10 (framing, rule of thirds), Quality/10 (noise, artifacts, resolution)
- Aesthetic (50): Emotion/20 (mood, expression, story), Interest/15 (uniqueness, eye-catching), Keeper/15 (would a photographer keep this?)
- total = sum of all 7 sub-scores

Be strict: most casual photos score 40-60. Only exceptional shots score 80+."""

_SCORE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sharpness": {"type": "INTEGER"},
        "exposure": {"type": "INTEGER"},
        "composition": {"type": "INTEGER"},
        "quality": {"type": "INTEGER"},
        "emotion": {"type": "INTEGER"},
        "interest": {"type": "INTEGER"},
        "keeper": {"type": "INTEGER"},
        "total": {"type": "INTEGER"},
        "reasoning": {"type": "STRING"},
    },
    "required": [
        "sharpness",
        "exposure",
        "composition",
        "quality",
        "emotion",
        "interest",
        "keeper",
        "total",
        "reasoning",
    ],
}


def score_photo(path: str) -> dict[str, Any]:
    """Score a single photo. Returns breakdown with _hint."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return {"error": f"Not an image: {ext}"}

    mtime = os.path.getmtime(path)

    # Check cache
    with _db_lock:
        conn = _get_conn()
        cached = _cached_score(conn, path, mtime)
    if cached:
        cached["_hint"] = f"Score: {cached['total']}/100 — {cached['reasoning']}"
        cached["cached"] = True
        return cached

    # Score with Gemini
    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set"}

    try:
        from google.genai import types
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img = _preprocess_image(img, 384)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_bytes = buf.getvalue()
        buf.close()
        img.close()

        model = os.environ.get("GEMINI_MODEL_LITE", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"))
        response = _gemini_call_with_retry(
            client,
            model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=_SCORE_PROMPT),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                response_mime_type="application/json",
                response_json_schema=_SCORE_SCHEMA,
            ),
        )

        scores = json.loads(response.text)

        # Clamp sub-scores to valid ranges
        fields = {
            "sharpness": 15,
            "exposure": 15,
            "composition": 10,
            "quality": 10,
            "emotion": 20,
            "interest": 15,
            "keeper": 15,
        }
        for f, mx in fields.items():
            scores[f] = max(0, min(mx, int(scores.get(f, 0))))
        scores["total"] = sum(scores[f] for f in fields)
        scores["reasoning"] = str(scores.get("reasoning", ""))[:200]

        # Save to cache
        with _db_lock:
            conn = _get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO photo_scores
                (path, mtime, sharpness, exposure, composition, quality, emotion, interest, keeper, total, reasoning, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
                (
                    path,
                    mtime,
                    scores["sharpness"],
                    scores["exposure"],
                    scores["composition"],
                    scores["quality"],
                    scores["emotion"],
                    scores["interest"],
                    scores["keeper"],
                    scores["total"],
                    scores["reasoning"],
                ),
            )
            conn.commit()

        # Index reasoning as caption for future search (free — Gemini already saw the image)
        if scores.get("reasoning"):
            _index_caption(path, scores["reasoning"])

        scores["path"] = path
        scores["cached"] = False
        scores["_hint"] = f"Score: {scores['total']}/100 — {scores['reasoning']}"
        return scores

    except Exception as e:
        return {"error": f"Scoring failed: {e}", "path": path}


# --- Background cull jobs ---

_cull_jobs = {}  # folder → progress dict
_cull_lock = threading.Lock()


def _make_thumbnail_b64(path: str, size: int = 160) -> str | None:
    """Create a tiny JPEG thumbnail as base64 for the HTML report."""
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        img.close()
        import base64

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        buf.close()
        return b64
    except Exception:
        return None


def _generate_html_report(
    folder: str,
    scored_photos: list[dict[str, Any]],
    kept_paths: list[str],
    rejected_paths: list[str],
    rejects_folder: str,
    threshold: int,
) -> str:
    """Generate a lightweight HTML gallery with thumbnails. Click opens original."""
    # Sort by score descending
    scored_photos.sort(key=lambda x: x.get("total", 0), reverse=True)

    kept_set = set(kept_paths)
    avg_kept = sum(s["total"] for s in scored_photos if s["path"] in kept_set) / max(len(kept_paths), 1)
    avg_rejected = sum(s["total"] for s in scored_photos if s["path"] not in kept_set) / max(len(rejected_paths), 1)

    rows = []
    for s in scored_photos:
        is_kept = s["path"] in kept_set
        thumb = _make_thumbnail_b64(s["path"])
        if not thumb:
            continue
        # For rejected photos, link to the new location in rejects folder
        original_path = s["path"]
        if not is_kept and rejects_folder:
            moved_path = os.path.join(rejects_folder, os.path.basename(s["path"]))
            if os.path.exists(moved_path):
                original_path = moved_path

        status_class = "kept" if is_kept else "rejected"
        status_label = "KEPT" if is_kept else "REJECTED"
        color = "#2d7" if is_kept else "#e55"

        esc_name = html_mod.escape(os.path.basename(s["path"]))
        esc_reason = html_mod.escape(s.get("reasoning", ""))
        esc_file_url = html_mod.escape(original_path.replace(chr(92), "/"))
        rows.append(f"""<div class="card {status_class}" onclick="window.open('file:///{esc_file_url}')">
  <img src="data:image/jpeg;base64,{thumb}" alt="{esc_name}">
  <div class="score" style="background:{color}">{s["total"]}</div>
  <div class="info">
    <b>{esc_name}</b>
    <span class="badge" style="color:{color}">{status_label}</span>
    <div class="breakdown">S{s.get("sharpness", 0)} E{s.get("exposure", 0)} C{s.get("composition", 0)} Q{s.get("quality", 0)} | Em{s.get("emotion", 0)} I{s.get("interest", 0)} K{s.get("keeper", 0)}</div>
    <div class="reason">{esc_reason}</div>
  </div>
</div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Pinpoint Photo Cull — {html_mod.escape(os.path.basename(folder))}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#eee;padding:20px}}
h1{{font-size:1.4em;margin-bottom:4px}}
.stats{{color:#aaa;margin-bottom:20px;font-size:.9em}}
.stats b{{color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}}
.card{{background:#16213e;border-radius:8px;overflow:hidden;cursor:pointer;transition:transform .15s;position:relative}}
.card:hover{{transform:scale(1.03)}}
.card img{{width:100%;aspect-ratio:1;object-fit:cover;display:block}}
.score{{position:absolute;top:8px;right:8px;font-weight:700;font-size:1.1em;padding:2px 8px;border-radius:12px;color:#fff}}
.info{{padding:8px}}
.info b{{font-size:.8em;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.badge{{font-size:.7em;font-weight:700}}
.breakdown{{font-size:.7em;color:#888;margin-top:2px}}
.reason{{font-size:.7em;color:#aaa;margin-top:3px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
.card.rejected{{opacity:.7}} .card.rejected:hover{{opacity:1}}
.filter-bar{{margin-bottom:16px;display:flex;gap:8px}}
.filter-bar button{{padding:6px 16px;border:none;border-radius:6px;cursor:pointer;font-size:.85em;background:#16213e;color:#eee}}
.filter-bar button.active{{background:#0f3460;color:#fff}}
</style></head><body>
<h1>Photo Cull Report — {os.path.basename(folder)}</h1>
<div class="stats">
  <b>{len(kept_paths)}</b> kept (avg {avg_kept:.0f}) · <b>{len(rejected_paths)}</b> rejected (avg {avg_rejected:.0f}) · threshold <b>{threshold}</b>/100 · {len(scored_photos)} total
</div>
<div class="filter-bar">
  <button class="active" onclick="filterCards('all',this)">All</button>
  <button onclick="filterCards('kept',this)">Kept</button>
  <button onclick="filterCards('rejected',this)">Rejected</button>
</div>
<div class="grid">
{"".join(rows)}
</div>
<script>
function filterCards(f,btn){{
  document.querySelectorAll('.filter-bar button').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c=>{{
    if(f==='all')c.style.display='';
    else c.style.display=c.classList.contains(f)?'':'none';
  }});
}}
</script>
</body></html>"""

    report_path = os.path.join(folder, "_cull_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def cull_photos(folder: str, keep_pct: int = 80, rejects_folder: str | None = None) -> dict[str, Any]:
    """Start background photo culling. Returns immediately with status."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return {"error": f"Folder not found: {folder}"}

    # Collect images
    images = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(folder, f))
    if not images:
        return {"error": "No images found in folder"}

    if not rejects_folder:
        rejects_folder = os.path.join(folder, "_rejects")

    keep_pct = max(1, min(99, keep_pct))

    # Init progress
    progress = {
        "status": "scoring",
        "folder": folder,
        "total": len(images),
        "scored": 0,
        "errors": 0,
        "keep_pct": keep_pct,
        "started_at": time.time(),
        "eta_seconds": None,
        "report_path": None,
    }
    with _cull_lock:
        _cull_jobs[folder] = progress

    def _run() -> None:
        scored = []
        start = time.time()

        # First: score cached photos (free, no API calls)
        uncached_images = []
        with _db_lock:
            conn = _get_conn()
            for img in images:
                abs_path = os.path.abspath(img)
                try:
                    mtime = os.path.getmtime(abs_path)
                except OSError:
                    progress["errors"] += 1
                    progress["scored"] += 1
                    continue
                cached = _cached_score(conn, abs_path, mtime)
                if cached:
                    cached["path"] = abs_path
                    cached["cached"] = True
                    scored.append(cached)
                    progress["scored"] += 1
                else:
                    uncached_images.append(img)

        if scored:
            print(f"[Cull] {len(scored)} from cache, {len(uncached_images)} need scoring")

        # Score uncached photos via concurrent API calls
        if uncached_images:
            print(f"[Cull] Scoring {len(uncached_images)} photos...")
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = {pool.submit(score_photo, img): img for img in uncached_images}
                for future in as_completed(futures):
                    if progress.get("stop"):
                        pool.shutdown(wait=False, cancel_futures=True)
                        progress["status"] = "cancelled"
                        progress["kept"] = len(scored)
                        progress["elapsed_seconds"] = round(time.time() - start, 1)
                        return
                    result = future.result()
                    if "error" in result:
                        progress["errors"] += 1
                    else:
                        scored.append(result)
                    progress["scored"] += 1
                    elapsed = time.time() - start
                    remaining = progress["total"] - progress["scored"]
                    rate = progress["scored"] / max(elapsed, 0.1)
                    progress["eta_seconds"] = round(remaining / max(rate, 0.01))

        if not scored:
            progress["status"] = "error"
            progress["error"] = "No photos could be scored"
            return

        # Rank and split
        progress["status"] = "moving"
        scored.sort(key=lambda x: x["total"], reverse=True)
        keep_count = max(1, round(len(scored) * keep_pct / 100))
        kept = scored[:keep_count]
        rejected = scored[keep_count:]
        threshold = rejected[0]["total"] if rejected else scored[-1]["total"]

        kept_paths = [s["path"] for s in kept]
        rejected_paths = [s["path"] for s in rejected]

        # Generate HTML report BEFORE moving (thumbnails need original paths)
        progress["status"] = "report"
        report_path = _generate_html_report(folder, scored, kept_paths, rejected_paths, rejects_folder, threshold)

        # Move rejects
        moved = 0
        if rejected:
            os.makedirs(rejects_folder, exist_ok=True)
            for s in rejected:
                try:
                    dest = os.path.join(rejects_folder, os.path.basename(s["path"]))
                    # Avoid overwriting files with duplicate basenames
                    if os.path.exists(dest):
                        base, ext = os.path.splitext(os.path.basename(s["path"]))
                        counter = 1
                        while os.path.exists(dest):
                            dest = os.path.join(rejects_folder, f"{base}_{counter}{ext}")
                            counter += 1
                    shutil.move(s["path"], dest)
                    moved += 1
                except Exception as e:
                    print(f"[Cull] Move failed: {s['path']} — {e}")
                    progress["errors"] += 1

        # Final stats
        progress["status"] = "done"
        progress["kept"] = len(kept)
        progress["rejected"] = moved
        progress["threshold"] = threshold
        progress["avg_kept"] = round(sum(s["total"] for s in kept) / max(len(kept), 1), 1)
        progress["avg_rejected"] = (
            round(sum(s["total"] for s in rejected) / max(len(rejected), 1), 1) if rejected else 0
        )
        progress["rejects_folder"] = rejects_folder
        progress["report_path"] = report_path
        progress["elapsed_seconds"] = round(time.time() - start, 1)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "started": True,
        "folder": folder,
        "total_images": len(images),
        "keep_pct": keep_pct,
        "rejects_folder": rejects_folder,
        "_hint": f"Culling {len(images)} photos (keep top {keep_pct}%). Use cull_status to check progress.",
    }


def get_cull_status(folder: str, cancel: bool = False) -> dict[str, Any]:
    """Get culling progress for a folder. Set cancel=True to stop the job."""
    folder = os.path.abspath(folder)
    with _cull_lock:
        progress = _cull_jobs.get(folder)
    if not progress:
        return {"error": "No cull job found for this folder", "_hint": "Start one with cull_photos first."}

    if cancel and progress["status"] not in ("done", "cancelled", "error"):
        progress["stop"] = True
        return {
            "status": "cancelling",
            "folder": folder,
            "_hint": "Cancellation requested. Check status again shortly.",
        }

    result = dict(progress)
    if result["status"] == "done":
        result["_hint"] = (
            f"Done! Kept {result['kept']} (avg {result['avg_kept']}), "
            f"rejected {result['rejected']} (avg {result['avg_rejected']}), "
            f"threshold {result['threshold']}/100. "
            f"Report: {result.get('report_path', 'N/A')}"
        )
    elif result["status"] == "scoring":
        pct = round(result["scored"] / max(result["total"], 1) * 100)
        eta = result.get("eta_seconds")
        eta_str = f", ~{eta}s left" if eta else ""
        result["_hint"] = f"Scoring: {result['scored']}/{result['total']} ({pct}%){eta_str}"
    elif result["status"] == "moving":
        result["_hint"] = "Moving rejects to folder..."
    elif result["status"] == "report":
        result["_hint"] = "Generating HTML report..."
    elif result["status"] == "cancelled":
        result["_hint"] = f"Cancelled after scoring {result.get('scored', 0)}/{result.get('total', 0)} photos."

    return result


# --- Background group jobs ---

_group_jobs = {}  # folder → progress dict
_group_lock = threading.Lock()

_CLASSIFY_PROMPT_TEMPLATE = """Classify this photo into EXACTLY ONE of these categories:
{categories}

Also write a short caption (10-20 words) describing what's in the photo."""


def _cached_classification(conn: Any, path: str, mtime: float) -> str | None:
    row = conn.execute(
        "SELECT category FROM photo_classifications WHERE path = ? AND mtime = ?", (path, mtime)
    ).fetchone()
    if row:
        return row["category"]
    return None


def _get_indexed_caption(conn: Any, path: str) -> str | None:
    """Check if this photo is already indexed in documents DB → return caption if so."""
    row = conn.execute("SELECT hash FROM documents WHERE path = ? AND active = 1", (path,)).fetchone()
    if row:
        content_row = conn.execute("SELECT text FROM content WHERE hash = ?", (row["hash"],)).fetchone()
        if content_row and content_row["text"]:
            return content_row["text"]
    return None


def _classify_photo(path: str, categories: list[str], categories_lower: list[str]) -> dict[str, Any]:
    """Classify a single photo. Priority: cache → caption (text-only Gemini) → Gemini vision."""
    abs_path = os.path.abspath(path)
    mtime = os.path.getmtime(abs_path)

    with _db_lock:
        conn = _get_conn()

        # 1. Check classification cache (free)
        cached_cat = _cached_classification(conn, abs_path, mtime)
        if cached_cat is not None:
            cached_lower = cached_cat.lower()
            for i, cl in enumerate(categories_lower):
                if cl == cached_lower or cl in cached_lower or cached_lower in cl:
                    return {"path": abs_path, "category": categories[i], "cached": True}

        # 2. Check indexed caption from documents DB
        caption = _get_indexed_caption(conn, abs_path)

    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set", "path": abs_path}

    from google.genai import types

    cat_list = "\n".join(f"- {c}" for c in categories)
    model = os.environ.get("GEMINI_MODEL_LITE", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"))
    single_schema = {
        "type": "OBJECT",
        "properties": {
            "category": {"type": "STRING", "enum": list(categories)},
            "caption": {"type": "STRING"},
        },
        "required": ["category", "caption"],
    }

    # 2b. If caption exists → text-only Gemini call (much cheaper than vision)
    if caption:
        try:
            prompt = f'Photo description: "{caption}"\n\nClassify into EXACTLY ONE of:\n{cat_list}'
            response = _gemini_call_with_retry(
                client,
                model,
                contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=single_schema,
                ),
            )
            cat = json.loads(response.text).get("category")
            if cat:
                _save_classification(abs_path, mtime, cat)
                return {"path": abs_path, "category": cat, "cached": False, "source": "caption"}
        except Exception:
            pass  # Fall through to vision

    # 3. No cache, no caption (or caption didn't match) → Gemini vision
    try:
        from PIL import Image

        img = Image.open(abs_path).convert("RGB")
        img = _preprocess_image(img, 384)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_bytes = buf.getvalue()
        buf.close()
        img.close()

        prompt = _CLASSIFY_PROMPT_TEMPLATE.format(categories=cat_list)

        response = _gemini_call_with_retry(
            client,
            model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                response_mime_type="application/json",
                response_json_schema=single_schema,
            ),
        )

        result_data = json.loads(response.text)
        cat = result_data.get("category")
        caption_out = result_data.get("caption", "")
        _save_classification(abs_path, mtime, cat or "_uncategorized")
        if caption_out:
            _index_caption(abs_path, caption_out)

        return {"path": abs_path, "category": cat, "caption": caption_out, "cached": False}

    except Exception as e:
        return {"error": str(e), "path": abs_path}


def _fuzzy_match(text: str, categories: list[str], categories_lower: list[str]) -> str | None:
    """Fuzzy match Gemini response to a category. Returns category name or None."""
    text = text.strip().strip('"').strip("'").strip("-").strip()
    t = text.lower()
    for i, cl in enumerate(categories_lower):
        if cl == t or cl in t or t in cl:
            return categories[i]
    return None


def _save_classification(abs_path: str, mtime: float, category: str) -> None:
    """Save a single classification to DB cache."""
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO photo_classifications
            (path, mtime, category, classified_at)
            VALUES (?, ?, ?, datetime('now'))
        """,
            (abs_path, mtime, category),
        )
        conn.commit()


def _index_caption(abs_path: str, caption: str) -> None:
    """Index a photo caption into the documents table for FTS search."""
    try:
        from database import get_db, DB_PATH, upsert_document
        conn = get_db(DB_PATH)
        upsert_document(conn, abs_path, caption, file_type="image")
    except Exception as e:
        print(f"[Group] Index caption failed for {os.path.basename(abs_path)}: {e}")


_VISION_BATCH_SIZE = 5  # images per Gemini vision call
_CAPTION_BATCH_SIZE = 20  # captions per text-only call


def _classify_schema(categories: list[str]) -> dict[str, Any]:
    """Build structured output schema for batch classification — category enum enforced by Gemini.
    Also captures a short caption for indexing (free — Gemini already sees the image)."""
    return {
        "type": "OBJECT",
        "properties": {
            "classifications": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "filename": {"type": "STRING"},
                        "category": {"type": "STRING", "enum": list(categories)},
                        "caption": {"type": "STRING"},
                    },
                    "required": ["filename", "category", "caption"],
                },
            },
        },
        "required": ["classifications"],
    }


def _classify_batch_vision(
    items: list[tuple[str, float]],
    categories: list[str],
    categories_lower: list[str],
    cached_content: str | None = None,
) -> list[dict[str, Any]]:
    """Classify a batch of images via one Gemini vision call with structured output.
    items: [(abs_path, mtime), ...]
    cached_content: optional cache name (holds category prompt, saves input tokens)
    Returns list of {"path": ..., "category": ..., "cached": False} dicts.
    """
    if not items:
        return []

    client = _get_gemini()
    if not client:
        return [{"error": "GEMINI_API_KEY not set", "path": p} for p, _ in items]

    from google.genai import types
    from PIL import Image

    cat_list = "\n".join(f"- {c}" for c in categories)
    model = os.environ.get("GEMINI_MODEL_LITE", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"))

    # Build parts: all images + filenames
    parts = []
    valid_items = []
    for abs_path, mtime in items:
        try:
            img = Image.open(abs_path).convert("RGB")
            img = _preprocess_image(img, 384)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
            buf.close()
            img.close()
            valid_items.append((abs_path, mtime, os.path.basename(abs_path)))
        except Exception as e:
            print(f"[GroupBatch] Skip {abs_path}: {e}")

    if not valid_items:
        return []

    filenames = [v[2] for v in valid_items]
    # If cached_content holds the system prompt, only send filenames
    if cached_content:
        prompt = f"Photos in order: {', '.join(filenames)}\nFor each: classify + write a short caption (what's in the photo, 10-20 words)."
    else:
        prompt = (
            f"Classify each photo into EXACTLY ONE of these categories:\n{cat_list}\n\n"
            f"Photos in order: {', '.join(filenames)}\n"
            f"For each: classify + write a short caption describing what's in the photo (10-20 words)."
        )
    parts.append(types.Part.from_text(text=prompt))

    config = types.GenerateContentConfig(
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
        response_mime_type="application/json",
        response_json_schema=_classify_schema(categories),
    )
    if cached_content:
        config.cached_content = cached_content

    try:
        response = _gemini_call_with_retry(
            client,
            model,
            contents=[types.Content(parts=parts)],
            config=config,
        )
        data = json.loads(response.text)
        classifications = data.get("classifications", [])
        mapping = {c["filename"]: c["category"] for c in classifications}
        captions = {c["filename"]: c.get("caption", "") for c in classifications}
    except Exception as e:
        print(f"[GroupBatch] Vision batch failed ({len(valid_items)} images): {e}")
        # Fallback: classify individually
        results = []
        for abs_path, mtime, _ in valid_items:
            r = _classify_photo(abs_path, categories, categories_lower)
            results.append(r)
        return results

    # Parse results — category is already from enum, no fuzzy match needed
    results = []
    for abs_path, mtime, fname in valid_items:
        cat = mapping.get(fname)
        _save_classification(abs_path, mtime, cat or "_uncategorized")
        caption = captions.get(fname, "")
        if caption:
            _index_caption(abs_path, caption)
        results.append({"path": abs_path, "category": cat, "caption": caption, "cached": False})

    return results


def _classify_batch_captions(
    items: list[tuple[str, float, str]],
    categories: list[str],
    categories_lower: list[str],
    cached_content: str | None = None,
) -> list[dict[str, Any]]:
    """Classify a batch of captioned images via one text-only Gemini call with structured output.
    items: [(abs_path, mtime, caption), ...]
    cached_content: optional cache name (holds category prompt, saves input tokens)
    Returns list of {"path": ..., "category": ..., "cached": False, "source": "caption"} dicts.
    """
    if not items:
        return []

    client = _get_gemini()
    if not client:
        return [{"error": "GEMINI_API_KEY not set", "path": p} for p, _, _ in items]

    from google.genai import types

    cat_list = "\n".join(f"- {c}" for c in categories)
    model = os.environ.get("GEMINI_MODEL_LITE", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"))

    # Build prompt with all captions
    lines = []
    for abs_path, mtime, caption in items:
        fname = os.path.basename(abs_path)
        lines.append(f'- {fname}: "{caption[:200]}"')

    if cached_content:
        prompt = f"Photo descriptions:\n{''.join(chr(10) + l for l in lines)}\nClassify each. For caption, reuse the description given."
    else:
        prompt = (
            f"Photo descriptions:\n{''.join(chr(10) + l for l in lines)}\n\n"
            f"Classify each into EXACTLY ONE of:\n{cat_list}\n"
            f"For caption, reuse the description given."
        )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=_classify_schema(categories),
    )
    if cached_content:
        config.cached_content = cached_content

    try:
        response = _gemini_call_with_retry(
            client,
            model,
            contents=[types.Content(parts=[types.Part.from_text(text=prompt)])],
            config=config,
        )
        data = json.loads(response.text)
        mapping = {c["filename"]: c["category"] for c in data.get("classifications", [])}
    except Exception as e:
        print(f"[GroupBatch] Caption batch failed ({len(items)} captions): {e}")
        # Fallback: classify individually
        results = []
        for abs_path, mtime, _ in items:
            r = _classify_photo(abs_path, categories, categories_lower)
            results.append(r)
        return results

    # Parse results — category from enum, no fuzzy match needed
    results = []
    for abs_path, mtime, _ in items:
        fname = os.path.basename(abs_path)
        cat = mapping.get(fname)
        _save_classification(abs_path, mtime, cat or "_uncategorized")
        results.append({"path": abs_path, "category": cat, "cached": False, "source": "caption"})

    return results


_SUGGEST_PROMPT = """Look at these sample photos from a folder. Suggest 4-8 categories that would best organize this entire collection.
Be specific to what you see (e.g. "Ceremony" not "Events"). Keep names short (1-3 words)."""

_SUGGEST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "categories": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["categories"],
}


def suggest_categories(folder: str) -> dict[str, Any]:
    """Sample photos from a folder and suggest grouping categories via Gemini."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return {"error": f"Folder not found: {folder}"}

    # Collect images
    images = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(folder, f))
    if not images:
        return {"error": "No images found in folder"}

    # Check if photos are already indexed — use captions if available (free)
    captions = []
    with _db_lock:
        conn = _get_conn()
        for img_path in images[:100]:  # check up to 100
            cap = _get_indexed_caption(conn, img_path)
            if cap:
                captions.append(cap)

    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set"}

    try:
        from google.genai import types

        if len(captions) >= len(images) * 0.3:
            # 30%+ indexed — suggest from captions (text-only, much cheaper)
            sample_captions = captions[:30]
            caption_text = "\n".join(f"- {c[:150]}" for c in sample_captions)
            parts = [
                types.Part.from_text(
                    text=f"These are descriptions of photos in a folder:\n{caption_text}\n\n{_SUGGEST_PROMPT}"
                )
            ]
        else:
            # Not enough indexed — sample actual images (vision call)
            from PIL import Image

            sample_count = min(20, len(images))
            step = max(1, len(images) // sample_count)
            samples = [images[i] for i in range(0, len(images), step)][:sample_count]

            parts = []
            for s in samples:
                img = Image.open(s).convert("RGB")
                img = _preprocess_image(img, 384)  # Gemini LOW res = 384px
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=60)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
                buf.close()
                img.close()
            parts.append(types.Part.from_text(text=_SUGGEST_PROMPT))

        model = os.environ.get("GEMINI_MODEL_LITE", os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"))
        response = _gemini_call_with_retry(
            client,
            model,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                response_mime_type="application/json",
                response_json_schema=_SUGGEST_SCHEMA,
            ),
        )

        data = json.loads(response.text)
        categories = data.get("categories", [])
        if not isinstance(categories, list) or len(categories) < 2:
            return {"error": "Gemini returned invalid categories", "raw": response.text}

        sampled = len(sample_captions) if len(captions) >= len(images) * 0.3 else len(samples)
        return {
            "folder": folder,
            "total_images": len(images),
            "sampled": sampled,
            "categories": categories,
            "_hint": f"Suggested {len(categories)} categories for {len(images)} photos: {', '.join(categories)}. Confirm with user, then call group_photos.",
        }

    except Exception as e:
        return {"error": f"Category suggestion failed: {e}"}


def group_photos(folder: str, categories: list[str], uncategorized_folder: str | None = None) -> dict[str, Any]:
    """Start background photo grouping by Gemini vision classification. Returns immediately."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return {"error": f"Folder not found: {folder}"}

    if not categories or len(categories) < 2:
        return {"error": "Need at least 2 categories"}

    # Collect images
    images = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(folder, f))
    if not images:
        return {"error": "No images found in folder"}

    if not uncategorized_folder:
        uncategorized_folder = os.path.join(folder, "_uncategorized")

    # Init progress
    progress = {
        "status": "classifying",
        "folder": folder,
        "total": len(images),
        "classified": 0,
        "errors": 0,
        "categories": categories,
        "started_at": time.time(),
        "eta_seconds": None,
        "report_path": None,
    }
    with _group_lock:
        _group_jobs[folder] = progress

    def _run() -> None:
        classified = []
        categories_lower = [c.lower() for c in categories]
        start = time.time()
        gemini_cache_name = None  # caching removed — lite/preview models don't support it and it can hang

        # Phase 1: Check cache + gather captions for all images
        cached_items = []
        caption_items = []  # (abs_path, mtime, caption)
        vision_items = []  # (abs_path, mtime)

        print(f"[Group] Phase 1: scanning {len(images)} images for cache/captions...")
        with _db_lock:
            conn = _get_conn()
            for img_path in images:
                abs_path = os.path.abspath(img_path)
                try:
                    mtime = os.path.getmtime(abs_path)
                except OSError:
                    progress["errors"] += 1
                    progress["classified"] += 1
                    continue

                # Check classification cache
                cached_cat = _cached_classification(conn, abs_path, mtime)
                if cached_cat is not None:
                    matched = _fuzzy_match(cached_cat, categories, categories_lower)
                    if matched:
                        cached_items.append({"path": abs_path, "category": matched, "cached": True})
                        progress["classified"] += 1
                        continue

                # Check indexed caption
                caption = _get_indexed_caption(conn, abs_path)
                if caption:
                    caption_items.append((abs_path, mtime, caption))
                else:
                    vision_items.append((abs_path, mtime))

        print(f"[Group] Phase 1 done: {len(cached_items)} cached, {len(caption_items)} captions, {len(vision_items)} need vision")
        classified.extend(cached_items)
        if cached_items:
            print(
                f"[Group] {len(cached_items)} from cache, {len(caption_items)} with captions, {len(vision_items)} need vision"
            )

        def _cleanup_cache() -> None:
            pass  # no-op — caching removed

        # Phase 2: Batch-classify captioned images (text-only, cheap)
        for i in range(0, len(caption_items), _CAPTION_BATCH_SIZE):
            if progress.get("stop"):
                progress["status"] = "cancelled"
                progress["classified_count"] = len(classified)
                progress["elapsed_seconds"] = round(time.time() - start, 1)
                _cleanup_cache()
                return
            batch = caption_items[i : i + _CAPTION_BATCH_SIZE]
            results = _classify_batch_captions(batch, categories, categories_lower, cached_content=gemini_cache_name)
            for r in results:
                if "error" in r:
                    progress["errors"] += 1
                else:
                    classified.append(r)
                progress["classified"] += 1
            # ETA
            elapsed = time.time() - start
            remaining = progress["total"] - progress["classified"]
            rate = progress["classified"] / max(elapsed, 0.1)
            progress["eta_seconds"] = round(remaining / max(rate, 0.01))

        # Phase 3: Batch-classify remaining via vision (5 images per call, concurrent)
        print(f"[Group] Phase 3: {len(vision_items)} images to classify via vision...")
        vision_batches = [
            vision_items[i : i + _VISION_BATCH_SIZE] for i in range(0, len(vision_items), _VISION_BATCH_SIZE)
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_classify_batch_vision, batch, categories, categories_lower, gemini_cache_name): batch
                for batch in vision_batches
            }
            for future in as_completed(futures):
                if progress.get("stop"):
                    pool.shutdown(wait=False, cancel_futures=True)
                    progress["status"] = "cancelled"
                    progress["classified_count"] = len(classified)
                    progress["elapsed_seconds"] = round(time.time() - start, 1)
                    _cleanup_cache()
                    return
                results = future.result()
                for r in results:
                    if "error" in r:
                        progress["errors"] += 1
                    else:
                        classified.append(r)
                    progress["classified"] += 1
                # ETA
                elapsed = time.time() - start
                remaining = progress["total"] - progress["classified"]
                rate = progress["classified"] / max(elapsed, 0.1)
                progress["eta_seconds"] = round(remaining / max(rate, 0.01))

        if not classified:
            progress["status"] = "error"
            progress["error"] = "No photos could be classified"
            return

        # Move to category subfolders
        progress["status"] = "moving"
        group_counts = {}
        moved = 0
        for item in classified:
            cat = item["category"]
            if cat is None:
                dest_folder = uncategorized_folder
                cat_name = "_uncategorized"
            else:
                dest_folder = os.path.join(folder, cat)
                cat_name = cat

            os.makedirs(dest_folder, exist_ok=True)
            try:
                dest = os.path.join(dest_folder, os.path.basename(item["path"]))
                # Avoid overwriting files with duplicate basenames
                if os.path.exists(dest):
                    base, ext = os.path.splitext(os.path.basename(item["path"]))
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(dest_folder, f"{base}_{counter}{ext}")
                        counter += 1
                shutil.move(item["path"], dest)
                group_counts[cat_name] = group_counts.get(cat_name, 0) + 1
                moved += 1
            except Exception as e:
                print(f"[Group] Move failed: {item['path']} — {e}")
                progress["errors"] += 1

        # Generate HTML report
        progress["status"] = "report"
        report_path = _generate_group_report(folder, classified, categories, group_counts)

        # Final stats
        progress["status"] = "done"
        progress["moved"] = moved
        progress["group_counts"] = group_counts
        progress["report_path"] = report_path
        progress["elapsed_seconds"] = round(time.time() - start, 1)

        # (Gemini caching removed — category prompt sent inline with each call)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "started": True,
        "folder": folder,
        "total_images": len(images),
        "categories": categories,
        "_hint": f"Grouping {len(images)} photos into {len(categories)} categories. Use group_status to check progress.",
    }


def _generate_group_report(
    folder: str, classified: list[dict[str, Any]], categories: list[str], group_counts: dict[str, int]
) -> str:
    """Generate HTML report showing groups with thumbnails."""
    groups = {}
    for item in classified:
        cat = item.get("category") or "_uncategorized"
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(item)

    sections = []
    for cat in categories + ["_uncategorized"]:
        items = groups.get(cat, [])
        if not items:
            continue

        cards = []
        for item in items:
            moved_path = os.path.join(folder, cat, os.path.basename(item["path"]))
            display_path = moved_path if os.path.exists(moved_path) else item["path"]
            thumb = _make_thumbnail_b64(display_path)
            if not thumb:
                continue
            esc_name = html_mod.escape(os.path.basename(item["path"]))
            esc_url = html_mod.escape(display_path.replace(chr(92), "/"))
            cards.append(f"""<div class="card" onclick="window.open('file:///{esc_url}')">
  <img src="data:image/jpeg;base64,{thumb}" alt="{esc_name}">
  <div class="info"><b>{esc_name}</b></div>
</div>""")

        count = group_counts.get(cat, len(items))
        esc_cat = html_mod.escape(cat)
        sections.append(f"""<div class="group">
  <h2>{esc_cat} ({count})</h2>
  <div class="grid">{"".join(cards)}</div>
</div>""")

    counts_summary = ", ".join(f"<b>{html_mod.escape(cat)}</b> ({group_counts.get(cat, 0)})" for cat in group_counts)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Pinpoint Photo Groups — {html_mod.escape(os.path.basename(folder))}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#1a1a2e;color:#eee;padding:20px}}
h1{{font-size:1.4em;margin-bottom:4px}}
h2{{font-size:1.1em;margin:16px 0 8px;color:#7ec8e3}}
.stats{{color:#aaa;margin-bottom:20px;font-size:.9em}}
.stats b{{color:#fff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}}
.card{{background:#16213e;border-radius:8px;overflow:hidden;cursor:pointer;transition:transform .15s}}
.card:hover{{transform:scale(1.03)}}
.card img{{width:100%;aspect-ratio:1;object-fit:cover;display:block}}
.info{{padding:6px}}
.info b{{font-size:.75em;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.group{{margin-bottom:24px}}
</style></head><body>
<h1>Photo Groups — {html_mod.escape(os.path.basename(folder))}</h1>
<div class="stats">{len(classified)} photos → {len(groups)} groups: {counts_summary}</div>
{"".join(sections)}
</body></html>"""

    report_path = os.path.join(folder, "_group_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def get_group_status(folder: str, cancel: bool = False) -> dict[str, Any]:
    """Get grouping progress for a folder. Set cancel=True to stop the job."""
    folder = os.path.abspath(folder)
    with _group_lock:
        progress = _group_jobs.get(folder)
    if not progress:
        return {"error": "No group job found for this folder", "_hint": "Start one with group_photos first."}

    if cancel and progress["status"] not in ("done", "cancelled", "error"):
        progress["stop"] = True
        return {
            "status": "cancelling",
            "folder": folder,
            "_hint": "Cancellation requested. Check status again shortly.",
        }

    result = dict(progress)
    if result["status"] == "done":
        counts_str = ", ".join(f"{k}: {v}" for k, v in result.get("group_counts", {}).items())
        result["_hint"] = (
            f"Done! {result['moved']} photos grouped. {counts_str}. Report: {result.get('report_path', 'N/A')}"
        )
    elif result["status"] == "classifying":
        pct = round(result["classified"] / max(result["total"], 1) * 100)
        eta = result.get("eta_seconds")
        eta_str = f", ~{eta}s left" if eta else ""
        result["_hint"] = f"Classifying: {result['classified']}/{result['total']} ({pct}%){eta_str}"
    elif result["status"] == "moving":
        result["_hint"] = "Moving photos to category folders..."
    elif result["status"] == "report":
        result["_hint"] = "Generating HTML report..."
    elif result["status"] == "cancelled":
        result["_hint"] = f"Cancelled after classifying {result.get('classified', 0)}/{result.get('total', 0)} photos."

    return result
