"""
Pinpoint — Visual Image Search using Gemini Embedding 2.
Text-to-image similarity: embed images via API, search by text description.
Embeddings stored in SQLite (same DB as documents). No duplicates.
In-memory cache for instant repeat queries.

Uses Gemini Embedding 2 API (multimodal — text + images in same space).
Matryoshka truncation to 768 dims for compact storage.
"""

from __future__ import annotations

import io
import os
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 178_956_970  # ~256MP — prevent decompression bombs

from database import DB_PATH, get_db

# --- Config ---
EMBED_DIM = 768  # Matryoshka truncation (full: 3072, we use 768 for compact storage)
EMBED_MODEL = "gemini-embedding-2-preview"
BATCH_SIZE = 6  # Gemini Embedding 2 limit: 6 images per request
MAX_LOAD_DIM = 384  # Pre-resize before embedding (saves upload bytes)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic"}

# --- Lazy-loaded client + in-memory cache ---
_gemini_client = None
_mem_cache = {}  # folder_abs → {paths: [...], emb_norm: ndarray}
_MEM_CACHE_MAX = 10  # Evict oldest when cache exceeds this many folders

_db_conn = None
_db_lock = threading.RLock()


def _get_client() -> Any:
    """Lazy-load Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set — cannot use Gemini Embedding 2")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_conn() -> sqlite3.Connection:
    """Get or create a shared DB connection (thread-safe via double-checked locking)."""
    global _db_conn
    if _db_conn is None:
        with _db_lock:
            if _db_conn is None:
                _db_conn = get_db(DB_PATH)
    return _db_conn


# --- Embedding serialization (768 floats → bytes) ---


def _embedding_to_bytes(emb: Any) -> bytes:
    """Convert 1D float array to compact bytes."""
    return struct.pack(f"{EMBED_DIM}f", *emb.tolist())


def _bytes_to_embedding(data: bytes) -> np.ndarray:
    """Convert bytes back to 1D float array."""
    return np.array(struct.unpack(f"{EMBED_DIM}f", data), dtype=np.float32)


# --- Fast image loading ---


def _load_image_fast(path: str) -> Image.Image:
    """Load image with fast JPEG draft decoding + resize. Caller must close returned image."""
    raw = Image.open(path)
    raw.draft("RGB", (MAX_LOAD_DIM, MAX_LOAD_DIM))
    raw.load()
    img = raw.convert("RGB")
    if img is not raw:
        raw.close()
    img.thumbnail((MAX_LOAD_DIM, MAX_LOAD_DIM), Image.LANCZOS)
    return img


def _image_to_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes for API upload."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    data = buf.getvalue()
    buf.close()
    return data


def _get_image_files(folder: str, recursive: bool = False) -> list[str]:
    """Get sorted list of image files in a folder (optionally recursive)."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return []
    if recursive:
        files = []
        for root, _dirs, fnames in os.walk(folder):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    files.append(os.path.join(root, f))
        return sorted(files)
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS])


def _normalize(v: Any, axis: int = -1) -> Any:
    """L2-normalize along axis."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return v / norm


# --- DB operations ---


def _load_cached_embeddings(paths: list[str]) -> dict[str, Any]:
    """Load cached embeddings from DB. Returns {path: ndarray} for valid (unchanged) files."""
    if not paths:
        return {}
    conn = _get_conn()
    cached = {}
    # Query in chunks to avoid SQLite variable limit
    for i in range(0, len(paths), 500):
        chunk = paths[i : i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT path, embedding, mtime FROM image_embeddings WHERE path IN ({placeholders})", chunk
        ).fetchall()
        for row in rows:
            path, emb_bytes, stored_mtime = row
            try:
                current_mtime = os.path.getmtime(path)
                if abs(current_mtime - stored_mtime) < 1.0:
                    cached[path] = _bytes_to_embedding(emb_bytes)
            except OSError:
                pass
    return cached


def _save_embeddings(path_embeddings: list[tuple[str, Any, float]]) -> None:
    """Save embeddings to DB. path_embeddings: [(path, ndarray, mtime), ...]"""
    if not path_embeddings:
        return
    conn = _get_conn()
    now = datetime.now(UTC).isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO image_embeddings (path, embedding, mtime, embedded_at) VALUES (?, ?, ?, ?)",
        [(p, _embedding_to_bytes(e), m, now) for p, e, m in path_embeddings],
    )
    conn.commit()


# --- Gemini Embedding 2 API calls ---


def _embed_images_batch(image_bytes_list: list[bytes]) -> list[Any]:
    """Embed a batch of images (max 6) via Gemini Embedding 2. Returns list of numpy arrays."""
    from google.genai import types

    client = _get_client()
    contents = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in image_bytes_list]

    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIM,
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            return [np.array(e.values, dtype=np.float32) for e in result.embeddings]
        except Exception as e:
            err = str(e)
            if ("429" in err or "503" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"[GeminiEmbed] Rate limited, retry in {wait}s...")
                time.sleep(wait)
                continue
            raise


def _embed_text_query(query: str) -> Any:
    """Embed a text query via Gemini Embedding 2. Returns normalized numpy array."""
    from google.genai import types

    client = _get_client()

    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=query,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIM,
                    task_type="RETRIEVAL_QUERY",
                ),
            )
            emb = np.array(result.embeddings[0].values, dtype=np.float32)
            return _normalize(emb)
        except Exception as e:
            err = str(e)
            if ("429" in err or "503" in err or "RESOURCE_EXHAUSTED" in err) and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"[GeminiEmbed] Rate limited, retry in {wait}s...")
                time.sleep(wait)
                continue
            raise


# --- Core functions ---


def embed_images(folder: str, progress_callback: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """
    Embed all images in a folder. Returns {path: embedding_ndarray}.
    Uses DB cache for unchanged files, only embeds new/changed ones.
    """
    files = _get_image_files(folder)
    if not files:
        return {}

    # Load from DB
    cached = _load_cached_embeddings(files)
    to_embed = [f for f in files if f not in cached]

    result = dict(cached)

    if to_embed:
        print(f"[GeminiEmbed] Embedding {len(to_embed)} images ({len(cached)} cached in DB)...")
        t0 = time.time()
        new_embeddings = []  # For DB save

        for i in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[i : i + BATCH_SIZE]
            batch_bytes = []
            batch_valid = []
            for fpath in batch:
                try:
                    img = _load_image_fast(fpath)
                    batch_bytes.append(_image_to_bytes(img))
                    batch_valid.append(fpath)
                    img.close()
                except Exception as e:
                    print(f"[GeminiEmbed] Skip {os.path.basename(fpath)}: {e}")

            if not batch_bytes:
                continue

            try:
                embs = _embed_images_batch(batch_bytes)
                if len(embs) != len(batch_valid):
                    print(f"[GeminiEmbed] Shape mismatch: got {len(embs)} for {len(batch_valid)} images, skipping")
                    continue
            except Exception as e:
                print(f"[GeminiEmbed] Batch error: {e}")
                continue

            for j, fpath in enumerate(batch_valid):
                emb = embs[j]
                result[fpath] = emb
                try:
                    mtime = os.path.getmtime(fpath)
                    new_embeddings.append((fpath, emb, mtime))
                except OSError:
                    pass

            if progress_callback:
                progress_callback(min(i + BATCH_SIZE, len(to_embed)), len(to_embed))

        # Save new embeddings to DB
        _save_embeddings(new_embeddings)

        elapsed = time.time() - t0
        print(
            f"[GeminiEmbed] Embedded {len(to_embed)} images in {elapsed:.1f}s ({elapsed / max(len(to_embed), 1) * 1000:.0f}ms/img)"
        )

    return result


def embed_text(query: str) -> Any:
    """Embed a text query. Returns normalized embedding array."""
    return _embed_text_query(query)


def _search_images_gemini(
    folder: str,
    query: str,
    limit: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
    recursive: bool = False,
) -> dict[str, Any]:
    """Gemini vision fallback — multi-image concurrent ranking with LOW resolution.
    Handles 10K+ images: 200/batch, 5 concurrent workers, 280 tokens/image (LOW res)."""
    from extractors import _get_gemini

    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set", "results": []}
    import json as _json
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from google.genai import types

    folder = os.path.abspath(folder)
    files = _get_image_files(folder, recursive=recursive)
    if not files:
        return {"error": f"No images in {folder}", "results": []}

    t0 = time.time()
    BATCH = 200  # images per Gemini call (200 × 280 tokens = 56K at LOW res)
    WORKERS = 5  # concurrent API calls
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    all_scores = {}
    scores_lock = threading.Lock()
    done_count = [0]

    def _score_batch(batch_files: list[str]) -> dict[str, float]:
        parts = []
        names = []
        for fpath in batch_files:
            try:
                img = _load_image_fast(fpath)
                parts.append(types.Part.from_bytes(data=_image_to_bytes(img), mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text=f"[{os.path.basename(fpath)}]"))
                names.append(fpath)
                img.close()
            except Exception:
                continue
        if not names:
            return {}
        parts.append(types.Part.from_text(text=f"Query: '{query}'\nRate each image 0-100 for relevance to the query."))
        _score_schema = {
            "type": "OBJECT",
            "properties": {
                "scores": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "filename": {"type": "STRING"},
                            "score": {"type": "NUMBER"},
                        },
                        "required": ["filename", "score"],
                    },
                },
            },
            "required": ["scores"],
        }
        try:
            from extractors import gemini_call_with_retry

            resp = gemini_call_with_retry(
                client,
                model=model,
                contents=[types.Content(parts=parts)],
                config=types.GenerateContentConfig(
                    media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    response_mime_type="application/json",
                    response_json_schema=_score_schema,
                ),
            )
            data = _json.loads(resp.text)
            batch_scores = {}
            for item in data.get("scores", []):
                fname = item.get("filename", "")
                score = float(item.get("score", 0))
                for fp in names:
                    if os.path.basename(fp) == fname:
                        batch_scores[fp] = score
                        break
            return batch_scores
        except Exception as e:
            print(f"[GeminiSearch] Batch error: {e}")
            return {}

    # Build batch list
    batches = [files[i : i + BATCH] for i in range(0, len(files), BATCH)]
    total_batches = len(batches)
    print(f"[GeminiSearch] Scoring {len(files)} images in {total_batches} batches ({WORKERS} concurrent)...")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_score_batch, b): idx for idx, b in enumerate(batches)}
        for future in as_completed(futures):
            batch_scores = future.result()
            with scores_lock:
                all_scores.update(batch_scores)
                done_count[0] += 1
            if progress_callback:
                progress_callback(done_count[0], total_batches)

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    elapsed = time.time() - t0
    print(f"[GeminiSearch] Scored {len(all_scores)}/{len(files)} images in {elapsed:.1f}s")
    return {
        "folder": folder,
        "total_images": len(files),
        "query": query,
        "results": [{"filename": os.path.basename(p), "path": p, "match_pct": round(s, 1)} for p, s in ranked],
        "embed_time_s": round(elapsed, 2),
        "search_time_s": 0.0,
        "cached": False,
        "note": f"Top {len(ranked)} of {len(files)} (Gemini vision, {total_batches} batches)",
    }


def search_images(folder: str, query: str, limit: int = 10, recursive: bool = False) -> dict[str, Any]:
    """
    Search images in a folder by text description.
    Returns dict with results, timing, and cache info.
    Uses Gemini Embedding 2 for fast vector search, falls back to Gemini vision scoring.
    """
    # Check if Gemini API key is available
    if not os.environ.get("GEMINI_API_KEY"):
        return {"error": "GEMINI_API_KEY not set", "results": []}

    folder = os.path.abspath(folder)
    files = _get_image_files(folder, recursive=recursive)
    if not files:
        return {"error": f"No images found in {folder}", "results": []}

    # Check in-memory cache first (instant — no DB hit)
    mem = _mem_cache.get(folder)
    if mem and len(mem["paths"]) == len(files):
        paths = mem["paths"]
        emb_norm = mem["emb_norm"]
        embed_time = 0.0
    else:
        # Embed images (uses DB cache for unchanged files)
        t0 = time.time()
        try:
            image_embeddings = embed_images(folder)
        except Exception as e:
            print(f"[GeminiEmbed] Embedding failed ({e}), falling back to Gemini vision...")
            return _search_images_gemini(folder, query, limit, recursive=recursive)
        embed_time = time.time() - t0

        if not image_embeddings:
            return {"error": "No images could be embedded", "results": []}

        # Build matrix and cache in memory
        paths = list(image_embeddings.keys())
        emb_matrix = np.stack([image_embeddings[p] for p in paths])
        emb_norm = _normalize(emb_matrix, axis=-1)
        # Evict oldest entries if cache is full
        if len(_mem_cache) >= _MEM_CACHE_MAX and folder not in _mem_cache:
            oldest = next(iter(_mem_cache))
            del _mem_cache[oldest]
        _mem_cache[folder] = {"paths": paths, "emb_norm": emb_norm}

    # Embed query and search
    t1 = time.time()
    try:
        query_emb = embed_text(query)
    except Exception as e:
        print(f"[GeminiEmbed] Text embed failed ({e}), falling back to Gemini vision...")
        return _search_images_gemini(folder, query, limit, recursive=recursive)
    similarities = emb_norm @ query_emb  # [N]
    search_time = time.time() - t1

    # Get top-K
    k = min(limit, len(paths))
    top_indices = np.argsort(similarities)[::-1][:k]

    # Normalize scores to 0-100% (relative to this result set)
    raw_scores = similarities[top_indices]
    max_score = raw_scores[0] if len(raw_scores) > 0 else 1.0
    min_all = similarities.min()
    score_range = max_score - min_all if max_score > min_all else 1.0

    results = []
    for idx in top_indices:
        fpath = paths[idx]
        raw = similarities[idx]
        pct = round(float((raw - min_all) / score_range * 100), 1)
        results.append(
            {
                "filename": os.path.basename(fpath),
                "path": fpath,
                "match_pct": pct,
            }
        )

    return {
        "folder": folder,
        "total_images": len(files),
        "query": query,
        "results": results,
        "note": f"Top {len(results)} matches out of {len(files)} images, ranked by visual similarity. These are the best matches for '{query}'.",
        "embed_time_s": round(embed_time, 2),
        "search_time_s": round(search_time, 3),
        "cached": embed_time < 1.0,
    }
