"""
Pinpoint — Visual Image Search using SigLIP2 ONNX embeddings.
Text-to-image similarity: embed images, search by text description.
Embeddings stored in SQLite (same DB as documents). No duplicates.
In-memory cache for instant repeat queries.

Uses ONNX runtime (no torch dependency). Same model as InsightFace runtime.
"""

from __future__ import annotations

import os
import sqlite3
import struct
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

# Ensure CUDA 12 libs from pip nvidia packages are on LD_LIBRARY_PATH (onnxruntime needs them)
_site_pkgs = os.path.join(
    sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia"
)
for _subdir in ["cublas/lib", "cudnn/lib", "cuda_runtime/lib"]:
    _p = os.path.join(_site_pkgs, _subdir)
    if os.path.isdir(_p):
        os.environ["LD_LIBRARY_PATH"] = _p + ":" + os.environ.get("LD_LIBRARY_PATH", "")
from datetime import UTC, datetime

from database import DB_PATH, get_db

# SigLIP2 availability detection
_HAS_SIGLIP = False
try:
    import onnxruntime

    _HAS_SIGLIP = True
except ImportError:
    pass
print(f"[Pinpoint] SigLIP2: {'available' if _HAS_SIGLIP else 'not installed — Gemini vision fallback'}")

# --- Config ---
SIGLIP_MODEL = "onnx-community/siglip2-base-patch16-224-ONNX"
EMBED_DIM = 768
BATCH_SIZE = 16
MAX_LOAD_DIM = 384  # Pre-resize before SigLIP (saves I/O time)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic"}

# --- Lazy-loaded model + in-memory cache ---
_siglip = None  # (vision_session, text_session, processor)
_mem_cache = {}  # folder_abs → {paths: [...], emb_norm: ndarray}


def _get_siglip() -> tuple[Any, Any, Any]:
    """Lazy-load SigLIP2 ONNX sessions and processor."""
    global _siglip
    if _siglip is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor

        print(f"[SigLIP2] Loading ONNX model {SIGLIP_MODEL}...")
        t0 = time.time()

        # Pick fp16 for GPU, fp32 for CPU
        providers = ort.get_available_providers()
        has_cuda = "CUDAExecutionProvider" in providers
        suffix = "_fp16" if has_cuda else ""
        ep = ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]

        # Download ONNX files
        vision_path = hf_hub_download(SIGLIP_MODEL, f"onnx/vision_model{suffix}.onnx")
        text_path = hf_hub_download(SIGLIP_MODEL, f"onnx/text_model{suffix}.onnx")

        # Create sessions (fall back to CPU + fp32 if GPU fails)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        try:
            vision_session = ort.InferenceSession(vision_path, sess_opts, providers=ep)
            text_session = ort.InferenceSession(text_path, sess_opts, providers=ep)
        except Exception as gpu_err:
            if has_cuda:
                print(f"[SigLIP2] GPU session failed ({gpu_err}), falling back to CPU fp32...")
                has_cuda = False
                ep = ["CPUExecutionProvider"]
                vision_path = hf_hub_download(SIGLIP_MODEL, "onnx/vision_model.onnx")
                text_path = hf_hub_download(SIGLIP_MODEL, "onnx/text_model.onnx")
                vision_session = ort.InferenceSession(vision_path, sess_opts, providers=ep)
                text_session = ort.InferenceSession(text_path, sess_opts, providers=ep)
            else:
                raise

        # Processor handles image preprocessing + tokenization (no torch needed)
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, use_fast=False)

        device = "GPU" if has_cuda else "CPU"
        print(f"[SigLIP2] Loaded on {device} in {time.time() - t0:.1f}s")
        _siglip = (vision_session, text_session, processor)
    return _siglip


def _get_conn() -> sqlite3.Connection:
    """Get DB connection (same database as documents)."""
    return get_db(DB_PATH)


# --- Embedding serialization (768 floats → bytes) ---


def _embedding_to_bytes(emb: Any) -> bytes:
    """Convert 1D float array to compact bytes."""
    return struct.pack(f"{EMBED_DIM}f", *emb.tolist())


def _bytes_to_embedding(data: bytes) -> Any:
    """Convert bytes back to 1D float array."""
    return np.array(struct.unpack(f"{EMBED_DIM}f", data), dtype=np.float32)


# --- Fast image loading ---


def _load_image_fast(path: str) -> Any:
    """Load image with fast JPEG draft decoding + resize."""
    img = Image.open(path)
    img.draft("RGB", (MAX_LOAD_DIM, MAX_LOAD_DIM))
    img.load()
    img = img.convert("RGB")
    img.thumbnail((MAX_LOAD_DIM, MAX_LOAD_DIM), Image.LANCZOS)
    return img


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
        vision_session, _, processor = _get_siglip()
        print(f"[SigLIP2] Embedding {len(to_embed)} images ({len(cached)} cached in DB)...")
        t0 = time.time()
        new_embeddings = []  # For DB save

        for i in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[i : i + BATCH_SIZE]
            batch_imgs = []
            batch_valid = []
            for fpath in batch:
                try:
                    img = _load_image_fast(fpath)
                    batch_imgs.append(img)
                    batch_valid.append(fpath)
                except Exception as e:
                    print(f"[SigLIP2] Skip {os.path.basename(fpath)}: {e}")

            if not batch_imgs:
                continue

            inputs = processor(images=batch_imgs, return_tensors="np", padding=True)
            pixel_values = inputs["pixel_values"].astype(np.float32)

            # Run vision model
            input_name = vision_session.get_inputs()[0].name
            outputs = vision_session.run(None, {input_name: pixel_values})
            embs = outputs[1]  # pooler_output [batch, embed_dim]
            if len(embs) != len(batch_valid):
                print(f"[SigLIP2] Shape mismatch: got {len(embs)} embeddings for {len(batch_valid)} images, skipping batch")
                continue

            for j, fpath in enumerate(batch_valid):
                emb = embs[j].astype(np.float32)
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
            f"[SigLIP2] Embedded {len(to_embed)} images in {elapsed:.1f}s ({elapsed / len(to_embed) * 1000:.0f}ms/img)"
        )

    return result


def embed_text(query: str) -> Any:
    """Embed a text query. Returns normalized embedding array."""
    _, text_session, processor = _get_siglip()
    inputs = processor(text=[query], return_tensors="np", padding="max_length")
    input_ids = inputs["input_ids"].astype(np.int64)

    # Build feed dict from all processor outputs (input_ids, attention_mask, etc.)
    feed = {}
    session_input_names = {inp.name for inp in text_session.get_inputs()}
    for key, val in inputs.items():
        if key in session_input_names:
            feed[key] = np.array(val, dtype=np.int64)

    outputs = text_session.run(None, feed)
    emb = outputs[1].astype(np.float32)  # pooler_output [1, embed_dim]
    return _normalize(emb).squeeze(0)


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
    import io
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
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text=f"[{os.path.basename(fpath)}]"))
                names.append(fpath)
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
    """
    if not _HAS_SIGLIP:
        return _search_images_gemini(folder, query, limit, recursive=recursive)
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
        image_embeddings = embed_images(folder)
        embed_time = time.time() - t0

        if not image_embeddings:
            return {"error": "No images could be embedded", "results": []}

        # Build matrix and cache in memory
        paths = list(image_embeddings.keys())
        emb_matrix = np.stack([image_embeddings[p] for p in paths])
        emb_norm = _normalize(emb_matrix, axis=-1)
        _mem_cache[folder] = {"paths": paths, "emb_norm": emb_norm}

    # Embed query and search
    t1 = time.time()
    query_emb = embed_text(query)
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
