"""
Pinpoint — Visual Image Search using SigLIP2 ONNX embeddings.
Text-to-image similarity: embed images, search by text description.
Embeddings stored in SQLite (same DB as documents). No duplicates.
In-memory cache for instant repeat queries.

Uses ONNX runtime (no torch dependency). Same model as InsightFace runtime.
"""

import os
import time
import struct
import sqlite3
import numpy as np
from PIL import Image
from datetime import datetime

from database import DB_PATH, get_db

# --- Config ---
SIGLIP_MODEL = "onnx-community/siglip2-base-patch16-224-ONNX"
EMBED_DIM = 768
BATCH_SIZE = 16
MAX_LOAD_DIM = 384  # Pre-resize before SigLIP (saves I/O time)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic"}

# --- Lazy-loaded model + in-memory cache ---
_siglip = None  # (vision_session, text_session, processor)
_mem_cache = {}  # folder_abs → {paths: [...], emb_norm: ndarray}


def _get_siglip():
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

        # Create sessions
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        vision_session = ort.InferenceSession(vision_path, sess_opts, providers=ep)
        text_session = ort.InferenceSession(text_path, sess_opts, providers=ep)

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

def _embedding_to_bytes(emb: np.ndarray) -> bytes:
    """Convert 1D float array to compact bytes."""
    return struct.pack(f"{EMBED_DIM}f", *emb.tolist())


def _bytes_to_embedding(data: bytes) -> np.ndarray:
    """Convert bytes back to 1D float array."""
    return np.array(struct.unpack(f"{EMBED_DIM}f", data), dtype=np.float32)


# --- Fast image loading ---

def _load_image_fast(path: str) -> Image.Image:
    """Load image with fast JPEG draft decoding + resize."""
    img = Image.open(path)
    img.draft("RGB", (MAX_LOAD_DIM, MAX_LOAD_DIM))
    img.load()
    img = img.convert("RGB")
    img.thumbnail((MAX_LOAD_DIM, MAX_LOAD_DIM), Image.LANCZOS)
    return img


def _get_image_files(folder: str) -> list[str]:
    """Get sorted list of image files in a folder."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ])


def _normalize(v: np.ndarray, axis=-1) -> np.ndarray:
    """L2-normalize along axis."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return v / norm


# --- DB operations ---

def _load_cached_embeddings(paths: list[str]) -> dict:
    """Load cached embeddings from DB. Returns {path: ndarray} for valid (unchanged) files."""
    if not paths:
        return {}
    conn = _get_conn()
    cached = {}
    # Query in chunks to avoid SQLite variable limit
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT path, embedding, mtime FROM image_embeddings WHERE path IN ({placeholders})",
            chunk
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


def _save_embeddings(path_embeddings: list):
    """Save embeddings to DB. path_embeddings: [(path, ndarray, mtime), ...]"""
    if not path_embeddings:
        return
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO image_embeddings (path, embedding, mtime, embedded_at) VALUES (?, ?, ?, ?)",
        [(p, _embedding_to_bytes(e), m, now) for p, e, m in path_embeddings]
    )
    conn.commit()


# --- Core functions ---

def embed_images(folder: str, progress_callback=None) -> dict:
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
            batch = to_embed[i:i + BATCH_SIZE]
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
        print(f"[SigLIP2] Embedded {len(to_embed)} images in {elapsed:.1f}s ({elapsed / len(to_embed) * 1000:.0f}ms/img)")

    return result


def embed_text(query: str) -> np.ndarray:
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


def search_images(folder: str, query: str, limit: int = 10) -> dict:
    """
    Search images in a folder by text description.
    Returns dict with results, timing, and cache info.
    """
    folder = os.path.abspath(folder)
    files = _get_image_files(folder)
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
        results.append({
            "filename": os.path.basename(fpath),
            "path": fpath,
            "match_pct": pct,
        })

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
