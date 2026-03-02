"""
Pinpoint — Video Search using SigLIP2 ONNX embeddings + FFmpeg frame extraction.
Search inside videos by text description: extract frames, embed with SigLIP2,
find frames matching a text query. Embeddings cached in SQLite.
"""

import os
import time
import struct
import subprocess
import tempfile
import sqlite3
import numpy as np
from PIL import Image
from datetime import datetime

from database import DB_PATH, get_db

# --- Config ---
EMBED_DIM = 768
BATCH_SIZE = 16
MAX_LOAD_DIM = 384
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
DEFAULT_FPS = 1  # 1 frame per second


def _get_conn() -> sqlite3.Connection:
    return get_db(DB_PATH)


# --- Reuse SigLIP2 from image_search ---

def _get_siglip():
    """Reuse SigLIP2 model from image_search (shared lazy singleton)."""
    from image_search import _get_siglip as _img_get_siglip
    return _img_get_siglip()


def _embed_text(query: str) -> np.ndarray:
    """Reuse text embedding from image_search."""
    from image_search import embed_text
    return embed_text(query)


def _normalize(v: np.ndarray, axis=-1) -> np.ndarray:
    """L2-normalize along axis."""
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return v / norm


# --- Embedding serialization ---

def _embedding_to_bytes(emb: np.ndarray) -> bytes:
    return struct.pack(f"{EMBED_DIM}f", *emb.tolist())


def _bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"{EMBED_DIM}f", data), dtype=np.float32)


# --- Frame extraction with ffmpeg ---

def extract_frames(video_path: str, fps: float = DEFAULT_FPS, temp_dir: str = None) -> list:
    """
    Extract frames from video using ffmpeg.
    Returns list of (frame_path, timestamp_sec) tuples.
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="pinpoint_frames_")

    # Get video duration first
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

    # Extract frames at specified fps
    pattern = os.path.join(temp_dir, "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2",
         "-y", pattern],
        capture_output=True, check=True
    )

    # Collect extracted frames with timestamps
    frames = []
    for fname in sorted(os.listdir(temp_dir)):
        if not fname.startswith("frame_") or not fname.endswith(".jpg"):
            continue
        # frame_000001.jpg → index 0, timestamp = index / fps
        idx = int(fname.split("_")[1].split(".")[0]) - 1  # 1-based to 0-based
        timestamp = idx / fps
        frames.append((os.path.join(temp_dir, fname), timestamp))

    return frames, duration


# --- DB cache operations ---

def _load_cached_embeddings(video_path: str) -> dict:
    """Load cached frame embeddings for a video. Returns {frame_sec: ndarray} if valid."""
    conn = _get_conn()
    try:
        current_mtime = os.path.getmtime(video_path)
    except OSError:
        return {}

    rows = conn.execute(
        "SELECT frame_sec, embedding, mtime FROM video_embeddings WHERE video_path = ?",
        (video_path,)
    ).fetchall()

    if not rows:
        return {}

    # Check if video changed (compare mtime of first row)
    if abs(rows[0]["mtime"] - current_mtime) >= 1.0:
        # Video changed, invalidate cache
        conn.execute("DELETE FROM video_embeddings WHERE video_path = ?", (video_path,))
        conn.commit()
        return {}

    cached = {}
    for row in rows:
        cached[row["frame_sec"]] = _bytes_to_embedding(row["embedding"])
    return cached


def _save_embeddings(video_path: str, frame_embeddings: list):
    """Save frame embeddings. frame_embeddings: [(frame_sec, ndarray), ...]"""
    if not frame_embeddings:
        return
    conn = _get_conn()
    mtime = os.path.getmtime(video_path)
    now = datetime.utcnow().isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO video_embeddings (video_path, frame_sec, embedding, mtime, embedded_at) VALUES (?, ?, ?, ?, ?)",
        [(video_path, sec, _embedding_to_bytes(emb), mtime, now) for sec, emb in frame_embeddings]
    )
    conn.commit()


# --- Core functions ---

def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def embed_video(video_path: str, fps: float = DEFAULT_FPS, progress_callback=None) -> dict:
    """
    Embed all frames of a video. Returns {frame_sec: embedding_ndarray}.
    Uses DB cache if video unchanged.
    """
    video_path = os.path.abspath(video_path)

    # Check cache
    cached = _load_cached_embeddings(video_path)
    if cached:
        print(f"[VideoSearch] Using cached embeddings ({len(cached)} frames)")
        return cached

    # Extract frames
    print(f"[VideoSearch] Extracting frames at {fps} fps...")
    t0 = time.time()
    temp_dir = tempfile.mkdtemp(prefix="pinpoint_vframes_")
    try:
        frames, duration = extract_frames(video_path, fps, temp_dir)
        print(f"[VideoSearch] Extracted {len(frames)} frames in {time.time() - t0:.1f}s (duration: {_format_timestamp(duration)})")

        if not frames:
            return {}

        # Embed frames with SigLIP2 ONNX
        vision_session, _, processor = _get_siglip()
        print(f"[VideoSearch] Embedding {len(frames)} frames...")
        t1 = time.time()

        result = {}
        new_embeddings = []

        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i + BATCH_SIZE]
            batch_imgs = []
            batch_secs = []

            for frame_path, sec in batch:
                try:
                    img = Image.open(frame_path).convert("RGB")
                    img.thumbnail((MAX_LOAD_DIM, MAX_LOAD_DIM), Image.LANCZOS)
                    batch_imgs.append(img)
                    batch_secs.append(sec)
                except Exception as e:
                    print(f"[VideoSearch] Skip frame at {sec}s: {e}")

            if not batch_imgs:
                continue

            inputs = processor(images=batch_imgs, return_tensors="np", padding=True)
            pixel_values = inputs["pixel_values"].astype(np.float32)

            input_name = vision_session.get_inputs()[0].name
            outputs = vision_session.run(None, {input_name: pixel_values})
            embs = outputs[1]  # pooler_output [batch, embed_dim]

            for j, sec in enumerate(batch_secs):
                emb = embs[j].astype(np.float32)
                result[sec] = emb
                new_embeddings.append((sec, emb))

            if progress_callback:
                progress_callback(min(i + BATCH_SIZE, len(frames)), len(frames))

        # Save to DB
        _save_embeddings(video_path, new_embeddings)

        elapsed = time.time() - t1
        print(f"[VideoSearch] Embedded {len(frames)} frames in {elapsed:.1f}s ({elapsed / max(len(frames), 1) * 1000:.0f}ms/frame)")

        return result

    finally:
        # Cleanup temp frames
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def search_video(video_path: str, query: str, fps: float = DEFAULT_FPS, limit: int = 5) -> dict:
    """
    Search a video by text description.
    Returns matching frames with timestamps and similarity scores.
    """
    video_path = os.path.abspath(video_path)

    if not os.path.isfile(video_path):
        return {"error": f"Video not found: {video_path}", "results": []}

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in VIDEO_EXTS:
        return {"error": f"Unsupported video format: {ext}", "results": []}

    # Embed video frames
    t0 = time.time()
    frame_embeddings = embed_video(video_path, fps)
    embed_time = time.time() - t0

    if not frame_embeddings:
        return {"error": "No frames could be extracted/embedded", "results": []}

    # Build matrix
    secs = sorted(frame_embeddings.keys())
    emb_matrix = np.stack([frame_embeddings[s] for s in secs])
    emb_norm = _normalize(emb_matrix, axis=-1)

    # Embed query and search
    t1 = time.time()
    query_emb = _embed_text(query)
    similarities = emb_norm @ query_emb  # [N]
    search_time = time.time() - t1

    # Get top-K
    k = min(limit, len(secs))
    top_indices = np.argsort(similarities)[::-1][:k]

    # Normalize scores
    raw_scores = similarities[top_indices]
    max_score = raw_scores[0] if len(raw_scores) > 0 else 1.0
    min_all = similarities.min()
    score_range = max_score - min_all if max_score > min_all else 1.0

    results = []
    for idx in top_indices:
        sec = secs[idx]
        raw = similarities[idx]
        pct = round(float((raw - min_all) / score_range * 100), 1)
        results.append({
            "timestamp": _format_timestamp(sec),
            "seconds": round(sec, 1),
            "match_pct": pct,
        })

    return {
        "video": video_path,
        "total_frames": len(secs),
        "fps_used": fps,
        "query": query,
        "results": results,
        "note": f"Top {len(results)} matching moments in video for '{query}'.",
        "embed_time_s": round(embed_time, 2),
        "search_time_s": round(search_time, 3),
        "cached": embed_time < 1.0,
    }


def extract_frame_image(video_path: str, seconds: float, output_path: str = None) -> str:
    """Extract a single frame from video at given timestamp. Returns output path."""
    video_path = os.path.abspath(video_path)
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        ts = _format_timestamp(seconds).replace(":", "-")
        output_path = os.path.join(tempfile.gettempdir(), f"{base}_frame_{ts}.jpg")

    subprocess.run(
        ["ffmpeg", "-ss", str(seconds), "-i", video_path,
         "-frames:v", "1", "-q:v", "2", "-y", output_path],
        capture_output=True, check=True
    )
    return output_path


# --- Quick test ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        vpath = sys.argv[1]
        query = sys.argv[2]
        fps = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        result = search_video(vpath, query, fps=fps)
        if "error" in result and not result.get("results"):
            print(f"Error: {result['error']}")
        else:
            print(f"Video: {result['video']}")
            print(f"Frames: {result['total_frames']} ({result['fps_used']} fps)")
            print(f"Cached: {result['cached']}")
            print(f"Embed: {result['embed_time_s']}s, Search: {result['search_time_s']}s")
            print(f"\nResults for '{query}':")
            for r in result["results"]:
                print(f"  {r['timestamp']} ({r['seconds']}s) — {r['match_pct']}% match")
    else:
        print("Usage: python video_search.py <video_path> <query> [fps]")
