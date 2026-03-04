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
from datetime import datetime, timezone

from database import DB_PATH, get_db
from image_search import _HAS_SIGLIP

# --- Config ---
EMBED_DIM = 768
BATCH_SIZE = 16
MAX_LOAD_DIM = 384
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
VIDEO_MIME = {
    ".mp4": "video/mp4", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    ".mov": "video/quicktime", ".wmv": "video/x-ms-wmv", ".flv": "video/x-flv",
    ".webm": "video/webm",
}
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


from image_search import _normalize


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
        capture_output=True, text=True, timeout=60
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

    # Extract frames at specified fps
    pattern = os.path.join(temp_dir, "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2",
         "-y", pattern],
        capture_output=True, check=True, timeout=300
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
    now = datetime.now(timezone.utc).isoformat()
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


def _timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS timestamp back to seconds."""
    parts = ts.split(":")
    if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
    return 0.0


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


def _search_video_gemini(video_path: str, query: str, fps: float = DEFAULT_FPS, limit: int = 5) -> dict:
    """Gemini native video analysis — upload full video, no frame extraction needed.
    Note: fps is accepted for API compatibility but unused (Gemini analyzes the full video)."""
    from extractors import _get_gemini
    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set", "results": []}
    from google.genai import types
    import json as _json

    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        return {"error": f"Video not found: {video_path}", "results": []}

    ext = os.path.splitext(video_path)[1].lower()
    mime = VIDEO_MIME.get(ext, "video/mp4")
    file_size = os.path.getsize(video_path)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    t0 = time.time()

    # Build video part: inline <100MB, File API for larger
    if file_size < 100 * 1024 * 1024:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_part = types.Part.from_bytes(data=video_bytes, mime_type=mime)
    else:
        print(f"[GeminiVideoSearch] Uploading {file_size / 1024 / 1024:.0f}MB via File API...")
        uploaded = client.files.upload(file=video_path, config={"mime_type": mime})
        waited = 0
        while uploaded.state.name == "PROCESSING":
            if waited >= 300:
                return {"error": "File upload timed out after 5 minutes", "results": []}
            time.sleep(2)
            waited += 2
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            return {"error": f"File upload failed: {uploaded.state.name}", "results": []}
        video_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime)

    prompt = (
        f"Find the top {limit} moments in this video that best match: '{query}'\n"
        f"Return ONLY valid JSON array: [{{\"timestamp\": \"MM:SS\", \"match_pct\": 0-100, \"description\": \"brief\"}}]\n"
        f"Use HH:MM:SS for videos over 1 hour. Sort by relevance (highest first)."
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=[video_part, types.Part.from_text(prompt)])],
            config={"media_resolution": "MEDIA_RESOLUTION_LOW"},
        )
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        results_raw = _json.loads(text)

        results = []
        for item in results_raw[:limit]:
            ts = item.get("timestamp", "0:00")
            secs = _timestamp_to_seconds(ts)
            results.append({
                "timestamp": ts,
                "seconds": round(secs, 1),
                "match_pct": round(float(item.get("match_pct", 0)), 1),
            })
    except Exception as e:
        print(f"[GeminiVideoSearch] Error: {e}")
        return {"error": f"Gemini video analysis failed: {e}", "results": []}

    elapsed = time.time() - t0
    print(f"[GeminiVideoSearch] Analyzed video in {elapsed:.1f}s, found {len(results)} matches")
    return {
        "video": video_path,
        "query": query,
        "results": results,
        "note": f"Top {len(results)} matching moments (Gemini full video analysis)",
        "search_time_s": round(elapsed, 2),
        "cached": False,
        "_hint": f"{len(results)} matches found — use extract_frame to get specific frames for sending.",
    }


def search_video(video_path: str, query: str, fps: float = DEFAULT_FPS, limit: int = 5) -> dict:
    """
    Search a video by text description.
    Returns matching frames with timestamps and similarity scores.
    """
    # Gemini native video analysis (primary), SigLIP fallback
    from extractors import _get_gemini
    if _get_gemini():
        return _search_video_gemini(video_path, query, fps, limit)
    if not _HAS_SIGLIP:
        return {"error": "GEMINI_API_KEY not set and SigLIP unavailable", "results": []}
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
        capture_output=True, check=True, timeout=60
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
            print(f"Search: {result.get('search_time_s', 0)}s")
            print(f"\nResults for '{query}':")
            for r in result["results"]:
                print(f"  {r['timestamp']} ({r['seconds']}s) — {r['match_pct']}% match")
    else:
        print("Usage: python video_search.py <video_path> <query> [fps]")
