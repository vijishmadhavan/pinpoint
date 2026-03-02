"""
Pinpoint — On-demand face analysis & search (InsightFace)

Full InsightFace capabilities exposed:
  - Face detection (bbox, confidence)
  - Face recognition (512-dim embeddings, cosine comparison)
  - Age estimation
  - Gender detection
  - Head pose (pitch, yaw, roll)
  - Face landmarks (5-point, 68-point 3D, 106-point 2D)
  - Face cropping with padding

Loads model lazily (only when first face operation is requested).
Caches face data in SQLite for instant repeat queries.
"""

import hashlib
import json
import os
import sqlite3
import struct
import tempfile

import numpy as np
from PIL import Image

# Max dimension for preprocessing (InsightFace works fine at 640px)
MAX_FACE_DIM = 1280


def _preprocess_for_face(img_path: str):
    """Load and resize image for face detection. Originals untouched."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_FACE_DIM:
        scale = MAX_FACE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)[:, :, ::-1]  # RGB→BGR for InsightFace


# --- Lazy model loading ---

_app = None


def _get_model():
    """Load InsightFace buffalo_l once, on first use."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        print("[FaceSearch] Loading InsightFace model (first time)...")
        _app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
        print("[FaceSearch] Model loaded.")
    return _app


# --- Embedding serialization ---

def _embedding_to_bytes(emb: np.ndarray) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


def _bytes_to_embedding(data: bytes) -> np.ndarray:
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


# --- File hash (for cache invalidation) ---

def _file_hash(path: str) -> str:
    stat = os.stat(path)
    key = f"{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


# --- Extract full face data from InsightFace result ---

def _extract_face_data(face, idx: int) -> dict:
    """Extract all available data from an InsightFace face object."""
    bbox = face.bbox.astype(int).tolist()
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    data = {
        "face_idx": idx,
        "bbox": str(bbox),
        "bbox_width": w,
        "bbox_height": h,
        "confidence": round(float(face.det_score), 3),
        "age": int(face.age) if hasattr(face, "age") and face.age is not None else None,
        "gender": "M" if getattr(face, "gender", None) == 1 else "F" if getattr(face, "gender", None) == 0 else None,
        "embedding": face.embedding,
    }

    # Head pose: [pitch, yaw, roll] in degrees
    if hasattr(face, "pose") and face.pose is not None:
        pose = face.pose.tolist()
        data["pose"] = {
            "pitch": round(pose[0], 1),
            "yaw": round(pose[1], 1),
            "roll": round(pose[2], 1),
        }

    return data


# --- Cache operations ---

def _get_cached_faces(conn: sqlite3.Connection, image_path: str):
    """Get cached face data if file hasn't changed. Returns list or None."""
    current_hash = _file_hash(image_path)
    rows = conn.execute(
        "SELECT face_idx, bbox, embedding, confidence, age, gender, pose "
        "FROM face_cache WHERE image_path = ? AND file_hash = ? ORDER BY face_idx",
        (image_path, current_hash)
    ).fetchall()

    if not rows:
        return None

    faces = []
    for row in rows:
        faces.append({
            "face_idx": row["face_idx"],
            "bbox": row["bbox"],
            "embedding": _bytes_to_embedding(row["embedding"]),
            "confidence": row["confidence"],
            "age": row["age"],
            "gender": row["gender"],
            "pose": json.loads(row["pose"]) if row["pose"] else None,
        })
    return faces


def _cache_faces(conn: sqlite3.Connection, image_path: str, faces: list):
    """Store face data in cache."""
    fh = _file_hash(image_path)
    conn.execute("DELETE FROM face_cache WHERE image_path = ?", (image_path,))

    for face in faces:
        pose_json = json.dumps(face.get("pose")) if face.get("pose") else None
        conn.execute(
            "INSERT INTO face_cache(image_path, file_hash, face_idx, bbox, "
            "embedding, confidence, age, gender, pose) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_path, fh, face["face_idx"], face["bbox"],
             _embedding_to_bytes(face["embedding"]), face["confidence"],
             face.get("age"), face.get("gender"), pose_json)
        )
    conn.commit()


def _face_to_api(face: dict) -> dict:
    """Convert face dict to API-safe dict (no raw embeddings)."""
    result = {
        "face_idx": face["face_idx"],
        "bbox": face["bbox"],
        "confidence": face["confidence"],
    }
    if face.get("age") is not None:
        result["age"] = face["age"]
    if face.get("gender") is not None:
        result["gender"] = face["gender"]
    if face.get("pose") is not None:
        result["pose"] = face["pose"]
    if face.get("bbox_width") is not None:
        result["bbox_width"] = face["bbox_width"]
        result["bbox_height"] = face["bbox_height"]
    return result


# --- Core functions ---

def _recognize_against_known(faces_with_embeddings: list, conn: sqlite3.Connection, threshold: float = 0.5) -> dict:
    """Match face embeddings against known_faces table. Returns {face_idx: name}."""
    known = conn.execute("SELECT id, name, embedding FROM known_faces").fetchall()
    if not known:
        return {}

    # Group known embeddings by name
    known_by_name = {}  # name → [embedding, ...]
    for row in known:
        name = row["name"]
        emb = _bytes_to_embedding(row["embedding"])
        known_by_name.setdefault(name, []).append(emb)

    matches = {}
    for face in faces_with_embeddings:
        best_name = None
        best_sim = 0.0
        for name, embeddings in known_by_name.items():
            # Max similarity across a person's multiple embeddings
            max_sim = max(_cosine_sim(face["embedding"], emb) for emb in embeddings)
            if max_sim > best_sim:
                best_sim = max_sim
                best_name = name
        if best_sim >= threshold:
            matches[face["face_idx"]] = {"name": best_name, "similarity": round(best_sim, 3)}

    return matches


def detect_faces(image_path: str, conn: sqlite3.Connection = None):
    """
    Detect all faces in an image. Returns full analysis for each face:
    face_idx, bbox, confidence, age, gender, head pose.
    Auto-recognizes known faces if any are saved.
    """
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    # Check cache first
    cached_faces = None
    if conn:
        cached = _get_cached_faces(conn, image_path)
        if cached is not None:
            cached_faces = cached

    if cached_faces is None:
        # Run InsightFace (preprocessed — large images resized, originals untouched)
        app = _get_model()
        img = _preprocess_for_face(image_path)
        raw_faces = app.get(img)
        cached_faces = [_extract_face_data(f, i) for i, f in enumerate(raw_faces)]
        # Cache results
        if conn and cached_faces:
            _cache_faces(conn, image_path, cached_faces)

    api_faces = [_face_to_api(f) for f in cached_faces]

    # Auto-recognize against known faces
    if conn and cached_faces:
        try:
            has_known = conn.execute("SELECT 1 FROM known_faces LIMIT 1").fetchone()
            if has_known:
                matches = _recognize_against_known(cached_faces, conn)
                for face in api_faces:
                    match = matches.get(face["face_idx"])
                    if match:
                        face["name"] = match["name"]
                        face["recognition_similarity"] = match["similarity"]
        except Exception:
            pass  # known_faces table may not exist on old DBs

    return api_faces


def crop_face(image_path: str, face_idx: int, conn: sqlite3.Connection = None):
    """
    Crop a specific face from an image with padding.
    Saves to temp file. Returns path + face metadata.
    """
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    # Get face bbox
    bbox = None
    face_meta = None

    if conn:
        cached = _get_cached_faces(conn, image_path)
        if cached:
            for f in cached:
                if f["face_idx"] == face_idx:
                    bbox = eval(f["bbox"])
                    face_meta = _face_to_api(f)
                    break

    if bbox is None:
        app = _get_model()
        img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1]
        raw_faces = app.get(img)
        if face_idx >= len(raw_faces):
            return {"error": f"Face {face_idx} not found (only {len(raw_faces)} faces)"}
        bbox = raw_faces[face_idx].bbox.astype(int).tolist()
        face_data = _extract_face_data(raw_faces[face_idx], face_idx)
        face_meta = _face_to_api(face_data)

    # Crop with padding
    pil_img = Image.open(image_path).convert("RGB")
    w, h = pil_img.size
    x1, y1, x2, y2 = bbox
    pad = int((x2 - x1) * 0.25)  # 25% padding
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = pil_img.crop((x1, y1, x2, y2))

    # Save to temp
    temp_dir = os.path.join(tempfile.gettempdir(), "pinpoint_faces")
    os.makedirs(temp_dir, exist_ok=True)
    crop_path = os.path.join(temp_dir, f"face_{face_idx}_{os.path.basename(image_path)}")
    crop.save(crop_path, "JPEG", quality=90)

    result = {"path": crop_path, "bbox": bbox}
    if face_meta:
        result.update(face_meta)
    return result


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _scan_folder_for_faces(app, folder: str, conn: sqlite3.Connection = None):
    """Scan a folder and return all face data (cached or fresh)."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".heic"}
    all_faces = {}  # path → [face_dicts]
    scanned = 0
    errors = 0

    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        img_path = os.path.join(folder, name)
        scanned += 1

        try:
            candidate_faces = None
            if conn:
                cached = _get_cached_faces(conn, img_path)
                if cached:
                    candidate_faces = cached

            if candidate_faces is None:
                img = _preprocess_for_face(img_path)
                raw_faces = app.get(img)
                candidate_faces = [_extract_face_data(f, i) for i, f in enumerate(raw_faces)]
                if conn and candidate_faces:
                    _cache_faces(conn, img_path, candidate_faces)

            if candidate_faces:
                all_faces[img_path] = candidate_faces

        except Exception as e:
            errors += 1
            print(f"[FaceSearch] Error scanning {img_path}: {e}")

    return all_faces, scanned, errors


def find_person(reference_image: str, folder: str, conn: sqlite3.Connection = None, threshold: float = 0.4):
    """
    Find photos matching a reference face in a folder.
    Reference image must have exactly 1 face.
    If multiple faces, returns info so caller can use find_person_by_face.
    """
    reference_image = os.path.abspath(reference_image)
    folder = os.path.abspath(folder)

    if not os.path.exists(reference_image):
        return {"error": f"Reference image not found: {reference_image}"}
    if not os.path.isdir(folder):
        return {"error": f"Folder not found: {folder}"}

    app = _get_model()
    ref_img = np.array(Image.open(reference_image).convert("RGB"))[:, :, ::-1]
    ref_faces = app.get(ref_img)

    if not ref_faces:
        return {"error": "No face detected in reference image. Please send a clearer photo."}

    if len(ref_faces) > 1:
        faces_info = [_face_to_api(_extract_face_data(f, i)) for i, f in enumerate(ref_faces)]
        return {
            "multiple_faces": True,
            "face_count": len(ref_faces),
            "faces": faces_info,
            "reference_image": reference_image,
            "message": f"Found {len(ref_faces)} faces. Use crop_face to show each to the user, "
                       "then call find_person_by_face with the chosen face_idx.",
        }

    target_embedding = ref_faces[0].embedding
    target_info = _face_to_api(_extract_face_data(ref_faces[0], 0))

    # Scan folder
    all_faces, scanned, errors = _scan_folder_for_faces(app, folder, conn)

    matches = []
    for img_path, faces in all_faces.items():
        for face in faces:
            sim = _cosine_sim(target_embedding, face["embedding"])
            if sim >= threshold:
                match = {
                    "path": img_path,
                    "similarity": round(sim, 3),
                    "face_bbox": face["bbox"],
                }
                if face.get("age") is not None:
                    match["age"] = face["age"]
                if face.get("gender") is not None:
                    match["gender"] = face["gender"]
                matches.append(match)
                break  # One match per image

    matches.sort(key=lambda m: m["similarity"], reverse=True)

    return {
        "reference_image": reference_image,
        "reference_face": target_info,
        "folder": folder,
        "matches": matches,
        "match_count": len(matches),
        "scanned": scanned,
        "errors": errors,
    }


def find_person_by_face(reference_image: str, face_idx: int, folder: str,
                         conn: sqlite3.Connection = None, threshold: float = 0.4):
    """
    Find photos matching a specific face (by index) from a multi-face reference.
    """
    reference_image = os.path.abspath(reference_image)
    folder = os.path.abspath(folder)

    app = _get_model()
    ref_img = np.array(Image.open(reference_image).convert("RGB"))[:, :, ::-1]
    ref_faces = app.get(ref_img)

    if face_idx >= len(ref_faces):
        return {"error": f"Face {face_idx} not found (only {len(ref_faces)} faces)"}

    target_embedding = ref_faces[face_idx].embedding
    target_info = _face_to_api(_extract_face_data(ref_faces[face_idx], face_idx))

    all_faces, scanned, errors = _scan_folder_for_faces(app, folder, conn)

    matches = []
    for img_path, faces in all_faces.items():
        for face in faces:
            sim = _cosine_sim(target_embedding, face["embedding"])
            if sim >= threshold:
                match = {
                    "path": img_path,
                    "similarity": round(sim, 3),
                    "face_bbox": face["bbox"],
                }
                if face.get("age") is not None:
                    match["age"] = face["age"]
                if face.get("gender") is not None:
                    match["gender"] = face["gender"]
                matches.append(match)
                break

    matches.sort(key=lambda m: m["similarity"], reverse=True)

    return {
        "reference_image": reference_image,
        "reference_face": target_info,
        "face_idx": face_idx,
        "folder": folder,
        "matches": matches,
        "match_count": len(matches),
        "scanned": scanned,
        "errors": errors,
    }


def count_faces(image_path: str, conn: sqlite3.Connection = None):
    """Quick count of faces in an image with age/gender summary."""
    faces = detect_faces(image_path, conn)
    if isinstance(faces, dict) and "error" in faces:
        return faces

    summary = {
        "image_path": os.path.abspath(image_path),
        "total_faces": len(faces),
        "faces": faces,
    }

    # Gender counts
    males = sum(1 for f in faces if f.get("gender") == "M")
    females = sum(1 for f in faces if f.get("gender") == "F")
    if males or females:
        summary["males"] = males
        summary["females"] = females

    # Age range
    ages = [f["age"] for f in faces if f.get("age") is not None]
    if ages:
        summary["age_range"] = {"min": min(ages), "max": max(ages), "avg": round(sum(ages) / len(ages))}

    return summary


def compare_faces(image_path_1: str, face_idx_1: int,
                  image_path_2: str, face_idx_2: int,
                  conn: sqlite3.Connection = None):
    """
    Compare two specific faces from two images.
    Returns similarity score and whether they're the same person.
    """
    image_path_1 = os.path.abspath(image_path_1)
    image_path_2 = os.path.abspath(image_path_2)

    # Get embeddings
    def _get_embedding(img_path, fidx):
        if conn:
            cached = _get_cached_faces(conn, img_path)
            if cached:
                for f in cached:
                    if f["face_idx"] == fidx:
                        return f["embedding"], _face_to_api(f)
        app = _get_model()
        img = np.array(Image.open(img_path).convert("RGB"))[:, :, ::-1]
        raw_faces = app.get(img)
        if fidx >= len(raw_faces):
            return None, {"error": f"Face {fidx} not found"}
        face_data = _extract_face_data(raw_faces[fidx], fidx)
        if conn:
            _cache_faces(conn, img_path, [_extract_face_data(f, i) for i, f in enumerate(raw_faces)])
        return face_data["embedding"], _face_to_api(face_data)

    emb1, meta1 = _get_embedding(image_path_1, face_idx_1)
    if emb1 is None:
        return meta1

    emb2, meta2 = _get_embedding(image_path_2, face_idx_2)
    if emb2 is None:
        return meta2

    sim = _cosine_sim(emb1, emb2)

    return {
        "similarity": round(sim, 3),
        "same_person": sim >= 0.4,
        "confidence": "high" if sim >= 0.6 else "medium" if sim >= 0.4 else "low",
        "face_1": meta1,
        "face_2": meta2,
    }


# --- Persistent face memory (Segment 18V) ---

def remember_face(image_path: str, face_idx: int, name: str, conn: sqlite3.Connection):
    """
    Save a face embedding for future recognition.
    One person can have multiple embeddings (different angles improve accuracy).
    """
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
    if not name or not name.strip():
        return {"error": "Name is required."}

    name = name.strip()

    # Get embedding from cache or detect fresh
    cached = _get_cached_faces(conn, image_path)
    if cached:
        face = next((f for f in cached if f["face_idx"] == face_idx), None)
        if face:
            embedding = face["embedding"]
        else:
            return {"error": f"Face {face_idx} not found in cache (only {len(cached)} faces). Run detect_faces first."}
    else:
        app = _get_model()
        img = _preprocess_for_face(image_path)
        raw_faces = app.get(img)
        if face_idx >= len(raw_faces):
            return {"error": f"Face {face_idx} not found (only {len(raw_faces)} faces detected)."}
        face_data = _extract_face_data(raw_faces[face_idx], face_idx)
        embedding = face_data["embedding"]
        # Cache all faces while we're at it
        all_faces = [_extract_face_data(f, i) for i, f in enumerate(raw_faces)]
        _cache_faces(conn, image_path, all_faces)

    # Insert into known_faces
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO known_faces(name, embedding, source_image, created_at) VALUES (?, ?, ?, ?)",
        (name, _embedding_to_bytes(embedding), image_path, now)
    )
    conn.commit()

    # Count total embeddings for this person
    count = conn.execute("SELECT COUNT(*) as n FROM known_faces WHERE name = ?", (name,)).fetchone()["n"]

    return {
        "name": name,
        "face_idx": face_idx,
        "source_image": image_path,
        "id": cursor.lastrowid,
        "total_embeddings": count,
    }


def forget_face(name: str, conn: sqlite3.Connection):
    """Delete all saved face data for a person."""
    if not name or not name.strip():
        return {"error": "Name is required."}
    name = name.strip()

    cursor = conn.execute("DELETE FROM known_faces WHERE LOWER(name) = LOWER(?)", (name,))
    conn.commit()

    return {"name": name, "deleted_count": cursor.rowcount}


def recognize_faces(image_path: str, conn: sqlite3.Connection, threshold: float = 0.5):
    """
    Detect faces and match against all known faces.
    Standalone recognition — for when detect_faces isn't being called.
    """
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    # Get faces with embeddings
    cached = _get_cached_faces(conn, image_path)
    if cached is None:
        app = _get_model()
        img = _preprocess_for_face(image_path)
        raw_faces = app.get(img)
        cached = [_extract_face_data(f, i) for i, f in enumerate(raw_faces)]
        if cached:
            _cache_faces(conn, image_path, cached)

    if not cached:
        return {"image_path": image_path, "faces": [], "face_count": 0}

    # Match against known faces
    matches = _recognize_against_known(cached, conn, threshold)

    faces = []
    for face in cached:
        result = _face_to_api(face)
        match = matches.get(face["face_idx"])
        if match:
            result["name"] = match["name"]
            result["recognition_similarity"] = match["similarity"]
        else:
            result["name"] = None
        faces.append(result)

    return {"image_path": image_path, "faces": faces, "face_count": len(faces)}


def list_known_faces(conn: sqlite3.Connection):
    """List all known faces (unique names with embedding counts)."""
    rows = conn.execute(
        "SELECT name, COUNT(*) as count, MIN(created_at) as first_added "
        "FROM known_faces GROUP BY name ORDER BY name"
    ).fetchall()
    return [{"name": r["name"], "embeddings": r["count"], "first_added": r["first_added"]} for r in rows]
