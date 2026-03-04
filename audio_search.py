"""
Pinpoint — Audio Search using Gemini native audio understanding.
Transcribe audio files and search within audio by text query.
Gemini supports up to 9.5 hours of audio at 32 tokens/sec.
"""

import os
import time
import json

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".aiff"}
AUDIO_MIME = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".wma": "audio/x-ms-wma",
    ".m4a": "audio/mp4", ".aiff": "audio/aiff",
}


def _get_audio_part(client, audio_path: str):
    """Build a Gemini Part from an audio file. Inline <100MB, File API for larger."""
    from google.genai import types

    ext = os.path.splitext(audio_path)[1].lower()
    mime = AUDIO_MIME.get(ext, "audio/mpeg")
    file_size = os.path.getsize(audio_path)

    if file_size < 100 * 1024 * 1024:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return types.Part.from_bytes(data=audio_bytes, mime_type=mime)
    else:
        print(f"[AudioSearch] Uploading {file_size / 1024 / 1024:.0f}MB via File API...")
        uploaded = client.files.upload(file=audio_path, config={"mime_type": mime})
        waited = 0
        while uploaded.state.name == "PROCESSING":
            if waited >= 300:
                raise RuntimeError("File upload timed out after 5 minutes")
            time.sleep(2)
            waited += 2
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"File upload failed: {uploaded.state.name}")
        return types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime)


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe an audio file to text using Gemini."""
    from extractors import _get_gemini
    from google.genai import types

    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set"}

    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in AUDIO_EXTS:
        return {"error": f"Unsupported audio format: {ext}. Supported: {', '.join(sorted(AUDIO_EXTS))}"}

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    t0 = time.time()

    try:
        audio_part = _get_audio_part(client, audio_path)
        prompt = (
            "Transcribe this audio completely. Include timestamps in [MM:SS] format "
            "at natural breaks (every 15-30 seconds or at speaker changes). "
            "Format: [MM:SS] transcribed text"
        )
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=[audio_part, types.Part.from_text(prompt)])],
        )
        text = (resp.text or "").strip()
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    elapsed = time.time() - t0
    print(f"[AudioSearch] Transcribed {os.path.basename(audio_path)} in {elapsed:.1f}s")
    return {
        "path": audio_path,
        "text": text,
        "method": "gemini",
        "transcribe_time_s": round(elapsed, 2),
        "_hint": "Transcription complete — answer the user's question from the text above.",
    }


def search_audio(audio_path: str, query: str, limit: int = 5) -> dict:
    """Search within an audio file for specific content using Gemini."""
    from extractors import _get_gemini
    from google.genai import types

    client = _get_gemini()
    if not client:
        return {"error": "GEMINI_API_KEY not set"}

    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in AUDIO_EXTS:
        return {"error": f"Unsupported audio format: {ext}. Supported: {', '.join(sorted(AUDIO_EXTS))}"}

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    t0 = time.time()

    try:
        audio_part = _get_audio_part(client, audio_path)
        prompt = (
            f"Find the top {limit} moments in this audio that match: '{query}'\n"
            f"Return ONLY valid JSON array: [{{\"timestamp\": \"MM:SS\", \"text\": \"what is said/heard\", \"match_pct\": 0-100}}]\n"
            f"Use HH:MM:SS for audio over 1 hour. Sort by relevance (highest first)."
        )
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(parts=[audio_part, types.Part.from_text(prompt)])],
        )
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        results = json.loads(text)
        results = results[:limit]
    except Exception as e:
        return {"error": f"Audio search failed: {e}", "results": []}

    elapsed = time.time() - t0
    print(f"[AudioSearch] Searched {os.path.basename(audio_path)} in {elapsed:.1f}s, found {len(results)} matches")
    return {
        "audio": audio_path,
        "query": query,
        "results": results,
        "search_time_s": round(elapsed, 2),
        "_hint": f"{len(results)} matches found — answer the user's question from results above.",
    }
