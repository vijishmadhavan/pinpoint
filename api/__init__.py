"""
Pinpoint — FastAPI server (bridge between Node.js and Python)

Router modules:
  core.py       — /ping, /status, /index, /indexing/status, /index-file
  search.py     — /search, /search-facts, /document/{id}, /web-read
  files.py      — /list_files, /file_info, /read_file, /move_file, /batch_move, etc.
  faces.py      — /detect-faces, /crop-face, /find-person, /count-faces, etc.
  media.py      — /search-images-visual, /search-video, /search-audio, /ocr, etc.
  data.py       — /analyze-data, /read_excel, /calculate, /extract-tables
  transform.py  — /write-file, /generate-excel, /resize-image, /merge-pdf, etc.
  memory.py     — /conversation/*, /memory/*, /setting, /reminders
  photos.py     — /score-photo, /cull-photos, /group-photos, etc.
  google.py     — /google/gmail-*, /google/calendar-*, /google/drive-* (via gws CLI)
"""

import hmac
import os

from dotenv import load_dotenv

from pinpoint import __version__

load_dotenv()

from fastapi import FastAPI

app = FastAPI(title="Pinpoint", version=__version__)

# --- API auth middleware ---
API_SECRET = os.environ.get("API_SECRET", "")


@app.middleware("http")
async def check_api_secret(request, call_next):
    """Require API_SECRET header on all endpoints except /ping."""
    if API_SECRET and request.url.path != "/ping":
        token = request.headers.get("X-API-Secret", "")
        if not hmac.compare_digest(token, API_SECRET):
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


# --- Register all routers ---
from api.core import router as core_router
from api.data import router as data_router
from api.faces import router as faces_router
from api.files import router as files_router
from api.google import router as google_router
from api.media import router as media_router
from api.memory import router as memory_router
from api.photos import router as photos_router
from api.search import router as search_router
from api.transform import router as transform_router

app.include_router(core_router)
app.include_router(search_router)
app.include_router(files_router)
app.include_router(faces_router)
app.include_router(media_router)
app.include_router(data_router)
app.include_router(transform_router)
app.include_router(memory_router)
app.include_router(photos_router)
app.include_router(google_router)


@app.on_event("startup")
def _on_startup():
    # Auto-scan common folders for path registry (background, non-blocking)
    from api.files import scan_paths_background
    scan_paths_background()
