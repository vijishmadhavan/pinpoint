"""Pinpoint API — thin launcher. All routes live in api/ package."""

if __name__ == "__main__":
    import uvicorn

    from api import app  # noqa: F401 — registers all routers

    print("[Pinpoint API] Starting on http://localhost:5123")
    print("[Pinpoint API] Docs at http://localhost:5123/docs")
    uvicorn.run(app, host="0.0.0.0", port=5123, log_level="info")
