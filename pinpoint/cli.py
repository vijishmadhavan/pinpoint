from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import dotenv_values

from pinpoint import __version__, user_data_dir

DEFAULT_ENV = {
    "API_SECRET": "",
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-3.1-flash-lite-preview",
    "GEMINI_MODEL_LITE": "gemini-3.1-flash-lite-preview",
    "LANGSEARCH_API_KEY": "",
    "JINA_API_KEY": "",
    "OLLAMA_MODEL": "",
    "OLLAMA_URL": "http://localhost:11434",
    "OLLAMA_THINK": "false",
    "OCR_DPI": "200",
    "TZ": "UTC",
}


def _env_path() -> Path:
    return user_data_dir() / ".env"


def _logs_dir() -> Path:
    return user_data_dir() / "logs"


def _load_env() -> dict[str, str]:
    env = DEFAULT_ENV.copy()
    path = _env_path()
    if path.exists():
        loaded = dotenv_values(path)
        env.update({k: str(v) for k, v in loaded.items() if v is not None})
    return env


def _write_env(values: dict[str, str]) -> Path:
    path = _env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={values.get(key, '')}" for key in DEFAULT_ENV]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prompt(current: str, label: str, *, required: bool = False) -> str:
    suffix = f" [{current}]" if current else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if current:
            return current
        if not required:
            return ""
        print(f"{label} is required.")


def cmd_setup(_args: argparse.Namespace) -> int:
    current = _load_env()
    detected_tz = os.environ.get("TZ", current.get("TZ") or "UTC")
    current["TZ"] = current.get("TZ") or detected_tz

    print("Pinpoint setup")
    print(f"Config will be stored in {_env_path()}")
    print("")

    current["GEMINI_API_KEY"] = _prompt(current.get("GEMINI_API_KEY", ""), "Gemini API key")
    current["TZ"] = _prompt(current.get("TZ", ""), "Timezone", required=True)
    current["API_SECRET"] = _prompt(current.get("API_SECRET", ""), "API secret (optional)")
    current["OLLAMA_MODEL"] = _prompt(current.get("OLLAMA_MODEL", ""), "Ollama model (optional)")
    current["OLLAMA_URL"] = _prompt(current.get("OLLAMA_URL", ""), "Ollama URL")
    current["JINA_API_KEY"] = _prompt(current.get("JINA_API_KEY", ""), "Jina API key (optional)")
    current["LANGSEARCH_API_KEY"] = _prompt(current.get("LANGSEARCH_API_KEY", ""), "LangSearch API key (optional)")
    current["OCR_DPI"] = _prompt(current.get("OCR_DPI", ""), "OCR DPI")

    path = _write_env(current)
    print("")
    print(f"Saved config to {path}")
    print("Run `pinpoint api` to start the backend or `pinpoint search \"query\"` to search locally.")
    return 0


def _api_ping() -> bool:
    try:
        with urlopen("http://127.0.0.1:5123/ping", timeout=2) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def cmd_status(_args: argparse.Namespace) -> int:
    from database import DB_PATH, init_db

    api_running = _api_ping()
    conn = init_db(DB_PATH)
    try:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        job_count = conn.execute(
            "SELECT COUNT(*) FROM background_jobs WHERE status IN ('pending', 'running', 'cancelling')"
        ).fetchone()[0]
        watch_count = conn.execute("SELECT COUNT(*) FROM watched_folders").fetchone()[0]
    finally:
        conn.close()

    print(f"Pinpoint {__version__}")
    print(f"API: {'running' if api_running else 'stopped'}")
    print(f"DB: {DB_PATH}")
    print(f"Documents: {doc_count}")
    print(f"Watched folders: {watch_count}")
    print(f"Active jobs: {job_count}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from database import DB_PATH
    from search import search

    result = search(args.query, DB_PATH, limit=args.limit)
    items = result.get("results", [])
    if result.get("ambiguous_search"):
        print(f"Ambiguous: {result.get('clarification_hint', '').strip()}")
    if not items:
        print("No results.")
        return 0
    for row in items:
        print(f"[{row.get('id')}] {row.get('filename') or row.get('path')}")
        snippet = (row.get("snippet") or "").strip()
        if snippet:
            print(f"  {snippet[:200]}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from database import DB_PATH
    from indexer import index_folder

    def _progress(done: int, total: int, path: str):
        print(f"[{done}/{total}] {path}")

    result = index_folder(args.path, DB_PATH, progress_callback=_progress)
    print(result)
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    env = _load_env()
    os.environ.update(env)

    import uvicorn

    from api import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    logs_dir = _logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    paths = [logs_dir / "api.log", logs_dir / "bot.log", Path.cwd() / "pinpoint.log"]
    for path in paths:
        if not path.exists():
            print(f"{path} (missing)")
            continue
        print(f"== {path} ==")
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[-args.lines :]:
            print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pinpoint", description="Pinpoint CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Create or update ~/.pinpoint/.env")
    p_setup.set_defaults(func=cmd_setup)

    p_status = sub.add_parser("status", help="Show API/db/job status")
    p_status.set_defaults(func=cmd_status)

    p_search = sub.add_parser("search", help="Run local search against the Pinpoint database")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_index = sub.add_parser("index", help="Index a folder into the Pinpoint database")
    p_index.add_argument("path")
    p_index.set_defaults(func=cmd_index)

    p_api = sub.add_parser("api", help="Start the FastAPI backend")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", type=int, default=5123)
    p_api.set_defaults(func=cmd_api)

    p_logs = sub.add_parser("logs", help="Show recent log lines")
    p_logs.add_argument("--lines", type=int, default=40)
    p_logs.set_defaults(func=cmd_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
