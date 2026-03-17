from __future__ import annotations

import argparse
import importlib.resources
import os
import shutil
import subprocess
import sys
import time
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


def _bot_auth_dir() -> Path:
    return user_data_dir() / "bot-auth"


def _qr_dir() -> Path:
    return user_data_dir() / "qr"


def _skills_dir() -> Path:
    explicit = os.environ.get("PINPOINT_SKILLS_DIR")
    if explicit:
        return Path(explicit)

    try:
        package_root = importlib.resources.files("skills")
        return Path(str(package_root))
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        return repo_root / "skills"


def _load_env() -> dict[str, str]:
    env = DEFAULT_ENV.copy()
    path = _env_path()
    if path.exists():
        loaded = dotenv_values(path)
        env.update({k: str(v) for k, v in loaded.items() if v is not None})
    return env


def _bot_installed() -> tuple[bool, str]:
    bot_bin = shutil.which("pinpoint-bot")
    return (bot_bin is not None, bot_bin or "")


def _db_path() -> str:
    from database import DB_PATH

    return DB_PATH


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
    bot_installed, bot_bin = _bot_installed()
    print("")
    print(f"Saved config to {path}")
    print(f"User data dir: {user_data_dir()}")
    if bot_installed:
        print(f"Bot command detected: {bot_bin}")
    else:
        print("Bot command not detected. Install it with: npm install -g pinpoint-bot")
    print("Run `pinpoint api` to start the backend, `pinpoint search \"query\"` to search locally, or `pinpoint doctor` to validate setup.")
    return 0


def _api_ping() -> bool:
    try:
        with urlopen("http://127.0.0.1:5123/ping", timeout=2) as resp:
            return resp.status == 200
    except (URLError, OSError):
        return False


def _child_env(host: str, port: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(_load_env())
    env["PINPOINT_ENV_PATH"] = str(_env_path())
    env["PINPOINT_USER_DIR"] = str(user_data_dir())
    env["PINPOINT_LOG_DIR"] = str(_logs_dir())
    env["PINPOINT_AUTH_DIR"] = str(_bot_auth_dir())
    env["PINPOINT_QR_DIR"] = str(_qr_dir())
    env["PINPOINT_SKILLS_DIR"] = str(_skills_dir())
    env["PINPOINT_API_URL"] = f"http://127.0.0.1:{port}"
    env.setdefault("TZ", env.get("TZ", "UTC"))
    return env


def _wait_for_api(timeout_seconds: float = 15.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _api_ping():
            return True
        time.sleep(0.5)
    return False


def cmd_status(_args: argparse.Namespace) -> int:
    from database import init_db

    api_running = _api_ping()
    db_path = _db_path()
    bot_installed, bot_bin = _bot_installed()
    env_path = _env_path()
    auth_dir = _bot_auth_dir()
    qr_dir = _qr_dir()
    logs_dir = _logs_dir()
    conn = init_db(db_path)
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
    print(f"DB: {db_path}")
    print(f"Documents: {doc_count}")
    print(f"Watched folders: {watch_count}")
    print(f"Active jobs: {job_count}")
    print(f"Config: {env_path} ({'present' if env_path.exists() else 'missing'})")
    print(f"Bot: {'installed' if bot_installed else 'missing'}" + (f" ({bot_bin})" if bot_installed else ""))
    print(f"Bot auth dir: {auth_dir} ({'present' if auth_dir.exists() else 'missing'})")
    print(f"QR dir: {qr_dir} ({'present' if qr_dir.exists() else 'missing'})")
    print(f"Logs dir: {logs_dir} ({'present' if logs_dir.exists() else 'missing'})")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    env = _load_env()
    env_path = _env_path()
    logs_dir = _logs_dir()
    auth_dir = _bot_auth_dir()
    qr_dir = _qr_dir()
    skills_dir = _skills_dir()
    bot_installed, bot_bin = _bot_installed()

    checks: list[tuple[str, bool, str, bool]] = []
    checks.append(("config file", env_path.exists(), str(env_path)))
    checks.append(("user data dir", user_data_dir().exists(), str(user_data_dir()), True))
    checks.append(("logs dir", logs_dir.exists(), str(logs_dir), True))
    checks.append(("bot auth dir", auth_dir.exists(), str(auth_dir), True))
    checks.append(("qr dir", qr_dir.exists(), str(qr_dir), True))
    checks.append(("skills dir", skills_dir.exists(), str(skills_dir), True))
    checks.append(("database path parent", Path(_db_path()).parent.exists(), str(Path(_db_path()).parent), True))
    checks.append(("API ping", _api_ping(), "http://127.0.0.1:5123/ping", False))
    checks.append(("bot command", bot_installed, bot_bin or "pinpoint-bot not on PATH", False))
    checks.append(("Gemini configured", bool(env.get("GEMINI_API_KEY", "").strip()), "required for AI-heavy features", False))
    checks.append(("Ollama configured", bool(env.get("OLLAMA_MODEL", "").strip()), "optional bot fallback", False))

    missing = 0
    print(f"Pinpoint doctor ({__version__})")
    print(f"User data dir: {user_data_dir()}")
    config_ok, config_detail = checks[0][1], checks[0][2]
    print(f"[{'OK ' if config_ok else 'NO '}] {checks[0][0]}: {config_detail}")
    if not config_ok:
        missing += 1
    for label, ok, detail, required in checks[1:]:
        prefix = "OK " if ok else "NO "
        print(f"[{prefix}] {label}: {detail}")
        if required and not ok:
            missing += 1

    print("")
    if not env_path.exists():
        print("Next: run `pinpoint setup` to create the shared config.")
    elif not _api_ping():
        print("Next: run `pinpoint api` or `pinpoint start --api` to start the backend.")
    elif not bot_installed:
        print("Next: install the bot with `npm install -g pinpoint-bot` if you want WhatsApp support.")
    else:
        print("Core setup looks usable.")

    return 0 if missing == 0 else 1


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


def cmd_start(args: argparse.Namespace) -> int:
    logs_dir = _logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    _bot_auth_dir().mkdir(parents=True, exist_ok=True)
    _qr_dir().mkdir(parents=True, exist_ok=True)

    env = _child_env(args.host, args.port)
    api_log_path = logs_dir / "api.log"
    bot_log_path = logs_dir / "bot.log"
    api_log = api_log_path.open("a", encoding="utf-8")
    bot_log = bot_log_path.open("a", encoding="utf-8")

    children: list[subprocess.Popen] = []

    try:
        if args.bot_only:
            if not _api_ping():
                print(f"API is not reachable at {env['PINPOINT_API_URL']}. Start it first with `pinpoint api` or `pinpoint start --api`.")
                return 1
        else:
            print("Starting Pinpoint API...")
            api_proc = subprocess.Popen(
                [sys.executable, "-m", "pinpoint.cli", "api", "--host", args.host, "--port", str(args.port)],
                stdout=api_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            children.append(api_proc)
            if not _wait_for_api():
                print(f"API failed to become healthy within 15s. Check {api_log_path}")
                return 1
            print(f"API ready at {env['PINPOINT_API_URL']}")

        if args.api_only:
            print("API-only mode enabled. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

        bot_bin = shutil.which("pinpoint-bot")
        if not bot_bin:
            print("pinpoint-bot not found. Run: npm install -g pinpoint-bot")
            print("Continuing in API-only mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

        print(f"Starting bot via {bot_bin}...")
        bot_proc = subprocess.Popen(
            [bot_bin],
            env=env,
        )
        children.append(bot_proc)
        print("Pinpoint is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping Pinpoint...")
        return 0
    finally:
        for proc in reversed(children):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        api_log.close()
        bot_log.close()


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

    p_doctor = sub.add_parser("doctor", help="Validate the local Pinpoint setup")
    p_doctor.set_defaults(func=cmd_doctor)

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

    p_start = sub.add_parser("start", help="Start the API and, if installed, the WhatsApp bot")
    p_start.add_argument("--host", default="0.0.0.0")
    p_start.add_argument("--port", type=int, default=5123)
    mode = p_start.add_mutually_exclusive_group()
    mode.add_argument("--api", dest="api_only", action="store_true", help="Start only the API")
    mode.add_argument("--bot", dest="bot_only", action="store_true", help="Start only the bot (API must already be running)")
    p_start.set_defaults(func=cmd_start, api_only=False, bot_only=False)

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
