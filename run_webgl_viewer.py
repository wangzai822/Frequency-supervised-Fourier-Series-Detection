# -*- coding: utf-8 -*-
"""
FSD WebGL Damage Intelligence Viewer - Launcher
================================================

This file is the local launcher for the WebGL-based FSD database viewer.

Run from the current SQLite directory:

    python run_webgl_viewer.py

Then open:

    http://127.0.0.1:8000

Project layout expected:

    SQLite/
    ├── run_webgl_viewer.py
    ├── fsd_geometry.py
    ├── fsd_db.py
    ├── viewer_app/
    └── webgl_viewer/
        ├── __init__.py
        ├── server.py
        ├── state.py
        ├── fsd_service.py
        └── static/
            ├── index.html
            ├── style.css
            ├── api.js
            ├── webgl_canvas.js
            └── app.js

Notes
-----
- The visual UI will be fully English to look professional and paper/demo ready.
- This launcher only starts the local FastAPI/Uvicorn backend.
- The actual WebGL interface and API routes are implemented in webgl_viewer/server.py.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------
FILE = Path(__file__).resolve()
SQLITE_DIR = FILE.parent
PROJECT_DIR = SQLITE_DIR.parent

DEFAULT_APP_IMPORT = "webgl_viewer.server:app"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def inject_python_paths() -> None:
    """
    Make sure local modules are importable.

    Required because the backend will reuse existing modules such as:
        - fsd_geometry.py
        - viewer_app.repository
        - viewer_app.utils
    """
    paths = [
        SQLITE_DIR,
        PROJECT_DIR,
    ]

    for p in paths:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def is_port_available(host: str, port: int) -> bool:
    """
    Check whether host:port is available.

    This is only a convenience check. Uvicorn will still do its own binding.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((host, int(port))) != 0
    except Exception:
        return True


def find_free_port(host: str, start_port: int, max_tries: int = 50) -> int:
    """
    Find a free port starting from start_port.
    """
    port = int(start_port)
    for _ in range(max_tries):
        if is_port_available(host, port):
            return port
        port += 1
    raise RuntimeError(
        f"No free port found from {start_port} to {start_port + max_tries - 1}."
    )


def browser_host_for(host: str) -> str:
    """
    Convert binding host to a browser-friendly host.
    """
    host = str(host).strip()
    if host in ["0.0.0.0", "::"]:
        return "127.0.0.1"
    return host or "127.0.0.1"


def open_browser_later(url: str, delay: float = 1.0) -> None:
    """
    Open browser in a daemon thread after a short delay.
    """
    def _open() -> None:
        time.sleep(max(float(delay), 0.0))
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def human_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def print_banner(
    host: str,
    port: int,
    url: str,
    app_import: str,
    reload: bool,
    auto_browser: bool,
) -> None:
    """
    Print an English professional startup banner.
    """
    line = "=" * 78

    print()
    print(line)
    print(" FSD WebGL Damage Intelligence Viewer")
    print(" High-performance local viewer for Fourier-based defect databases")
    print(line)
    print(f" Working directory : {human_path(SQLITE_DIR)}")
    print(f" Project root      : {human_path(PROJECT_DIR)}")
    print(f" ASGI app          : {app_import}")
    print(f" Host              : {host}")
    print(f" Port              : {port}")
    print(f" URL               : {url}")
    print(f" Reload            : {'enabled' if reload else 'disabled'}")
    print(f" Auto browser      : {'enabled' if auto_browser else 'disabled'}")
    print("-" * 78)
    print(" Backend           : FastAPI + Uvicorn")
    print(" Frontend          : HTML/CSS/JavaScript")
    print(" Renderer          : PixiJS / WebGL")
    print(" UI language       : English")
    print("-" * 78)
    print(" Press CTRL+C to stop the server.")
    print(line)
    print()


def check_basic_layout() -> None:
    """
    Check that the expected directories exist.

    This function does not create files silently, because we are building
    the project step by step and want missing files to be obvious.
    """
    webgl_dir = SQLITE_DIR / "webgl_viewer"
    static_dir = webgl_dir / "static"

    if not webgl_dir.exists():
        print(
            "[WARN] Missing directory: webgl_viewer\n"
            "       Please create it before continuing."
        )

    if not static_dir.exists():
        print(
            "[WARN] Missing directory: webgl_viewer/static\n"
            "       Please create it before continuing."
        )

    server_py = webgl_dir / "server.py"
    if not server_py.exists():
        print(
            "[WARN] Missing file: webgl_viewer/server.py\n"
            "       This file will be created in the next step."
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_webgl_viewer.py",
        description=(
            "Launch the local FSD WebGL Damage Intelligence Viewer. "
            "The interface is designed as an English professional research/demo viewer."
        ),
    )

    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Server host. Default: 127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port. Default: 8000",
    )

    parser.add_argument(
        "--app",
        type=str,
        default=DEFAULT_APP_IMPORT,
        help=f"ASGI app import path. Default: {DEFAULT_APP_IMPORT}",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload for development.",
    )

    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="If the requested port is occupied, automatically use the next free port.",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )

    parser.add_argument(
        "--browser-delay",
        type=float,
        default=1.0,
        help="Delay before opening browser, in seconds. Default: 1.0",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level. Default: info",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------
def main() -> None:
    inject_python_paths()
    check_basic_layout()

    args = parse_args()

    host = str(args.host).strip() or DEFAULT_HOST
    port = int(args.port)
    app_import = str(args.app).strip() or DEFAULT_APP_IMPORT
    reload_enabled = bool(args.reload)
    auto_browser = not bool(args.no_browser)

    if args.auto_port:
        port = find_free_port(host=browser_host_for(host), start_port=port)
    else:
        bind_check_host = browser_host_for(host)
        if not is_port_available(bind_check_host, port):
            print()
            print(f"[WARN] Port {port} appears to be occupied on {bind_check_host}.")
            print("       You can use one of the following:")
            print(f"       python run_webgl_viewer.py --port {port + 1}")
            print(f"       python run_webgl_viewer.py --auto-port")
            print()

    url = f"http://{browser_host_for(host)}:{port}"

    # Provide environment variables for downstream modules if needed.
    os.environ.setdefault("FSD_WEBGL_SQLITE_DIR", str(SQLITE_DIR))
    os.environ.setdefault("FSD_WEBGL_PROJECT_DIR", str(PROJECT_DIR))
    os.environ.setdefault("FSD_WEBGL_STATIC_DIR", str(SQLITE_DIR / "webgl_viewer" / "static"))

    try:
        import uvicorn
    except Exception as e:
        print()
        print("[ERROR] Uvicorn/FastAPI backend dependency is not available.")
        print("        Please install required packages:")
        print()
        print("        pip install fastapi uvicorn")
        print()
        print(f"        Original error: {e}")
        print()
        raise SystemExit(1)

    print_banner(
        host=host,
        port=port,
        url=url,
        app_import=app_import,
        reload=reload_enabled,
        auto_browser=auto_browser,
    )

    if auto_browser:
        open_browser_later(url, delay=float(args.browser_delay))

    # Important:
    # Use import string instead of app object.
    # This works better with --reload on Windows.
    uvicorn.run(
        app_import,
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=args.log_level,
        reload_dirs=[str(SQLITE_DIR / "webgl_viewer")] if reload_enabled else None,
    )


if __name__ == "__main__":
    main()