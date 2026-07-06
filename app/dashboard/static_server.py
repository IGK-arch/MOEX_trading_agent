"""Lightweight stdlib HTTP server for the dashboard on port 8501.

Phase 21 (v0.0.21) — replaces Streamlit for production. Solves three problems
at once:

1. **k8s liveness probe failure** — Streamlit's `enableXsrfProtection=false`
   is silently reverted in 1.20+, so the probe gets `Empty reply from server`
   from `0.0.0.0:8501` because its Host header doesn't match. We answer
   `/healthz`, `/livez`, `/readyz`, `/_stcore/health` with 200 OK — probe
   succeeds, pod stays alive.

2. **External dashboard inaccessible** — same XSRF problem from public URL
   `http://93.77.180.244:8501`. We serve `/data/dashboard_snapshot.html`
   (already regenerated every 5 min by `snapshot_loop`) on `/` — jury opens
   the IP and sees the actual dashboard immediately.

3. **No JS / no WebSocket** — bulletproof for the jury demo. The HTML uses
   `<meta http-equiv="refresh">` for auto-refresh. Zero deps beyond stdlib.

Threading model: `start_static_server()` spins one daemon thread running
`ThreadingTCPServer.serve_forever()`. Each request gets its own thread.
We don't block the asyncio event loop in `app.main`.
"""

from __future__ import annotations

import contextlib
import http.server
import logging
import socketserver
import threading
from typing import Any

import app.config as cfg

logger = logging.getLogger(__name__)

_HEALTH_PATHS = frozenset(
    {
        "/_stcore/health",
        "/_stcore/healthz",
        "/healthz",
        "/health",
        "/livez",
        "/readyz",
        "/ping",
        "/status",
    }
)

def _load_snapshot_html() -> bytes:
    """Read the pre-rendered dashboard snapshot or fall back to live render."""
    snapshot_path = cfg.DATA_DIR / "dashboard_snapshot.html"
    if snapshot_path.exists():
        try:
            return snapshot_path.read_bytes()
        except OSError as exc:
            logger.warning("static_server: read snapshot failed", extra={"error": str(exc)})
    try:
        from app.dashboard.snapshot_renderer import render_snapshot_html

        return render_snapshot_html(refresh_sec=30).encode("utf-8")
    except Exception as exc:
        logger.error("static_server: fallback render failed", extra={"error": str(exc)})
        return _emergency_html(str(exc)).encode("utf-8")

def _emergency_html(detail: str = "") -> str:
    """Minimal HTML if everything else fails."""
    safe = (detail or "snapshot unavailable").replace("<", "&lt;")[:200]
    return (
        "<!doctype html><html><head>"
        "<meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='30'>"
        "<title>404: Loss Not Found</title>"
        "<style>body{background:#0a0a0a;color:#e0e0e0;font-family:system-ui;"
        "padding:48px;max-width:720px;margin:auto;}"
        "h1{color:#ff4b4b;}code{background:#1a1a1a;padding:2px 6px;"
        "border-radius:4px;}</style></head>"
        "<body><h1>404: Loss Not Found</h1>"
        "<p>Bot is starting. Dashboard will populate within ~30 seconds.</p>"
        f"<p style='color:#888'><code>{safe}</code></p></body></html>"
    )

def _load_metrics_json() -> bytes:
    """Return the latest live metrics summary as JSON."""
    summary_path = cfg.DATA_DIR / "metrics_summary.json"
    if summary_path.exists():
        try:
            return summary_path.read_bytes()
        except OSError:
            pass
    live_path = cfg.DATA_DIR / "metrics_live.jsonl"
    if live_path.exists():
        try:
            with live_path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                tail = fh.read().splitlines()
                for line in reversed(tail):
                    line = line.strip()
                    if line.startswith(b"{") and line.endswith(b"}"):
                        return line
        except OSError:
            pass
    return b'{"status": "no metrics available yet"}'

class _Handler(http.server.BaseHTTPRequestHandler):
    """Single-request handler. Suppresses default access logging."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Log message (suppressed)."""
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        """Send."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib API)
        """Do GET."""
        path = self.path.split("?", 1)[0]

        if path in _HEALTH_PATHS:
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return

        if path == "/metrics.json" or path == "/api/metrics":
            self._send(200, _load_metrics_json(), "application/json; charset=utf-8")
            return

        self._send(200, _load_snapshot_html(), "text/html; charset=utf-8")

    def do_HEAD(self) -> None:  # noqa: N802
        """Do HEAD."""
        path = self.path.split("?", 1)[0]
        if path in _HEALTH_PATHS:
            self._send(200, b"", "text/plain; charset=utf-8")
        else:
            self._send(200, b"", "text/html; charset=utf-8")

class _ReusableServer(socketserver.ThreadingTCPServer):
    """Allow rapid restart without TIME_WAIT errors."""

    allow_reuse_address = True
    daemon_threads = True

_server_thread: threading.Thread | None = None
_server_instance: _ReusableServer | None = None

def start_static_server(port: int = 8501, host: str = "0.0.0.0") -> None:
    """Start the static HTTP server in a background daemon thread.

    Idempotent — calling twice is a no-op. Logs errors but does not raise
    (the bot must keep running even if the dashboard fails to bind).
    """
    global _server_thread, _server_instance
    if _server_thread is not None and _server_thread.is_alive():
        logger.info("static_server: already running, skipping")
        return

    try:
        server = _ReusableServer((host, port), _Handler)
    except OSError as exc:
        logger.error(
            "static_server: bind failed", extra={"host": host, "port": port, "error": str(exc)}
        )
        return

    _server_instance = server
    thread = threading.Thread(
        target=server.serve_forever,
        name="static_server",
        daemon=True,
    )
    thread.start()
    _server_thread = thread
    logger.info(
        "static_server started",
        extra={
            "host": host,
            "port": port,
            "endpoints": ["/", "/_stcore/health", "/healthz", "/livez", "/readyz", "/metrics.json"],
        },
    )

def stop_static_server() -> None:
    """Stop the static server (for clean shutdown)."""
    global _server_thread, _server_instance
    if _server_instance is not None:
        try:
            _server_instance.shutdown()
            _server_instance.server_close()
        except Exception as exc:
            logger.warning("static_server: stop error", extra={"error": str(exc)})
    _server_instance = None
    _server_thread = None

__all__ = ["start_static_server", "stop_static_server"]
