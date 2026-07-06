"""Структурное JSON-логирование."""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from typing import Any

try:
    from pythonjsonlogger import jsonlogger  # noqa: F401  # type: ignore

    _HAS_JSONLOGGER = True
except ImportError:
    _HAS_JSONLOGGER = False

from contextvars import ContextVar

from app.config import LOGS_DIR

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

def new_trace_id() -> str:
    """Generate a fresh trace_id and store it in the context."""
    tid = uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid

def get_trace_id() -> str:
    """Get trace id."""
    return _trace_id_var.get() or "—"

def set_trace_id(tid: str) -> None:
    """Set trace id."""
    _trace_id_var.set(tid)

class _MoexJsonFormatter(logging.Formatter):
    """Formats log records as compact JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Format."""
        import json as _json

        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        base: dict[str, Any] = {
            "ts": ts,
            "trace_id": get_trace_id(),
            "level": record.levelname,
            "module": record.module,
            "fn": record.funcName,
            "msg": record.getMessage(),
        }

        skip = {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
        for key, val in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                base[key] = val

        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        try:
            return _json.dumps(base, ensure_ascii=False, default=str)
        except Exception:
            base["msg"] = str(base.get("msg", ""))
            return _json.dumps(base, ensure_ascii=False, default=str)

class _DailyFileHandler(logging.StreamHandler):
    """Writes log records to /data/logs/YYYY-MM-DD.jsonl with daily rotation."""

    def __init__(self) -> None:
        """Init."""
        super().__init__(stream=sys.stdout)
        self._current_date = ""
        self._file_handle: Any = None
        self._open_today()

    def _today_str(self) -> str:
        """Today str."""
        return datetime.now().strftime("%Y-%m-%d")

    def _open_today(self) -> None:
        """Open today."""
        date_str = self._today_str()
        if date_str == self._current_date:
            return
        if self._file_handle:
            with suppress(Exception):
                self._file_handle.close()
        path = LOGS_DIR / f"{date_str}.jsonl"
        self._file_handle = open(path, "a", encoding="utf-8", buffering=1)
        self.stream = self._file_handle
        self._current_date = date_str

    def emit(self, record: logging.LogRecord) -> None:
        """Emit."""
        self._open_today()
        super().emit(record)

_configured = False

def setup_logging(level: str = "INFO") -> None:
    """Configure JSON logging to daily file + stdout. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    formatter = _MoexJsonFormatter()

    file_handler = _DailyFileHandler()
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "json") == "pretty":
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(module)s.%(funcName)s — %(message)s")
        )
    else:
        console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for noisy in ["httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)

@contextmanager
def track_latency(stage: str, logger: logging.Logger | None = None) -> Iterator[dict[str, Any]]:
    """Log latency_ms when the block exits.

    Emits an INFO log line `"{stage}_latency"` with `extra={"stage": stage,
    "latency_ms": <int>}` so that Grafana/Loki can parse it as JSON. The
    yielded dict is mutable — callers may stash additional context (e.g.
    `ctx["n_items"] = 5`) which is merged into the extras at exit time.

    Usage:
        with track_latency("risk_eval") as ctx:
            result = await risk.evaluate(d)
            ctx["ticker"] = d.ticker
        # logs INFO "risk_eval_latency" with latency_ms automatically

    Args:
        stage: short stage label, used in both the message and the
            ``stage`` extra field (e.g. ``"gather"``, ``"risk_eval"``).
        logger: optional logger to write to. Defaults to a module-named
            logger inside this file so call sites need not pass anything.

    Yields:
        dict[str, Any]: mutable context dict merged into the log extras.
    """
    log = logger if logger is not None else get_logger("app.utils.logging")
    ctx: dict[str, Any] = {}
    start = time.monotonic()
    try:
        yield ctx
    finally:
        elapsed_ms = round((time.monotonic() - start) * 1000)
        extra = {"stage": stage, "latency_ms": elapsed_ms}
        for k, v in ctx.items():
            if k not in extra:
                extra[k] = v
        log.info(f"{stage}_latency", extra=extra)
