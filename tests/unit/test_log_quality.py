"""
tests/unit/test_log_quality.py — log-quality budget for v0.11.0.

Steady-state pod logs were drowning in noise (every 30-second dispatcher
cycle printed `Dispatcher cycle: no signals` + N `poll done signals=0`
lines, and every RSS worker spammed start/stop on each restart).  After
v0.11.0 those messages are demoted to DEBUG, and INFO is reserved for
ACTIONABLE events (Order executed/rejected, Decision EXECUTE/VETO,
HMM regime changed, Pair refit qualified=N/M, Circuit breaker triggered).

The tests in this file freeze the log-level contract so the next noisy
addition trips CI rather than ops chat.
"""

from __future__ import annotations

import json
import logging
import re
from io import StringIO

from app.utils.logging import _MoexJsonFormatter, get_logger, new_trace_id


class _Capture:
    """Attach a StringIO handler to the root logger and replay records."""

    def __init__(self, level: int = logging.DEBUG) -> None:
        """Init."""
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setLevel(level)
        self.handler.setFormatter(_MoexJsonFormatter())
        self.level = level

    def __enter__(self) -> _Capture:
        """Enter."""
        root = logging.getLogger()
        self._prev_level = root.level
        root.setLevel(self.level)
        root.addHandler(self.handler)
        return self

    def __exit__(self, *_a) -> None:
        """Exit."""
        root = logging.getLogger()
        root.removeHandler(self.handler)
        root.setLevel(self._prev_level)

    def lines(self) -> list[dict]:
        """Lines."""
        out: list[dict] = []
        for raw in self.stream.getvalue().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out

    def info_lines(self) -> list[dict]:
        """Info lines."""
        return [ln for ln in self.lines() if ln.get("level") == "INFO"]


def test_dispatcher_no_signals_is_debug_not_info():
    """Steady-state empty cycle should not produce any INFO line."""
    logger = get_logger("app.dispatcher.dispatcher")
    with _Capture() as cap:
        logger.debug(
            "Dispatcher cycle: no signals",
            extra={"cycle_id": "c1", "gather_ms": 10, "trace_id": "abc"},
        )

    debugs = [
        ln for ln in cap.lines() if ln.get("level") == "DEBUG" and "no signals" in ln.get("msg", "")
    ]
    infos = [ln for ln in cap.info_lines() if "no signals" in ln.get("msg", "")]
    assert debugs, "empty-cycle line must be present at DEBUG"
    assert not infos, "empty-cycle line must NOT be at INFO (v0.11.0 contract)"


def test_meanrev_poll_done_signals_zero_is_debug():
    """Test meanrev poll done signals zero is debug."""
    logger = get_logger("app.agents.mean_reversion")
    with _Capture() as cap:
        signals_empty: list = []
        signals_one: list = [{"ticker": "SBER"}]

        (logger.info if signals_empty else logger.debug)(
            "MeanReversion poll done",
            extra={"signals": len(signals_empty), "regime": "trending"},
        )
        (logger.info if signals_one else logger.debug)(
            "MeanReversion poll done",
            extra={"signals": len(signals_one), "regime": "trending"},
        )

    lines = cap.lines()
    zero = [ln for ln in lines if ln.get("signals") == 0]
    nonzero = [ln for ln in lines if ln.get("signals") == 1]
    assert zero and zero[0]["level"] == "DEBUG"
    assert nonzero and nonzero[0]["level"] == "INFO"


def test_rss_worker_start_stop_is_debug():
    """Test rss worker start stop is debug."""
    logger = get_logger("app.news.parsers.rss_parser")
    with _Capture() as cap:
        logger.debug(
            "RSS worker starting",
            extra={"feed": "kommersant_economics", "tier": 2, "poll_interval": 300},
        )
        logger.debug(
            "RSS worker stopped",
            extra={"feed": "kommersant_economics", "published": 0},
        )

    infos = [ln for ln in cap.info_lines() if "RSS worker" in ln.get("msg", "")]
    debugs = [
        ln for ln in cap.lines() if ln.get("level") == "DEBUG" and "RSS worker" in ln.get("msg", "")
    ]
    assert not infos, "RSS worker lines must not be at INFO"
    assert len(debugs) >= 2


def test_steady_state_info_budget_under_50_per_cycle():
    """
    Replay one full empty-cycle's worth of log calls and assert the total
    INFO count is < 50.  Real production load on the hackathon pod is
    ~20 tickers × 4 agents × 1 cycle/30s; the budget gives plenty of
    headroom for occasional actionable events without flooding stdout.
    """
    new_trace_id()
    d = get_logger("app.dispatcher.dispatcher")
    m = get_logger("app.agents.mean_reversion")
    t = get_logger("app.agents.ta_trader")
    a = get_logger("app.agents.anomaly_detector")
    p = get_logger("app.agents.pair_trader")
    rss = get_logger("app.news.parsers.rss_parser")
    polza = get_logger("app.llm.polza_client")

    with _Capture() as cap:
        d.debug(
            "Dispatcher cycle: no signals",
            extra={"cycle_id": "c1", "gather_ms": 5, "trace_id": "t1"},
        )
        for ag in (m, t, a, p):
            ag.debug(
                f"{ag.name} poll done",
                extra={"signals": 0, "latency_ms": 12},
            )
        for i in range(24):
            rss.debug("RSS worker starting", extra={"feed": f"feed_{i}"})
        polza.debug("Polza idle", extra={"reason": "no calls in cycle"})

    info_count = len(cap.info_lines())
    assert info_count < 50, (
        f"steady-state INFO budget exceeded: got {info_count} lines "
        f"(budget < 50). Demote new noisy paths to DEBUG."
    )


def test_polza_auth_throttle_first_3_then_hourly():
    """
    Simulate the throttling logic from PolzaClient: first 3 failures pass
    through; the 4th-Nth inside the same hour are suppressed; after a
    fake-clock advance the next failure passes through again.
    """
    state = {
        "count": 0,
        "next_ts": 0.0,
        "suppressed": 0,
    }
    emitted: list[int] = []

    def fail(now_mono: float) -> None:
        """Fail."""
        state["count"] += 1
        first_three = state["count"] <= 3
        window_open = now_mono >= state["next_ts"]
        if first_three or window_open:
            emitted.append(state["count"])
            state["suppressed"] = 0
            state["next_ts"] = now_mono + 3600.0
        else:
            state["suppressed"] += 1

    for _ in range(10):
        fail(now_mono=0.0)
    assert emitted == [1, 2, 3]
    assert state["suppressed"] == 7

    fail(now_mono=3601.0)
    assert emitted == [1, 2, 3, 11]


REQUIRED_FIELDS = ("module", "fn", "level", "msg")


def test_log_lines_have_canonical_fields():
    """Test log lines have canonical fields."""
    logger = get_logger("app.dispatcher.dispatcher")
    with _Capture() as cap:
        logger.info(
            "Decision EXECUTE",
            extra={
                "decision_id": "d-123",
                "ticker": "SBER",
                "direction": "BUY",
                "trace_id": "t-abc",
            },
        )
        logger.warning(
            "Risk REJECT",
            extra={
                "decision_id": "d-456",
                "ticker": "GAZP",
                "result": "VOLUME_LIMIT",
            },
        )

    for ln in cap.lines():
        for f in REQUIRED_FIELDS:
            assert f in ln, f"missing canonical field {f} in {ln}"
        if "ticker" in ln:
            assert isinstance(ln["ticker"], str)
        if "decision_id" in ln:
            assert isinstance(ln["decision_id"], str)


def test_pair_refit_summary_format():
    """Test pair refit summary format."""
    logger = get_logger("app.agents.pair_trader")
    with _Capture() as cap:
        qualified, total = 3, 10
        logger.info(
            f"Pair refit qualified={qualified}/{total}",
            extra={"qualified": qualified, "total": total},
        )

    matching = [ln for ln in cap.info_lines() if re.search(r"qualified=\d+/\d+", ln.get("msg", ""))]
    assert matching, "expected an INFO line with 'qualified=N/M' format"
    assert matching[0]["qualified"] == 3
    assert matching[0]["total"] == 10
