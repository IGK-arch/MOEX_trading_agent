"""
app/dashboard/snapshot_renderer.py — Static HTML snapshot of live metrics.

When the platform doesn't expose port 8501 to the outside world, we still
need a way for the jury / experts to see what the bot is doing. This
module writes a self-contained HTML file to `/data/dashboard_snapshot.html`
every `cfg.DASHBOARD_SNAPSHOT_INTERVAL_SEC` (default 5 min). The expert can
open it through the Monitor panel.

Design constraints:
  - No external network requests in the rendered file (everything inline)
  - Auto-refresh via `<meta http-equiv="refresh">` so the viewer doesn't
    need to hit reload
  - Cyberpunk dark theme (matches the team brand "404: Loss Not Found")
  - Latest 50 decisions table + KPI cards + activity feed

We read decisions / trades directly from SQLite — no Streamlit needed.
"""

from __future__ import annotations

import asyncio
import html
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import app.config as cfg
from app.dashboard.metrics_writer import build_metrics_snapshot
from app.utils.logging import get_logger

logger = get_logger(__name__)

_CSS = """
:root {
  --bg:
  --panel:
  --border:
  --fg:
  --muted:
  --accent:
  --accent2:
  --green:
  --red:
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 24px;
  line-height: 1.5;
}
h1 {
  font-size: 24px;
  margin: 0 0 4px 0;
  color: var(--accent);
}
.subtitle { color: var(--muted); margin-bottom: 24px; font-size: 13px; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}
.kpi {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.kpi-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.kpi-value { font-size: 22px; font-weight: 600; color: var(--fg); }
.kpi-value.green { color: var(--green); }
.kpi-value.red { color: var(--red); }
.kpi-value.cyan { color: var(--accent2); }
.section {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}
.section h2 { margin: 0 0 12px 0; font-size: 15px; color: var(--accent2); font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; }
tr:hover { background: rgba(255, 51, 96, 0.04); }
.tag {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 11px; font-weight: 500;
}
.tag.buy { background: rgba(76, 175, 80, 0.2); color: var(--green); }
.tag.sell { background: rgba(255, 82, 82, 0.2); color: var(--red); }
.tag.execute { background: rgba(0, 221, 255, 0.15); color: var(--accent2); }
.tag.no_trade { background: rgba(138, 138, 152, 0.15); color: var(--muted); }
.tag.veto { background: rgba(255, 165, 0, 0.15); color: orange; }
.footer { text-align: center; color: var(--muted); font-size: 11px; margin-top: 24px; }
"""

def _esc(s: Any) -> str:
    """Esc."""
    if s is None:
        return ""
    return html.escape(str(s))

def _kpi(label: str, value: str, klass: str = "") -> str:
    """Kpi."""
    return (
        f'<div class="kpi"><div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value {klass}">{_esc(value)}</div></div>'
    )

def _load_recent_decisions(limit: int = 50) -> list[dict]:
    """Load recent decisions."""
    db = cfg.DATA_DIR / "decisions.db"
    if not db.exists():
        return []
    try:
        with sqlite3.connect(str(db)) as cn:
            cn.row_factory = sqlite3.Row
            sql = (
                "SELECT decision_id, ticker, action, tier, direction, "
                "       combined_magnitude, expected_rr, rationale, created_at, "
                "       executed_bool, meta_score "
                "  FROM decisions ORDER BY created_at DESC LIMIT ?"
            )
            return [dict(r) for r in cn.execute(sql, (limit,)).fetchall()]
    except sqlite3.Error:
        return []

def _load_recent_trades(limit: int = 30) -> list[dict]:
    """Load recent trades."""
    db = cfg.DATA_DIR / "trades.db"
    if not db.exists():
        return []
    try:
        with sqlite3.connect(str(db)) as cn:
            cn.row_factory = sqlite3.Row
            try:
                rows = cn.execute(
                    "SELECT * FROM trades ORDER BY rowid DESC LIMIT ?", (limit,)
                ).fetchall()
            except sqlite3.OperationalError:
                return []
            return [dict(r) for r in rows]
    except sqlite3.Error:
        return []

def render_snapshot_html(*, refresh_sec: int = 300) -> str:
    """Return a full self-contained HTML document."""
    snap = build_metrics_snapshot()
    decisions = _load_recent_decisions(50)
    trades = _load_recent_trades(30)
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    eq = snap.get("equity_rub", 0.0)
    daily = snap.get("daily_pnl_pct", 0.0)
    max_dd = snap.get("max_dd_pct", 0.0)
    current_dd = snap.get("current_dd_pct", 0.0)
    n_trades = snap.get("n_trades_today", 0)
    n_open = snap.get("n_open_positions", 0)
    regime = snap.get("hmm_regime", "unknown")

    eq_class = "green" if eq >= 1_000_000 else "red"
    daily_class = "green" if daily >= 0 else "red"
    dd_class = "red" if max_dd > 0.05 else "cyan"

    kpis = "\n".join(
        [
            _kpi("Капитал, ₽", f"{eq:,.0f}", eq_class),
            _kpi("PnL за сегодня", f"{daily * 100:+.2f}%", daily_class),
            _kpi("Просадка тек.", f"{current_dd * 100:.2f}%", dd_class),
            _kpi("Просадка макс.", f"{max_dd * 100:.2f}%", dd_class),
            _kpi("Сделок сегодня", str(n_trades), "cyan"),
            _kpi("Открытых позиций", str(n_open), ""),
            _kpi("Режим HMM", regime, ""),
            _kpi("Бюджет polza ост., ₽", f"{snap.get('polza_budget_remaining_rub', 0):,.0f}", ""),
        ]
    )

    dec_rows: list[str] = []
    for d in decisions:
        action = (d.get("action") or "").lower()
        direction = (d.get("direction") or "").lower()
        meta = d.get("meta_score")
        meta_str = f"{float(meta):.2f}" if meta is not None else "—"
        dec_rows.append(
            f"<tr>"
            f"<td>{_esc(d.get('created_at', '')[:19])}</td>"
            f"<td><strong>{_esc(d.get('ticker'))}</strong></td>"
            f'<td><span class="tag {action}">{_esc(d.get("action"))}</span></td>'
            f'<td><span class="tag {direction}">{_esc(d.get("direction"))}</span></td>'
            f"<td>T{_esc(d.get('tier'))}</td>"
            f"<td>{float(d.get('combined_magnitude') or 0.0):.2f}</td>"
            f"<td>{meta_str}</td>"
            f"<td>{_esc((d.get('rationale') or '')[:120])}</td>"
            f"</tr>"
        )

    dec_table = (
        "<table><thead><tr>"
        "<th>Время</th><th>Тикер</th><th>Действие</th>"
        "<th>Направление</th><th>Tier</th><th>Magnitude</th>"
        "<th>Meta</th><th>Обоснование</th></tr></thead>"
        f"<tbody>{''.join(dec_rows)}</tbody></table>"
    )

    trade_rows_list: list[str] = []
    for t in trades:
        trade_time = t.get("trade_time") or t.get("tradetime", "")
        trade_ticker = t.get("ticker") or t.get("secid", "")
        trade_dir = (t.get("direction") or "").lower()
        trade_rows_list.append(
            f"<tr><td>{_esc(trade_time)}</td>"
            f"<td><strong>{_esc(trade_ticker)}</strong></td>"
            f'<td><span class="tag {trade_dir}">{_esc(t.get("direction", ""))}</span></td>'
            f"<td>{_esc(t.get('quantity', ''))}</td>"
            f"<td>{_esc(t.get('price', ''))}</td>"
            f"<td>{_esc(t.get('order_value', ''))}</td></tr>"
        )
    trade_rows = "".join(trade_rows_list)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta http-equiv="refresh" content="{refresh_sec}" />
<title>404: Loss Not Found — состояние агента</title>
<style>{_CSS}</style>
</head>
<body>
<h1>404: Loss Not Found</h1>
<div class="subtitle">
  Снимок состояния: {_esc(ts)} ·
  Обновление каждые {refresh_sec} сек ·
  Режим: {_esc(snap.get("run_mode", "paper"))} ·
  Live sizing: {_esc("да" if snap.get("live_sizing") else "нет")}
</div>

<div class="kpi-grid">{kpis}</div>

<div class="section">
<h2>Последние решения ({len(decisions)})</h2>
{dec_table}
</div>

<div class="section">
<h2>Последние сделки ({len(trades)})</h2>
<table><thead><tr><th>Время</th><th>Тикер</th><th>Направление</th>
<th>Количество</th><th>Цена</th><th>Стоимость, ₽</th></tr></thead>
<tbody>
{trade_rows}
</tbody>
</table>
</div>

<div class="footer">404: Loss Not Found — automatic snapshot, no manual control.</div>
</body></html>
"""

async def snapshot_loop(
    out_path: Path | None = None,
    interval_sec: float | None = None,
) -> None:
    """Background task — call from main.py via asyncio.create_task."""
    out_path = out_path or (cfg.DATA_DIR / "dashboard_snapshot.html")
    interval = float(interval_sec or cfg.DASHBOARD_SNAPSHOT_INTERVAL_SEC)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("snapshot_loop started", extra={"interval_sec": interval, "path": str(out_path)})
    while True:
        try:
            html_body = render_snapshot_html(refresh_sec=int(interval))
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(html_body, encoding="utf-8")
            tmp.replace(out_path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("snapshot render failed", extra={"error": str(exc)})
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise

def get_dashboard_mode() -> str:
    """
    Read /data/dashboard_mode.txt produced by scripts/probe_dashboard.py.
    Defaults to "snapshot" if file absent (safer).
    """
    p = cfg.DATA_DIR / "dashboard_mode.txt"
    if not p.exists():
        return "snapshot"
    try:
        mode = p.read_text(encoding="utf-8").strip().lower()
        if mode in ("external", "snapshot"):
            return mode
    except Exception:
        pass
    return "snapshot"

__all__ = ["render_snapshot_html", "snapshot_loop", "get_dashboard_mode"]
