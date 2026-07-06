"""Загрузка MT5 CSV для backtest."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import app.config as cfg
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

_MT5_COL_MAP: dict[str, str] = {
    "<date>": "date",
    "<time>": "time",
    "<open>": "open",
    "<high>": "high",
    "<low>": "low",
    "<close>": "close",
    "<tickvol>": "volume",
    "<vol>": "real_vol",
    "<spread>": "spread",

    "date": "date",
    "time": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vol": "real_vol",
    "tickvol": "volume",
    "tick volume": "volume",
}

_INTERVAL_FROM_SUFFIX: dict[str, int] = {
    "M1": 1, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

class CSVLoader:
    """Load MT5 OHLCV CSV files for backtest playback."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Init."""
        self._data_dir = data_dir or (cfg.DATA_DIR / "historical_csv")
        self._cache: dict[tuple[str, int], Any] = {}

    def _find_file(self, ticker: str, interval: int) -> Path | None:
        """Locate a CSV file for (ticker, interval). Returns None if not found."""
        ticker = ticker.upper()
        data_dir = self._data_dir

        if not data_dir.exists():
            return None

        reverse_map: dict[int, list[str]] = {}
        for suf, mins in _INTERVAL_FROM_SUFFIX.items():
            reverse_map.setdefault(mins, []).append(suf)

        suffixes = reverse_map.get(interval, [])

        for suf in suffixes:
            candidate = data_dir / f"{ticker}_{suf}.csv"
            if candidate.exists():
                return candidate

        candidates = sorted(data_dir.glob(f"{ticker}_*.csv"))
        if candidates:

            best: Path | None = None
            best_diff = float("inf")
            for path in candidates:
                stem = path.stem
                parts = stem.split("_")
                if len(parts) >= 2:
                    suf = parts[-1].upper()
                    mins = _INTERVAL_FROM_SUFFIX.get(suf, 0)
                    if mins > 0:
                        diff = abs(mins - interval)
                        if diff < best_diff:
                            best_diff = diff
                            best = path
            if best:
                logger.warning(
                    "CSV: exact interval not found, using closest file",
                    extra={"ticker": ticker, "requested_interval": interval, "using": best.name},
                )
                return best

        return None

    def _infer_interval(self, path: Path) -> int:
        """Derive interval in minutes from filename suffix."""
        stem = path.stem
        parts = stem.upper().split("_")
        if len(parts) >= 2:
            suf = parts[-1]
            return _INTERVAL_FROM_SUFFIX.get(suf, 60)
        return 60

    def _normalise_columns(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Map MT5 column names to unified lowercase schema."""
        rename = {}
        for col in df.columns:
            mapped = _MT5_COL_MAP.get(col.lower().strip())
            if mapped:
                rename[col] = mapped
        if rename:
            df = df.rename(columns=rename)
        return df

    def _parse_datetime(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Merge 'date' + 'time' columns into a single 'begin' datetime (UTC)."""
        if "begin" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["begin"]):
                df["begin"] = pd.to_datetime(df["begin"], utc=True, errors="coerce")
            return df

        if "date" in df.columns and "time" in df.columns:
            df["begin"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
            )

            df["begin"] = df["begin"].dt.tz_localize("Europe/Moscow", ambiguous="NaT")
            df["begin"] = df["begin"].dt.tz_convert("UTC")
            df = df.drop(columns=["date", "time"], errors="ignore")
        elif "date" in df.columns:
            df["begin"] = pd.to_datetime(df["date"], errors="coerce")
            if df["begin"].dt.tz is None:
                df["begin"] = df["begin"].dt.tz_localize("Europe/Moscow")
                df["begin"] = df["begin"].dt.tz_convert("UTC")
            df = df.drop(columns=["date"], errors="ignore")

        return df

    def load(
        self,
        ticker: str,
        interval: int = 60,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> "pd.DataFrame | list":
        """Load OHLCV data for (ticker, interval).

        Args:
            ticker: MOEX ticker, e.g. "SBER".
            interval: candle interval in minutes.
            start: optional start datetime (inclusive).
            end: optional end datetime (inclusive).
        Returns:
            pd.DataFrame: sorted ascending by 'begin'; empty on failure.
        """
        cache_key = (ticker.upper(), interval)

        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            path = self._find_file(ticker, interval)
            if path is None:
                logger.warning(
                    "CSV file not found",
                    extra={"ticker": ticker, "interval": interval, "data_dir": str(self._data_dir)},
                )
                return pd.DataFrame() if _HAS_PANDAS else []

            if not _HAS_PANDAS:
                logger.error("pandas not installed — cannot load CSV")
                return []

            try:
                df = pd.read_csv(path, sep=",", engine="python", on_bad_lines="skip")
                df = self._normalise_columns(df)
                df = self._parse_datetime(df)

                for col in ("open", "high", "low", "close", "volume"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna(subset=["open", "high", "low", "close", "begin"])

                df = df.sort_values("begin").reset_index(drop=True)

                df["end"] = df["begin"] + pd.Timedelta(minutes=interval)

                logger.info(
                    "CSV loaded",
                    extra={"ticker": ticker, "interval": interval, "rows": len(df), "file": path.name},
                )

            except Exception as exc:
                logger.error(
                    "CSV load failed",
                    extra={"ticker": ticker, "path": str(path), "error": str(exc)},
                )
                return pd.DataFrame()

            self._cache[cache_key] = df

        if start is not None:
            if isinstance(start, str):
                start = pd.Timestamp(start, tz="UTC")
            elif isinstance(start, datetime) and start.tzinfo is None:
                start = pd.Timestamp(start, tz="UTC")
            df = df[df["begin"] >= start]

        if end is not None:
            if isinstance(end, str):
                end = pd.Timestamp(end, tz="UTC")
            elif isinstance(end, datetime) and end.tzinfo is None:
                end = pd.Timestamp(end, tz="UTC")
            df = df[df["begin"] <= end]

        return df.reset_index(drop=True)

    def load_all(
        self,
        tickers: list[str] | None = None,
        interval: int = 60,
    ) -> "dict[str, pd.DataFrame | list]":
        """Load data for all tickers. Returns: dict[ticker, DataFrame]."""
        if tickers is None:
            tickers = cfg.TICKERS

        results: dict[str, Any] = {}
        for ticker in tickers:
            df = self.load(ticker, interval=interval)
            results[ticker] = df

        loaded = sum(1 for v in results.values()
                     if _HAS_PANDAS and isinstance(v, pd.DataFrame) and not v.empty)
        logger.info(
            "CSV bulk load complete",
            extra={"requested": len(tickers), "loaded": loaded, "interval": interval},
        )
        return results

    def available_tickers(self, interval: int | None = None) -> list[str]:
        """Return tickers with at least one CSV file present."""
        if not self._data_dir.exists():
            return []

        tickers: set[str] = set()
        for path in self._data_dir.glob("*.csv"):
            parts = path.stem.upper().split("_")
            if len(parts) >= 2:
                ticker = parts[0]
                if interval is not None:
                    suf = parts[-1]
                    mins = _INTERVAL_FROM_SUFFIX.get(suf, 0)
                    if mins != interval:
                        continue
                tickers.add(ticker)
        return sorted(tickers)

    def date_range(self, ticker: str, interval: int = 60) -> tuple[datetime | None, datetime | None]:
        """Return (first_candle_dt, last_candle_dt) for the given ticker/interval."""
        df = self.load(ticker, interval=interval)
        if not _HAS_PANDAS or not isinstance(df, pd.DataFrame) or df.empty:
            return None, None
        return df["begin"].iloc[0].to_pydatetime(), df["begin"].iloc[-1].to_pydatetime()

_csv_loader: CSVLoader | None = None

def get_csv_loader() -> CSVLoader:
    """Get csv loader."""
    global _csv_loader
    if _csv_loader is None:
        _csv_loader = CSVLoader()
    return _csv_loader
