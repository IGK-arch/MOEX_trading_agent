"""Технические индикаторы (vectorized)."""

from __future__ import annotations

from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import numpy as np  # noqa: F401  # type: ignore
    import pandas as pd  # type: ignore

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import pandas_ta as ta  # type: ignore

    _HAS_TA = True
except ImportError:
    try:
        import pandas_ta_classic as ta  # type: ignore

        _HAS_TA = True
    except ImportError:
        _HAS_TA = False

def _check_df(df: pd.DataFrame, min_rows: int = 2) -> bool:
    """Return True if df is usable."""
    if not _HAS_PANDAS:
        return False
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False
    if len(df) < min_rows:
        return False
    required = {"open", "high", "low", "close"}
    return required.issubset(set(df.columns))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range.

    Returns:
        pd.Series: ATR values, NaN for first `period` rows.
    """
    if not _check_df(df, period + 1):
        return pd.Series(dtype=float) if _HAS_PANDAS else []

    if _HAS_TA:
        result = ta.atr(df["high"], df["low"], df["close"], length=period)
        if result is not None:
            return result

    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) + DI+ / DI-.

    Returns:
        pd.DataFrame: columns ADX, DMP (DI+), DMN (DI-).
    """
    if not _check_df(df, period * 2):
        return pd.DataFrame(columns=["ADX", "DMP", "DMN"]) if _HAS_PANDAS else {}

    if _HAS_TA:
        result = ta.adx(df["high"], df["low"], df["close"], length=period)
        if result is not None and hasattr(result, "columns"):
            rename = {}
            for col in result.columns:
                upper = col.upper()
                if upper.startswith("ADX"):
                    rename[col] = "ADX"
                elif upper.startswith("DMP"):
                    rename[col] = "DMP"
                elif upper.startswith("DMN") or upper.startswith("DMI_M"):
                    rename[col] = "DMN"
            if rename:
                result = result.rename(columns=rename)
            return result

    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    dm_plus = (high - prev_high).clip(lower=0).where((high - prev_high) > (prev_low - low), 0)
    dm_minus = (prev_low - low).clip(lower=0).where((prev_low - low) > (high - prev_high), 0)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    dmp = dm_plus.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth * 100
    dmn = dm_minus.ewm(alpha=1 / period, adjust=False).mean() / tr_smooth * 100
    dx = (dmp - dmn).abs() / (dmp + dmn).replace(0, float("nan")) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({"ADX": adx, "DMP": dmp, "DMN": dmn}, index=df.index)

def compute_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """Exponential Moving Average. Returns pd.Series."""
    if not _check_df(df):
        return pd.Series(dtype=float) if _HAS_PANDAS else []
    if column not in df.columns:
        return pd.Series(dtype=float)

    if _HAS_TA:
        result = ta.ema(df[column], length=period)
        if result is not None:
            return result

    return df[column].ewm(span=period, adjust=False).mean()

def compute_sma(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """Simple Moving Average."""
    if not _check_df(df):
        return pd.Series(dtype=float) if _HAS_PANDAS else []
    return df[column].rolling(period).mean()

def compute_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Relative Strength Index.

    Returns:
        pd.Series: values in [0, 100], NaN for first `period` rows.
    """
    if not _check_df(df, period + 1):
        return pd.Series(dtype=float) if _HAS_PANDAS else []

    if _HAS_TA:
        result = ta.rsi(df[column], length=period)
        if result is not None:
            return result

    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def compute_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns:
        pd.DataFrame: columns BBM (middle), BBU (upper), BBL (lower), BBB, BBP.
    """
    if not _check_df(df, period):
        return pd.DataFrame(columns=["BBM", "BBU", "BBL", "BBB", "BBP"]) if _HAS_PANDAS else {}

    if _HAS_TA:
        result = ta.bbands(df[column], length=period, std=std_dev)
        if result is not None:
            cols = result.columns.tolist()
            rename = {}
            for c in cols:
                if "BBL" in c:
                    rename[c] = "BBL"
                elif "BBM" in c:
                    rename[c] = "BBM"
                elif "BBU" in c:
                    rename[c] = "BBU"
                elif "BBB" in c:
                    rename[c] = "BBB"
                elif "BBP" in c:
                    rename[c] = "BBP"
            return result.rename(columns=rename)

    mid = df[column].rolling(period).mean()
    std = df[column].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    bandwidth = (upper - lower) / mid.replace(0, float("nan"))
    pct_b = (df[column] - lower) / (upper - lower).replace(0, float("nan"))
    return pd.DataFrame(
        {
            "BBM": mid,
            "BBU": upper,
            "BBL": lower,
            "BBB": bandwidth,
            "BBP": pct_b,
        },
        index=df.index,
    )

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (intraday cumulative). Returns pd.Series."""
    if not _check_df(df) or "volume" not in df.columns:
        return pd.Series(dtype=float) if _HAS_PANDAS else []

    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, float("nan"))

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume. Returns pd.Series (running total)."""
    if not _check_df(df) or "volume" not in df.columns:
        return pd.Series(dtype=float) if _HAS_PANDAS else []

    if _HAS_TA:
        result = ta.obv(df["close"], df["volume"])
        if result is not None:
            return result

    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * df["volume"]).cumsum()

def compute_volume_zscore(df: pd.DataFrame, period: int = 30) -> pd.Series:
    """Rolling Z-score of volume. Returns pd.Series (float)."""
    if not _check_df(df) or "volume" not in df.columns:
        return pd.Series(dtype=float) if _HAS_PANDAS else []

    vol = df["volume"].astype(float)
    rolling_mean = vol.rolling(period, min_periods=max(1, period // 3)).mean()
    rolling_std = vol.rolling(period, min_periods=max(1, period // 3)).std().replace(0, 1)
    return (vol - rolling_mean) / rolling_std

def compute_all(df: pd.DataFrame, atr_period: int = 14) -> dict[str, pd.Series | pd.DataFrame]:
    """Compute a standard set of indicators and return as a dict."""
    return {
        "atr": compute_atr(df, atr_period),
        "adx": compute_adx(df),
        "ema20": compute_ema(df, 20),
        "ema50": compute_ema(df, 50),
        "sma20": compute_sma(df, 20),
        "rsi": compute_rsi(df),
        "bb": compute_bollinger(df),
        "vwap": compute_vwap(df),
        "obv": compute_obv(df),
        "vol_z": compute_volume_zscore(df),
    }
