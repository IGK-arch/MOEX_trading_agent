"""Единый источник истины для всех констант."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"
PHASES_DIR = ROOT_DIR / "docs" / "phases"
HISTORICAL_CSV_DIR = DATA_DIR / "historical_csv"
MORNING_PLANS_DIR = DATA_DIR / "morning_plans"

for _d in [LOGS_DIR, MODELS_DIR, PHASES_DIR, HISTORICAL_CSV_DIR, MORNING_PLANS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RUN_MODE: str = os.getenv("RUN_MODE", "paper")

TICKERS: list[str] = [
    "LKOH",
    "SBER",
    "ROSN",
    "GAZP",
    "VTBR",
    "YDEX",
    "PLZL",
    "T",
    "NVTK",
    "X5",
    "GMKN",
    "MGNT",
    "ALRS",
    "AFLT",
    "CHMF",
    "NLMK",
    "MOEX",
    "SNGSP",
    "MTSS",
    "PIKK",
]

COINTEGRATED_PAIRS: list[tuple[str, str]] = [
    ("PLZL", "AFLT"),
    ("LKOH", "MGNT"),
    ("T", "GMKN"),
    ("AFLT", "SNGSP"),
    ("YDEX", "SNGSP"),
    ("YDEX", "GMKN"),
    ("T", "CHMF"),
    ("YDEX", "PLZL"),
    ("GMKN", "SNGSP"),
    ("T", "SNGSP"),
    ("LKOH", "NVTK"),
    ("T", "ALRS"),
    ("T", "MGNT"),
    ("T", "NLMK"),
    ("MOEX", "MTSS"),
]

MAIN_SESSION_OPEN_MSK = (7, 0)
MAIN_SESSION_CLOSE_MSK = (23, 50)
EVENING_SESSION_OPEN_MSK = (7, 0)
EVENING_SESSION_CLOSE_MSK = (23, 50)
WEEKEND_SESSION_OPEN_MSK = (10, 0)
WEEKEND_SESSION_CLOSE_MSK = (19, 0)
WEEKEND_TRADING_ENABLED: bool = True
ARENAGO_COMMISSION_PCT: float = float(os.getenv("ARENAGO_COMMISSION_PCT", "0.0"))
ARENAGO_DAILY_TRADE_LIMIT: int = 1000

SL_MONITOR_INTERVAL_SEC: float = float(os.getenv("SL_MONITOR_INTERVAL_SEC", "5.0"))
HARD_STOP_LOSS_PCT: float = float(os.getenv("HARD_STOP_LOSS_PCT", "0.015"))
HARD_TIME_STOP_HOURS: float = float(os.getenv("HARD_TIME_STOP_HOURS", "12.0"))

FORCE_CLOSE_BEFORE_CLOSE_MIN: int = 30

HIGH_WR_TICKERS: frozenset[str] = frozenset({"SBER", "GAZP"})

ARENAGO_BASE_URL = "https://arenago.ru"
ARENAGO_BOT_NAME: str = os.getenv("ARENAGO_BOT_NAME", "404: Loss Not Found")
ARENAGO_URL_VARIANT: str = "v2"
ARENAGO_DAILY_TRADE_LIMIT = 1000
ARENAGO_DAILY_TRADE_SOFT_LIMIT = 800
ARENAGO_DAILY_TRADE_SLOWDOWN = 950
ARENAGO_DAILY_TRADE_ENTRY_HALT = 999

ISS_BASE_URL = "https://iss.moex.com/iss"
ISS_BOARD = "TQBR"

ALGOPACK_BASE_URL = "https://apim.moex.com/iss/datashop/algopack"
ALGOPACK_SEMAPHORE_LIMIT = 4

POLZA_BASE_URL = "https://polza.ai/api/v1"
POLZA_MODEL_REASONING = os.getenv("POLZA_MODEL_REASONING", "deepseek/deepseek-r1-0528")
POLZA_MODEL_REACTIVE = os.getenv("POLZA_MODEL_REACTIVE", "deepseek/deepseek-v4-flash")
POLZA_MODEL_FALLBACK = os.getenv("POLZA_MODEL_FALLBACK", "qwen/qwen3.5-plus-20260420")
POLZA_MODEL_SANCTIONS = os.getenv("POLZA_MODEL_SANCTIONS", "deepseek/deepseek-r1-0528")
POLZA_MODEL_DEEP_REASONING = os.getenv(
    "POLZA_MODEL_DEEP_REASONING", "deepseek/deepseek-r1-0528"
)
POLZA_MODEL_RUSSIAN_FLUENT = os.getenv(
    "POLZA_MODEL_RUSSIAN_FLUENT", "qwen/qwen3.5-plus-20260420"
)
POLZA_MODEL_CHEAP_REASONING = os.getenv(
    "POLZA_MODEL_CHEAP_REASONING", "deepseek/deepseek-r1-distill-llama-70b"
)
POLZA_MODEL_MORNING_BRIEF = os.getenv(
    "POLZA_MODEL_MORNING_BRIEF", "deepseek/deepseek-r1-distill-llama-70b"
)
POLZA_MODEL_EMBEDDING = os.getenv("POLZA_MODEL_EMBEDDING", "qwen/qwen3-embedding-8b")
POLZA_BUDGET_TOTAL_RUB = 12_000.0
POLZA_BUDGET_HARD_STOP_RUB = 11_500.0
POLZA_BUDGET_SOFT_LIMIT_RUB = 9_600.0
POLZA_DAILY_SOFT_LIMIT_RUB = 100.0

POLZA_REQUEST_TIMEOUT_SEC: float = float(os.getenv("POLZA_REQUEST_TIMEOUT_SEC", "30.0"))

DISABLE_LLM: bool = os.getenv("DISABLE_LLM", "0") == "1"

NEWS_LOCAL_SENTIMENT_ENABLED: bool = os.getenv("NEWS_LOCAL_SENTIMENT_ENABLED", "1") == "1"
NEWS_LOCAL_SENT_THRESHOLD: float = float(os.getenv("NEWS_LOCAL_SENT_THRESHOLD", "0.5"))
NEWS_LOCAL_SENT_MAG_MULT: float = float(os.getenv("NEWS_LOCAL_SENT_MAG_MULT", "0.6"))

NEWS_EVENT_TYPE_MAG_MULT: dict[str, float] = {
    "sanctions": 1.20,
    "macro": 0.90,
    "commodity": 0.75,
    "earnings": 0.75,
    "guidance": 0.85,
    "other": 0.50,
}

NEWS_TIME_OF_DAY_MULT: dict[str, float] = {
    "premarket": 0.40,
    "morning": 0.85,
    "midday": 1.20,
    "afternoon": 1.10,
    "afterhours": 0.70,
}

NEWS_EMBEDDINGS_ENABLED: bool = os.getenv("NEWS_EMBEDDINGS_ENABLED", "0") == "1"

LIVE_SIZING: bool = os.getenv("LIVE_SIZING", "1") == "1"

STRICT_MODE: bool = os.getenv("STRICT_MODE", "0") == "1"

RISK_PER_TRADE_TEST = 0.005
MAX_POSITION_PCT_TEST = 0.02

RISK_PER_TRADE_LIVE = 0.020
MAX_POSITION_PCT_LIVE = 0.05

RISK_PER_TRADE: float = RISK_PER_TRADE_LIVE if LIVE_SIZING else RISK_PER_TRADE_TEST
MAX_POSITION_PCT: float = MAX_POSITION_PCT_LIVE if LIVE_SIZING else MAX_POSITION_PCT_TEST

MAX_OPEN_POSITIONS = 5 if STRICT_MODE else 20
MAX_SECTOR_EXPOSURE_PCT = 0.30
MAX_PORTFOLIO_EXPOSURE_PCT = 0.80
MAX_CORR_BETWEEN_POSITIONS = 0.70

KELLY_FRACTION = 0.25

CIRCUIT_DAILY_LOSS_PCT = -0.015
CIRCUIT_MAX_DD_PCT = -0.08
CIRCUIT_LOSING_STREAK = 3
CIRCUIT_WINNING_STREAK = 5

SECTOR_MAP: dict[str, str] = {
    "SBER": "banks",
    "VTBR": "banks",
    "T": "banks",
    "GAZP": "oil_gas",
    "LKOH": "oil_gas",
    "ROSN": "oil_gas",
    "NVTK": "oil_gas",
    "SNGSP": "oil_gas",
    "GMKN": "metals",
    "NLMK": "metals",
    "CHMF": "metals",
    "PLZL": "metals",
    "ALRS": "metals",
    "MGNT": "retail",
    "X5": "retail",
    "PIKK": "real_estate",
    "MTSS": "telecom",
    "AFLT": "transport",
    "YDEX": "tech",
    "MOEX": "exchange",
}

DISPATCHER_CYCLE_SECS = 30
POLL_TIMEOUT_SECS = 5.0
POLL_TIMEOUT_SECONDS = POLL_TIMEOUT_SECS
DISPATCHER_CYCLE_SECONDS = 30.0
MIN_HOLDING_PERIOD_SECS = 3600

CONFLUENCE_MULTIPLIER_2_SOURCES = 1.5
CONFLUENCE_MULTIPLIER_3_PLUS = 2.0

if STRICT_MODE:
    TIER1_MIN_MAGNITUDE = 0.70
    TIER1_MIN_RR = 2.0
    TIER2_MIN_MAGNITUDE = 0.55
    TIER2_MIN_RR = 1.5
    TIER3_MIN_MAGNITUDE = 0.40
    TIER3_MIN_RR = 1.2
else:
    TIER1_MIN_MAGNITUDE = 0.40
    TIER1_MIN_RR = 1.2
    TIER2_MIN_MAGNITUDE = 0.25
    TIER2_MIN_RR = 0.7
    TIER3_MIN_MAGNITUDE = 0.99
    TIER3_MIN_RR = 0.99

if LIVE_SIZING:
    TIER1_SIZE_PCT = 0.035
    TIER2_SIZE_PCT = 0.022
    TIER3_SIZE_PCT = 0.008
else:
    TIER1_SIZE_PCT = 0.0050
    TIER2_SIZE_PCT = 0.0030
    TIER3_SIZE_PCT = 0.0009

ATR_PERIOD = 14
ADX_PERIOD = 14
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
EMA_FAST = 9
EMA_SLOW = 21
VWAP_RESET_HOUR_MSK = 10

PIVOT_ORDER = 5
PIVOT_MERGE_ATR_FACTOR = 0.5
CATBOOST_MIN_CONFIDENCE = 0.40

META_ENABLED: bool = os.getenv("META_ENABLED", "1") == "1"
META_MIN_PROBA: float = float(
    os.getenv(
        "META_MIN_PROBA",
        "0.50" if STRICT_MODE else "0.35",
    )
)
META_FALLBACK_CONFIDENCE: float = 0.5

META_MIN_PROBA_FLOOR: float = 0.45
META_MIN_PROBA_CEILING: float = 0.70

DASHBOARD_SNAPSHOT_ENABLED: bool = os.getenv("DASHBOARD_SNAPSHOT_ENABLED", "1") == "1"
DASHBOARD_SNAPSHOT_PATH = DATA_DIR / "dashboard_snapshot.html"
DASHBOARD_SNAPSHOT_INTERVAL_SEC: float = 300.0
METRICS_LIVE_PATH = DATA_DIR / "metrics_live.jsonl"
METRICS_LIVE_INTERVAL_SEC: float = 30.0
METRICS_SUMMARY_PATH = DATA_DIR / "metrics_summary.json"
DASHBOARD_MODE_PATH = DATA_DIR / "dashboard_mode.txt"

RECOVERY_STATE_PATH = DATA_DIR / "recovery_state.json"
RECOVERY_SAVE_INTERVAL_SEC: float = float(os.getenv("RECOVERY_SAVE_INTERVAL_SEC", "5.0"))
RECOVERY_STALE_THRESHOLD_SEC: float = 1800.0

MICROSTRUCTURE_GATES_ENABLED: bool = os.getenv("MICROSTRUCTURE_GATES_ENABLED", "1") == "1"

MTF_CONFLUENCE_ENABLED: bool = os.getenv("MTF_CONFLUENCE_ENABLED", "1") == "1"
MTF_CONFLUENCE_ADX_MIN: float = float(os.getenv("MTF_CONFLUENCE_ADX_MIN", "20.0"))

VPIN_BLOCK_THRESHOLD: float = 0.60
VPIN_N_BUCKETS: int = 50

KYLES_LAMBDA_BLOCK_PCT: float = 0.90
KYLES_LAMBDA_WINDOW: int = 20

OFI_OPPOSITION_THRESHOLD: float = 0.30
OFI_OPPOSITION_WEAKEN_MULT: float = 0.7
OFI_WINDOW_BARS: int = 5

PAIR_ADF_LOOKBACK_DAYS = 60
PAIR_ADF_PVALUE_THRESHOLD = 0.05
PAIR_Z_ENTRY_THRESHOLD = 1.5
PAIR_Z_EXIT_THRESHOLD = 0.3
PAIR_Z_STOP_THRESHOLD = 3.5
PAIR_COOLDOWN_HOURS = 24

PAIR_ADF_DECAY_DAYS: int = int(os.getenv("PAIR_ADF_DECAY_DAYS", "3"))

PAIR_PARAMS: dict[str, dict[str, float | str | int]] = {
    "PLZL_AFLT": {
        "z_entry": 1.5,
        "z_exit": 0.0,
        "max_hold_bars": 72,
        "sizing_mode": "equal",
        "z_stop": 3.5,
    },
    "YDEX_PLZL": {
        "z_entry": 1.8,
        "z_exit": 0.2,
        "max_hold_bars": 72,
        "sizing_mode": "equal",
        "z_stop": 3.5,
    },
    "LKOH_MGNT": {
        "z_entry": 1.5,
        "z_exit": 0.0,
        "max_hold_bars": 72,
        "sizing_mode": "equal",
        "z_stop": 3.5,
    },
    "T_GMKN": {
        "z_entry": 2.0,
        "z_exit": 0.3,
        "max_hold_bars": 24,
        "sizing_mode": "beta",
        "z_stop": 3.5,
    },
    "T_CHMF": {
        "z_entry": 2.5,
        "z_exit": 0.0,
        "max_hold_bars": 48,
        "sizing_mode": "beta",
        "z_stop": 3.5,
    },
    "T_MGNT": {
        "z_entry": 2.5,
        "z_exit": 0.3,
        "max_hold_bars": 24,
        "sizing_mode": "beta",
        "z_stop": 3.5,
    },
    "AFLT_SNGSP": {
        "z_entry": 2.0,
        "z_exit": 0.3,
        "max_hold_bars": 48,
        "sizing_mode": "equal",
        "z_stop": 3.5,
    },
}

PAIR_DEFAULT_PARAMS: dict[str, float | str | int] = {
    "z_entry": PAIR_Z_ENTRY_THRESHOLD,
    "z_exit": PAIR_Z_EXIT_THRESHOLD,
    "max_hold_bars": 48,
    "sizing_mode": "equal",
    "z_stop": PAIR_Z_STOP_THRESHOLD,
}

def get_pair_params(pair_key: str) -> dict[str, float | str | int]:
    """Return per-pair params, falling back to PAIR_DEFAULT_PARAMS."""
    return {**PAIR_DEFAULT_PARAMS, **PAIR_PARAMS.get(pair_key, {})}

PAIR_USE_KALMAN: bool = os.getenv("PAIR_USE_KALMAN", "0") == "1"
PAIR_KALMAN_Q_ALPHA: float = float(os.getenv("PAIR_KALMAN_Q_ALPHA", "1e-8"))
PAIR_KALMAN_Q_BETA: float = float(os.getenv("PAIR_KALMAN_Q_BETA", "1e-7"))
PAIR_KALMAN_ZSCORE_WINDOW: int = int(os.getenv("PAIR_KALMAN_ZSCORE_WINDOW", "200"))

ANOMALY_VOLUME_ZSCORE_THRESHOLD = 3.0
ANOMALY_OFI_THRESHOLD = 0.60
ANOMALY_ATR_MULT_SPIKE = 2.0
ANOMALY_VOLUME_MULT_SPIKE = 3.0
ANOMALY_REVERSION_MIN_MOVE_ATR = 2.0
ANOMALY_COOLDOWN_MINUTES = 45

OFI_PER_TICKER_THRESHOLDS: bool = os.getenv("OFI_PER_TICKER_THRESHOLDS", "0") == "1"
OFI_TICKER_OVERRIDES: dict[str, float] = {
    "SBER": 0.55,
    "GAZP": 0.55,
    "LKOH": 0.55,
    "ROSN": 0.55,
    "VTBR": 0.65,
}

def ofi_threshold_for_ticker(ticker: str) -> float:
    """Return per-ticker OFI threshold if flag is on; else the global default."""
    if OFI_PER_TICKER_THRESHOLDS and ticker in OFI_TICKER_OVERRIDES:
        return OFI_TICKER_OVERRIDES[ticker]
    return ANOMALY_OFI_THRESHOLD

DETECTOR_BLACKLIST: frozenset[str] = frozenset(
    {
        "inv_head_shoulders",
        "bull_flag",
        "falling_wedge",
        "rounding_bottom",
        "megaphone_bottom",
        "compression_breakout_up",
        "diamond_bottom",
        "vcp",
        "pivot_reversal_long",
        "cdl_doji",
        "cdl_shortline",
        "cdl_longleggeddoji",
        "cdl_spinningtop",
        "cdl_belthold",
        "cdl_longline",
        "cdl_rickshawman",
        "cdl_engulfing",
        "cdl_highwave",
        "cdl_closingmarubozu",
        "cdl_harami",
        "cdl_hikkake",
        "cdl_takuri",
        "cdl_dragonflydoji",
        "cdl_marubozu",
        "cdl_haramicross",
        "cdl_gravestonedoji",
        "cdl_hammer",
        "cdl_hangingman",
        "cdl_3outside",
        "cdl_matchinglow",
        "cdl_separatinglines",
        "cdl_advanceblock",
        "cdl_invertedhammer",
        "cdl_eveningstar",
        "cdl_3inside",
        "cdl_morningstar",
        "cdl_3linestrike",
        "cdl_morningdojistar",
        "cdl_identical3crows",
        "cdl_tristar",
        "cdl_gapsidesidewhite",
        "cdl_sticksandwich",
        "cdl_hikkakemod",
        "cdl_homingpigeon",
        "cdl_tasukigap",
        "cdl_thrusting",
        "cdl_abandonedbaby",
        "cdl_unique3river",
        "cdl_ladderbottom",
    }
)

DETECTOR_GOLDLIST: frozenset[str] = frozenset(
    {
        "rectangle_breakdown",
        "descending_triangle",
        "head_shoulders",
        "symmetric_triangle",
        "smc_sweep_low",
    }
)

DETECTOR_GOLD_SIZE_MULT: float = 1.30

CANDLE_WHITELIST: frozenset[str] = frozenset(
    {
        "XSIDEGAP3METHODS",
        "STALLEDPATTERN",
        "EVENINGDOJISTAR",
        "SHOOTINGSTAR",
        "DOJISTAR",
        "3WHITESOLDIERS",
        "DARKCLOUDCOVER",
        "PIERCING",
    }
)

CONFLUENCE_VOLUME_CHECK: bool = os.getenv("CONFLUENCE_VOLUME_CHECK", "1") == "1"
CONFLUENCE_VOLUME_MULTIPLIER: float = float(os.getenv("CONFLUENCE_VOLUME_MULTIPLIER", "1.3"))
CONFLUENCE_HMM_ALIGN: bool = os.getenv("CONFLUENCE_HMM_ALIGN", "1") == "1"
CONFLUENCE_TOD_FILTER: bool = os.getenv("CONFLUENCE_TOD_FILTER", "1") == "1"
CONFLUENCE_ATR_PCTILE: bool = os.getenv("CONFLUENCE_ATR_PCTILE", "1") == "1"
CONFLUENCE_ATR_PCT_LOW: float = float(os.getenv("CONFLUENCE_ATR_PCT_LOW", "20"))
CONFLUENCE_ATR_PCT_HIGH: float = float(os.getenv("CONFLUENCE_ATR_PCT_HIGH", "90"))

MTF_HIGHER_TF_MIN: int = int(os.getenv("MTF_HIGHER_TF_MIN", "60"))
MTF_LOWER_TF_MIN: int = int(os.getenv("MTF_LOWER_TF_MIN", "5"))
MTF_MIN_BARS_HIGHER: int = int(os.getenv("MTF_MIN_BARS_HIGHER", "30"))
MTF_MIN_BARS_LOWER: int = int(os.getenv("MTF_MIN_BARS_LOWER", "30"))

DETECTOR_TICKER_GOLDLIST: dict[str, frozenset[str]] = {
    "LKOH": frozenset({"bear_flag", "bull_pennant", "triple_bottom"}),
    "SBER": frozenset({"bear_flag", "triple_bottom"}),
    "ROSN": frozenset(
        {"bear_pennant", "double_top", "inside_bar_breakout", "megaphone_top", "triple_top"}
    ),
    "GAZP": frozenset({"ascending_triangle", "bull_pennant", "double_top", "triple_top"}),
    "VTBR": frozenset(
        {"ascending_triangle", "bb_squeeze_breakout", "double_bottom", "double_top", "triple_top"}
    ),
    "YDEX": frozenset(
        {
            "cdl_dojistar",
            "compression_breakout_down",
            "double_top",
            "megaphone_top",
            "rising_wedge",
            "rounding_top",
            "triple_top",
        }
    ),
    "PLZL": frozenset(
        {
            "bb_squeeze_breakout",
            "bear_pennant",
            "double_top",
            "inside_bar_breakout",
            "rising_wedge",
            "rounding_top",
        }
    ),
    "T": frozenset({"bear_flag", "bull_pennant", "three_black_crows_vol"}),
    "NVTK": frozenset({"cdl_shootingstar", "inside_bar_breakout", "megaphone_top"}),
    "X5": frozenset({"rising_wedge", "rounding_top"}),
    "GMKN": frozenset(
        {
            "ascending_triangle",
            "bear_flag",
            "double_top",
            "megaphone_top",
            "rising_wedge",
            "rounding_top",
        }
    ),
    "MGNT": frozenset(
        {
            "ascending_triangle",
            "bear_pennant",
            "compression_breakout_down",
            "double_top",
            "rounding_top",
        }
    ),
    "ALRS": frozenset(
        {
            "bear_flag",
            "bear_pennant",
            "compression_breakout_down",
            "double_top",
            "inside_bar_breakout",
            "triple_top",
        }
    ),
    "AFLT": frozenset(
        {"cdl_xsidegap3methods", "double_top", "megaphone_top", "triple_bottom", "triple_top"}
    ),
    "CHMF": frozenset(
        {
            "bear_flag",
            "bear_pennant",
            "double_top",
            "megaphone_top",
            "rising_wedge",
            "rounding_top",
            "triple_top",
        }
    ),
    "NLMK": frozenset(
        {"ascending_triangle", "bear_pennant", "double_top", "rounding_top", "triple_top"}
    ),
    "MOEX": frozenset(
        {
            "ascending_triangle",
            "bb_squeeze_breakout",
            "bear_flag",
            "bear_pennant",
            "bull_pennant",
            "double_top",
            "megaphone_top",
            "rounding_top",
            "triple_bottom",
            "triple_top",
        }
    ),
    "SNGSP": frozenset({"ascending_triangle"}),
    "MTSS": frozenset({"ascending_triangle", "triple_bottom"}),
    "PIKK": frozenset({"rectangle_breakout_up", "triple_bottom"}),
}

DETECTOR_TICKER_BLACKLIST: dict[str, frozenset[str]] = {
    "LKOH": frozenset(
        {
            "ascending_triangle",
            "bear_pennant",
            "cdl_dojistar",
            "inside_bar_breakout",
            "megaphone_top",
            "rising_wedge",
            "rounding_top",
        }
    ),
    "SBER": frozenset(
        {
            "ascending_triangle",
            "bb_squeeze_breakout",
            "bear_pennant",
            "bull_pennant",
            "cdl_dojistar",
            "compression_breakout_down",
            "descending_triangle",
            "double_bottom",
            "double_top",
        }
    ),
    "ROSN": frozenset({"bear_flag", "bull_pennant", "rising_wedge"}),
    "GAZP": frozenset(
        {
            "bear_flag",
            "cdl_dojistar",
            "compression_breakout_down",
            "double_bottom",
            "megaphone_top",
            "rounding_top",
            "triple_bottom",
        }
    ),
    "VTBR": frozenset(
        {"bull_pennant", "cdl_dojistar", "cdl_shootingstar", "compression_breakout_down"}
    ),
    "YDEX": frozenset({"bear_flag", "double_bottom"}),
    "PLZL": frozenset({"bear_flag", "bull_pennant"}),
    "T": frozenset({"cdl_dojistar", "double_bottom"}),
    "NVTK": frozenset(
        {
            "bear_flag",
            "bear_pennant",
            "bull_pennant",
            "cdl_stalledpattern",
            "double_bottom",
            "rising_wedge",
            "triple_bottom",
            "triple_top",
        }
    ),
    "X5": frozenset(
        {
            "bear_flag",
            "compression_breakout_down",
            "descending_triangle",
            "double_top",
            "head_shoulders",
            "pivot_reversal_short",
            "rectangle_breakdown",
            "rectangle_breakout_up",
            "triple_top",
        }
    ),
    "GMKN": frozenset({"triple_bottom"}),
    "MGNT": frozenset({"double_bottom"}),
    "ALRS": frozenset({"bull_pennant", "cdl_dojistar", "double_bottom", "triple_bottom"}),
    "AFLT": frozenset({"bb_squeeze_breakout", "bear_flag", "cdl_dojistar"}),
    "CHMF": frozenset(
        {
            "bb_squeeze_breakout",
            "cdl_dojistar",
            "cdl_eveningdojistar",
            "double_bottom",
            "triple_bottom",
        }
    ),
    "NLMK": frozenset({"cdl_dojistar", "double_bottom"}),
    "MOEX": frozenset({"cdl_dojistar", "cdl_shootingstar"}),
    "SNGSP": frozenset(
        {
            "bb_squeeze_breakout",
            "bear_flag",
            "bear_pennant",
            "cdl_dojistar",
            "cdl_shootingstar",
            "compression_breakout_down",
            "descending_triangle",
            "inside_bar_breakout",
            "triple_bottom",
            "triple_top",
        }
    ),
    "MTSS": frozenset({"bear_pennant", "cdl_dojistar", "rounding_top"}),
    "PIKK": frozenset(
        {
            "bear_pennant",
            "cdl_dojistar",
            "compression_breakout_down",
            "megaphone_top",
            "rising_wedge",
            "rounding_top",
        }
    ),
}

DETECTOR_TOD_GOLDLIST: dict[str, frozenset[str]] = {
    "midday": frozenset({"ascending_triangle", "cdl_shootingstar"}),
    "afternoon": frozenset(
        {
            "bear_flag",
            "bear_pennant",
            "double_top",
            "inside_bar_breakout",
            "megaphone_top",
            "pivot_reversal_short",
            "rising_wedge",
            "three_black_crows_vol",
        }
    ),
}

DETECTOR_TOD_BLACKLIST: dict[str, frozenset[str]] = {
    "morning": frozenset(
        {
            "bb_squeeze_breakout",
            "bull_pennant",
            "compression_breakout_down",
            "double_bottom",
            "head_shoulders",
            "rectangle_breakdown",
            "rectangle_breakout_up",
            "triple_top",
        }
    ),
    "midday": frozenset({"cdl_eveningdojistar", "double_bottom", "triple_bottom"}),
    "afternoon": frozenset({"cdl_dojistar", "double_bottom"}),
}

DETECTOR_REGIME_GOLDLIST: dict[str, frozenset[str]] = {
    "mean_reverting": frozenset({"bear_flag", "bear_pennant", "rounding_top"}),
    "crisis": frozenset({"double_bottom"}),
}

DETECTOR_REGIME_BLACKLIST: dict[str, frozenset[str]] = {
    "mean_reverting": frozenset(
        {
            "ascending_triangle",
            "bb_squeeze_breakout",
            "cdl_dojistar",
            "cdl_eveningdojistar",
            "rectangle_breakout_up",
            "triple_top",
        }
    ),
    "trending": frozenset({"cdl_3whitesoldiers"}),
}

DETECTOR_COND_GOLD_MULT: float = 1.15
DETECTOR_COND_GOLD_MAX_STACK: float = 1.45

PER_TICKER_POLICY: dict[str, str] = {
    "SBER": "GOLD",
    "GAZP": "GOLD",
    "CHMF": "GOLD",
    "PIKK": "WHITELIST_ONLY",
    "LKOH": "WHITELIST_ONLY",
    "ROSN": "WHITELIST_ONLY",
    "VTBR": "WHITELIST_ONLY",
    "NVTK": "WHITELIST_ONLY",
    "YDEX": "DISABLED",
    "PLZL": "DISABLED",
    "T": "DISABLED",
    "X5": "DISABLED",
    "GMKN": "DISABLED",
    "MGNT": "GOLD",
    "ALRS": "DISABLED",
    "AFLT": "DISABLED",
    "NLMK": "DISABLED",
    "MOEX": "DISABLED",
    "SNGSP": "GOLD",
    "MTSS": "DISABLED",
}

WR_70_WHITELIST: dict[tuple[str, str, str, str], dict[str, float]] = {
    ("SBER", "bear_flag", "1.0", "3.0"): {"wr": 0.857, "pf": 13.49, "n": 14},
    ("SBER", "megaphone_top", "0.75", "1.0"): {"wr": 0.944, "pf": 9.12, "n": 18},
    ("SBER", "candle_hammer", "any", "any"): {"wr": 0.769, "pf": 4.46, "n": 26},
    ("SBER", "bb_squeeze_breakout", "any", "any"): {"wr": 0.765, "pf": 2.80, "n": 17},
    ("GAZP", "falling_wedge", "0.75", "1.0"): {"wr": 0.846, "pf": 15.48, "n": 13},
    ("GAZP", "research_family", "0.75", "1.0"): {"wr": 0.925, "pf": 10.09, "n": 40},
    ("GAZP", "bull_flag", "any", "any"): {"wr": 0.700, "pf": 10.30, "n": 15},
    ("GAZP", "rectangle_breakdown", "any", "any"): {"wr": 1.000, "pf": 9.99, "n": 10},
    ("CHMF", "research_family", "0.75", "1.5"): {"wr": 0.758, "pf": 3.71, "n": 33},
    ("PIKK", "candle_hammer", "any", "any"): {"wr": 0.769, "pf": 4.46, "n": 13},
}

DETECTOR_FAMILY_MAP: dict[str, str] = {
    "bear_flag": "continuation",
    "bull_flag": "continuation",
    "megaphone_top": "reversal",
    "falling_wedge": "continuation",
    "rectangle_breakdown": "continuation",
    "candle_hammer": "candle",
    "bb_squeeze_breakout": "research",
    "inside_bar_breakout": "research",
    "three_white_soldiers_vol": "research",
    "pivot_reversal_short": "research",
    "vcp": "research",
    "pivot_reversal_long": "research",
}

WR_70_TIER_A_MIN_WR: float = 0.85
WR_70_TIER_B_MIN_WR: float = 0.75
WR_70_TIER_C_MIN_WR: float = 0.70

if LIVE_SIZING:
    WR_70_TIER_A_SIZE_PCT: float = 0.040
    WR_70_TIER_B_SIZE_PCT: float = 0.030
    WR_70_TIER_C_SIZE_PCT: float = 0.020
else:
    WR_70_TIER_A_SIZE_PCT = 0.0050
    WR_70_TIER_B_SIZE_PCT = 0.0030
    WR_70_TIER_C_SIZE_PCT = 0.0015

def _wr70_tier_size_for_wr(wr: float) -> float:
    """Map a verified historical WR to the matching WR_70 tier size pct."""
    if wr >= WR_70_TIER_A_MIN_WR:
        return WR_70_TIER_A_SIZE_PCT
    if wr >= WR_70_TIER_B_MIN_WR:
        return WR_70_TIER_B_SIZE_PCT
    if wr >= WR_70_TIER_C_MIN_WR:
        return WR_70_TIER_C_SIZE_PCT
    return 0.0

def wr70_tier_size_pct(ticker: str, detector: str | None) -> float | None:
    """Return the WR-tier sizing pct for a (ticker, detector) combo if it's
    in WR_70_WHITELIST, else None (caller falls back to plain tier_size_pct).

    Matching rule:
      - Direct (ticker, detector_name, *, *) match wins first.
      - Otherwise (ticker, detector_family, *, *) match via DETECTOR_FAMILY_MAP.
      - Among multiple SL/RR bucket rows, the one with the HIGHEST WR wins
        so the sizing is anchored on the strongest historical edge.
      - Returns None if no row matches; the caller MUST default to
        tier_size_pct(decision.tier) so behaviour stays backward-compatible.
    """
    if not ticker or not detector:
        return None
    ticker_u = ticker.upper()
    family = DETECTOR_FAMILY_MAP.get(detector)
    family_alt = f"{family}_family" if family else None
    best_wr: float | None = None
    for (t, d_or_f, _sl, _rr), meta in WR_70_WHITELIST.items():
        if t != ticker_u:
            continue
        if d_or_f != detector and d_or_f != family and d_or_f != family_alt:
            continue
        wr = float(meta.get("wr", 0.0))
        if best_wr is None or wr > best_wr:
            best_wr = wr
    if best_wr is None:
        return None
    pct = _wr70_tier_size_for_wr(best_wr)
    if pct <= 0:
        return None
    return pct

def is_signal_allowed(ticker: str, detector: str | None = None) -> bool:
    """v0.4.0 — capital-preservation gate. Returns False if ticker is DISABLED
    or if WHITELIST_ONLY ticker tries to emit a non-whitelisted detector.

    GOLD tickers can emit any signal (broad edge).
    WHITELIST_ONLY tickers can only emit signals from WR_70_WHITELIST.
    DISABLED tickers emit NOTHING.

    If ticker not in PER_TICKER_POLICY: treat as DISABLED (capital safety).
    """
    policy = PER_TICKER_POLICY.get(ticker, "DISABLED")
    if policy == "DISABLED":
        return False
    if policy == "GOLD":
        return True
    if policy == "WHITELIST_ONLY":
        if not detector:
            return False
        family = DETECTOR_FAMILY_MAP.get(detector, detector)
        for (t, d_or_f, _sl, _rr), _meta in WR_70_WHITELIST.items():
            if t != ticker:
                continue
            if d_or_f in (detector, family):
                return True
        return False
    return False

WR_90_WHITELIST: dict[tuple[str, str, str, str], dict[str, float]] = {
    ("SBER", "megaphone_top", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 0.9655,
        "pf": 11.75,
        "side": "SELL",
        "n_trades": 29,
    },
    ("SBER", "bear_flag", "1.0", "1.0"): {
        "tier_pct": 0.05,
        "wr": 0.9524,
        "pf": 13.68,
        "side": "SELL",
        "n_trades": 21,
    },
    ("GAZP", "rounding_bottom", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 0.9659,
        "pf": 9.71,
        "side": "BUY",
        "n_trades": 88,
    },
    ("GAZP", "rising_wedge", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 0.9630,
        "pf": 11.76,
        "side": "SELL",
        "n_trades": 27,
    },
    ("LKOH", "head_shoulders", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "SELL",
        "n_trades": 18,
    },
    ("LKOH", "bear_flag", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 0.9667,
        "pf": 18.74,
        "side": "SELL",
        "n_trades": 30,
    },
    ("ROSN", "head_shoulders", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 0.9565,
        "pf": 5.26,
        "side": "SELL",
        "n_trades": 23,
    },
    ("ROSN", "bear_flag", "1.5", "0.7"): {
        "tier_pct": 0.04,
        "wr": 0.9200,
        "pf": 9.52,
        "side": "SELL",
        "n_trades": 25,
    },
    ("ROSN", "bear_flag", "2.0", "0.5"): {
        "tier_pct": 0.04,
        "wr": 0.9200,
        "pf": 7.98,
        "side": "SELL",
        "n_trades": 25,
    },
    ("VTBR", "double_bottom", "2.5", "0.3"): {
        "tier_pct": 0.05,
        "wr": 0.9800,
        "pf": 13.42,
        "side": "BUY",
        "n_trades": 50,
    },
    ("GMKN", "megaphone_bottom", "2.2", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "BUY",
        "n_trades": 14,
    },
    ("PLZL", "falling_wedge", "2.5", "0.3"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "BUY",
        "n_trades": 12,
    },
    ("NLMK", "bear_pennant", "2.2", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "SELL",
        "n_trades": 13,
    },
    ("CHMF", "bull_flag", "2.1", "0.4"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "BUY",
        "n_trades": 12,
    },
    ("ALRS", "double_top", "2.1", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "SELL",
        "n_trades": 17,
    },
    ("MGNT", "bear_pennant", "2.5", "1.0"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "SELL",
        "n_trades": 13,
    },
    ("X5", "falling_wedge", "2.5", "0.4"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "BUY",
        "n_trades": 15,
    },
    ("AFLT", "megaphone_top", "2.1", "0.7"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "SELL",
        "n_trades": 12,
    },
    ("T", "bull_pennant", "2.5", "1.4"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "BUY",
        "n_trades": 12,
    },
    ("PIKK", "rounding_bottom", "2.4", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 999.00,
        "side": "BUY",
        "n_trades": 33,
    },
    ("SNGSP", "vcp", "1.0", "1.0"): {
        "tier_pct": 0.04,
        "wr": 0.9412,
        "pf": 13.42,
        "side": "BUY",
        "n_trades": 17,
    },
    ("SNGSP", "vcp", "1.5", "0.7"): {
        "tier_pct": 0.04,
        "wr": 0.9412,
        "pf": 9.37,
        "side": "BUY",
        "n_trades": 17,
    },
    ("NVTK", "ascending_triangle", "1.5", "0.7"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 17,
    },
    ("NVTK", "ascending_triangle", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 17,
    },
    ("MOEX", "triple_bottom", "1.5", "0.7"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 15,
    },
    ("MOEX", "triple_bottom", "2.0", "0.5"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 15,
    },
    ("MTSS", "ascending_triangle", "1.0", "1.0"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 14,
    },
    ("MTSS", "ascending_triangle", "1.5", "0.7"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 14,
    },
    ("YDEX", "inv_head_shoulders", "1.0", "1.0"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 17,
    },
    ("YDEX", "inv_head_shoulders", "1.5", "0.7"): {
        "tier_pct": 0.05,
        "wr": 1.0000,
        "pf": 9999.00,
        "side": "BUY",
        "n_trades": 17,
    },
}

WR_90_TIER_S_MIN_WR: float = 0.95
WR_90_TIER_A_MIN_WR: float = 0.92
WR_90_TIER_B_MIN_WR: float = 0.90

if LIVE_SIZING:
    WR_90_TIER_S_SIZE_PCT: float = 0.05
    WR_90_TIER_A_SIZE_PCT: float = 0.04
    WR_90_TIER_B_SIZE_PCT: float = 0.03
else:
    WR_90_TIER_S_SIZE_PCT = 0.0060
    WR_90_TIER_A_SIZE_PCT = 0.0050
    WR_90_TIER_B_SIZE_PCT = 0.0040

def _wr90_tier_size_for_wr(wr: float) -> float:
    """Map a verified historical WR to the matching WR_90 tier size pct."""
    if wr >= WR_90_TIER_S_MIN_WR:
        return WR_90_TIER_S_SIZE_PCT
    if wr >= WR_90_TIER_A_MIN_WR:
        return WR_90_TIER_A_SIZE_PCT
    if wr >= WR_90_TIER_B_MIN_WR:
        return WR_90_TIER_B_SIZE_PCT
    return 0.0

def wr90_tier_size_pct(ticker: str, detector: str | None) -> float | None:
    """Return the WR_90-tier sizing pct for a (ticker, detector) combo if it's
    in WR_90_WHITELIST, else None (caller falls back to wr70/plain tier).

    Picks the row with the HIGHEST WR among matches so a signal that
    coincides with a top-edge combo gets the highest tier.
    """
    if not ticker or not detector:
        return None
    ticker_u = ticker.upper()
    family = DETECTOR_FAMILY_MAP.get(detector)
    family_alt = f"{family}_family" if family else None
    best_wr: float | None = None
    for (t, d_or_f, _sl, _rr), meta in WR_90_WHITELIST.items():
        if t != ticker_u:
            continue
        if d_or_f != detector and d_or_f != family and d_or_f != family_alt:
            continue
        wr = float(meta.get("wr", 0.0))
        if best_wr is None or wr > best_wr:
            best_wr = wr
    if best_wr is None:
        return None
    pct = _wr90_tier_size_for_wr(best_wr)
    if pct <= 0:
        return None
    return pct

PER_TICKER_DIRECTION_BIAS: dict[str, dict] = {
    "SBER": {
        "bias": "SELL",
        "strength": "VERY_STRONG",
        "best_detector": "megaphone_top",
        "wr": 0.9655,
    },
    "GAZP": {
        "bias": "BUY",
        "strength": "VERY_STRONG",
        "best_detector": "rounding_bottom",
        "wr": 0.9659,
    },
    "LKOH": {
        "bias": "SELL",
        "strength": "VERY_STRONG",
        "best_detector": "head_shoulders",
        "wr": 1.0000,
    },
    "ROSN": {
        "bias": "SELL",
        "strength": "VERY_STRONG",
        "best_detector": "head_shoulders",
        "wr": 0.9565,
    },
    "VTBR": {
        "bias": "BUY",
        "strength": "VERY_STRONG",
        "best_detector": "double_bottom",
        "wr": 0.9800,
    },
    "GMKN": {
        "bias": "BUY",
        "strength": "STRONG",
        "best_detector": "megaphone_bottom",
        "wr": 1.0000,
    },
    "PLZL": {"bias": "BUY", "strength": "STRONG", "best_detector": "falling_wedge", "wr": 1.0000},
    "NLMK": {"bias": "SELL", "strength": "STRONG", "best_detector": "bear_pennant", "wr": 1.0000},
    "CHMF": {"bias": "BUY", "strength": "STRONG", "best_detector": "bull_flag", "wr": 1.0000},
    "ALRS": {"bias": "SELL", "strength": "STRONG", "best_detector": "double_top", "wr": 1.0000},
    "MGNT": {"bias": "SELL", "strength": "STRONG", "best_detector": "bear_pennant", "wr": 1.0000},
    "X5": {"bias": "BUY", "strength": "STRONG", "best_detector": "falling_wedge", "wr": 1.0000},
    "AFLT": {"bias": "SELL", "strength": "STRONG", "best_detector": "megaphone_top", "wr": 1.0000},
    "T": {"bias": "BUY", "strength": "STRONG", "best_detector": "bull_pennant", "wr": 1.0000},
    "PIKK": {"bias": "BUY", "strength": "STRONG", "best_detector": "rounding_bottom", "wr": 1.0000},
    "SNGSP": {"bias": "BUY", "strength": "STRONG", "best_detector": "vcp", "wr": 0.9412},
    "NVTK": {
        "bias": "SELL",
        "strength": "VERY_STRONG",
        "best_detector": "ascending_triangle",
        "wr": 1.0000,
    },
    "MOEX": {
        "bias": "BUY",
        "strength": "VERY_STRONG",
        "best_detector": "triple_bottom",
        "wr": 1.0000,
    },
    "MTSS": {
        "bias": "BUY",
        "strength": "VERY_STRONG",
        "best_detector": "ascending_triangle",
        "wr": 1.0000,
    },
    "YDEX": {
        "bias": "SELL",
        "strength": "VERY_STRONG",
        "best_detector": "inv_head_shoulders",
        "wr": 1.0000,
    },
}

PER_TICKER_BIAS_ALIGN_MULT: float = 1.20
PER_TICKER_BIAS_COUNTER_MULT: float = 0.70

def get_ticker_bias(ticker: str) -> str | None:
    """Return 'BUY' / 'SELL' if ticker has a STRONG (or VERY_STRONG) bias,
    else None. Used as the gate in risk_manager to apply soft multipliers.

    Args:
        ticker: instrument code (case-insensitive)
    Returns:
        'BUY' | 'SELL' | None
    """
    if not ticker:
        return None
    entry = PER_TICKER_DIRECTION_BIAS.get(ticker.upper())
    if not entry:
        return None
    if entry.get("strength") not in ("STRONG", "VERY_STRONG"):
        return None
    bias = entry.get("bias")
    if bias in ("BUY", "SELL"):
        return bias
    return None

NEWS_STALE_CHECK_DELAY_POS_SECS = 5
NEWS_STALE_CHECK_DELAY_NEG_SECS = 10
NEWS_MICROSTRUCTURE_CONFIRM_MIN = 3
MATERIAL_FILTER_MIN_KEYWORDS = 1
INGESTION_BUS_MAX_SIZE = 1000
PROMPT_CACHE_TTL_SECONDS = 14400
PROMPT_CACHE_TTL_SECONDS_LONG = int(os.getenv("PROMPT_CACHE_TTL_SECONDS_LONG", "86400"))

NEWS_LLM_BATCH_ENABLED: bool = os.getenv("NEWS_LLM_BATCH_ENABLED", "0") == "1"
NEWS_LLM_BATCH_MAX_SIZE: int = int(os.getenv("NEWS_LLM_BATCH_MAX_SIZE", "5"))
NEWS_LLM_BATCH_WAIT_MS: int = int(os.getenv("NEWS_LLM_BATCH_WAIT_MS", "30000"))

NEWS_PROMPT_VERSION: str = os.getenv("NEWS_PROMPT_VERSION", "v2_dkcot")
SANCTIONS_PUSH_MODE: bool = os.getenv("SANCTIONS_PUSH_MODE", "1") == "1"

TURNOVER_TARGET_14D_RUB: float = float(os.getenv("TURNOVER_TARGET_14D_RUB", "30000000"))
TURNOVER_WARNING_DAY_7_RUB: float = TURNOVER_TARGET_14D_RUB * 0.40
TURNOVER_LOW_THRESHOLD_RUB: float = TURNOVER_TARGET_14D_RUB * 0.50

MIN_NOTIONAL_RUB: float = float(os.getenv("MIN_NOTIONAL_RUB", "5000"))

TURNOVER_TARGET_DAILY_RUB: float = TURNOVER_TARGET_14D_RUB / 14.0

TURNOVER_ESCALATION_LEVEL: int = 0
TURNOVER_ESCALATION_DAY_7_THRESHOLD_RUB: float = 4_000_000.0
TURNOVER_ESCALATION_DAY_10_THRESHOLD_RUB: float = 7_000_000.0
TURNOVER_ESCALATION_LEVEL_2_SIZE_MULT: float = 0.5

TURNOVER_ESCALATION_WHITELIST: tuple[str, ...] = (
    "LKOH",
    "VTBR",
    "MTSS",
    "MOEX",
)

TICKER_AVG_DAILY_VOLUME_RUB: dict[str, float] = {
    "SBER": 25_000_000_000.0,
    "GAZP": 8_500_000_000.0,
    "CHMF": 850_000_000.0,
    "PIKK": 450_000_000.0,
    "LKOH": 4_500_000_000.0,
    "VTBR": 2_200_000_000.0,
    "MTSS": 700_000_000.0,
    "MOEX": 900_000_000.0,
}

TURNOVER_INTRADAY_BOOST_CUTOFF_HOUR_MSK: int = 14
TURNOVER_INTRADAY_BOOST_MIN_TRADES: int = 5
TURNOVER_INTRADAY_BOOST_DELTA: float = 0.05
TURNOVER_INTRADAY_BOOST_ABS_FLOOR: float = 0.25

def get_size_mult_for_escalation() -> float:
    """Return current sizing multiplier given TURNOVER_ESCALATION_LEVEL."""
    if TURNOVER_ESCALATION_LEVEL >= 2:
        return TURNOVER_ESCALATION_LEVEL_2_SIZE_MULT
    return 1.0

def get_volume_weighted_size_mult(ticker: str) -> float:
    """Per-ticker volume-floor sizing. Tickers with higher average daily
    rouble volume can absorb larger fills with less slippage → larger mult
    (capped at 1.5×). Low-volume tickers get 1.0× (no penalty, no boost).
    SBER is the benchmark. Returns 1.0 when ticker not in
    TICKER_AVG_DAILY_VOLUME_RUB.
    """
    avg = TICKER_AVG_DAILY_VOLUME_RUB.get(ticker)
    if avg is None or avg <= 0:
        return 1.0
    bench = TICKER_AVG_DAILY_VOLUME_RUB.get("SBER", avg)
    import math

    log_ratio = math.log10(max(avg / bench, 1e-6))
    boost = 1.0 + 0.5 * max(0.0, min(1.0, (log_ratio + 1.5) / 1.5))
    return round(boost, 3)

def apply_turnover_escalation(level: int) -> dict[str, str]:
    """Mutate PER_TICKER_POLICY in-place per escalation level. Returns the
    updated policy dict. Idempotent — safe to call repeatedly.

    Level 0 — restore the v0.14.0 capital-preservation policy.
    Level 1 — promote TURNOVER_ESCALATION_WHITELIST tickers to WHITELIST_ONLY.
    Level 2 — also enable every remaining DISABLED ticker (WHITELIST_ONLY);
              caller is responsible for halving sizing via
              get_size_mult_for_escalation().
    """
    global TURNOVER_ESCALATION_LEVEL
    baseline: dict[str, str] = {
        "SBER": "GOLD",
        "GAZP": "GOLD",
        "CHMF": "GOLD",
        "PIKK": "WHITELIST_ONLY",
        "LKOH": "DISABLED",
        "ROSN": "DISABLED",
        "VTBR": "DISABLED",
        "NVTK": "DISABLED",
        "YDEX": "DISABLED",
        "PLZL": "DISABLED",
        "T": "DISABLED",
        "X5": "DISABLED",
        "GMKN": "DISABLED",
        "MGNT": "DISABLED",
        "ALRS": "DISABLED",
        "AFLT": "DISABLED",
        "NLMK": "DISABLED",
        "MOEX": "DISABLED",
        "SNGSP": "DISABLED",
        "MTSS": "DISABLED",
    }
    PER_TICKER_POLICY.clear()
    PER_TICKER_POLICY.update(baseline)
    if level >= 1:
        for tk in TURNOVER_ESCALATION_WHITELIST:
            if PER_TICKER_POLICY.get(tk) == "DISABLED":
                PER_TICKER_POLICY[tk] = "WHITELIST_ONLY"
    if level >= 2:
        for tk, pol in list(PER_TICKER_POLICY.items()):
            if pol == "DISABLED":
                PER_TICKER_POLICY[tk] = "WHITELIST_ONLY"
    TURNOVER_ESCALATION_LEVEL = max(0, min(2, int(level)))
    return PER_TICKER_POLICY

SCHEDULE_MORNING_BRIEF_MSK = "09:30"
SCHEDULE_EVENING_REFLECTION_MSK = "19:00"
SCHEDULE_MEMORY_DECAY_MSK = "00:30"
SCHEDULE_PAIR_REFIT_MSK = "09:15"
SCHEDULE_TURNOVER_CHECK_MSK = "18:45"
SCHEDULE_DATA_FEEDS_MSK = "06:00"
SCHEDULE_DAILY_CIRCUIT_BREAKER_RESET_MSK = "00:05"
SCHEDULE_EVENING_PIPELINE_MSK = "19:10"
SCHEDULE_WEEKLY_FULL_RETRAIN_MSK = "03:00"
SCHEDULE_WEEKLY_FULL_RETRAIN_DAY_OF_WEEK = "sun"

REFLEXIVE_CONTROL_ENABLED: bool = os.getenv("REFLEXIVE_CONTROL_ENABLED", "1") == "1"
REFLEXIVE_MIN_TRADES: int = int(os.getenv("REFLEXIVE_MIN_TRADES", "3"))
REFLEXIVE_MIN_CONFIDENCE: float = float(os.getenv("REFLEXIVE_MIN_CONFIDENCE", "0.65"))

MAX_CONCURRENT_POLLS = 4
HTTP_KEEPALIVE_CONNECTIONS = 20
HTTP_MAX_CONNECTIONS = 100
HTTP_TIMEOUT = 10.0
HTTP_RETRY_MAX = 3
HTTP_RETRY_BACKOFF_BASE = 0.5

ADAPTIVE_CAUTIOUS_DD_PCT: float = 0.003
ADAPTIVE_CAUTIOUS_LOSING_STREAK: int = 2
ADAPTIVE_CAUTIOUS_DAILY_PNL_PCT: float = -0.003

ADAPTIVE_DEFENSIVE_DD_PCT: float = 0.015
ADAPTIVE_DEFENSIVE_LOSING_STREAK: int = 3
ADAPTIVE_DEFENSIVE_DAILY_PNL_PCT: float = -0.008

ADAPTIVE_CRISIS_DD_PCT: float = 0.06
ADAPTIVE_CRISIS_LOSING_STREAK: int = 5
ADAPTIVE_CRISIS_DAILY_PNL_PCT: float = -0.025

DISABLE_REGIME_DD_CHECK: bool = os.getenv("DISABLE_REGIME_DD_CHECK", "1") == "1"

EQUITY_SANITY_FLOOR_FRACTION: float = float(
    os.getenv("EQUITY_SANITY_FLOOR_FRACTION", "0.7")
)

HARD_TAKE_PROFIT_PCT: float = float(os.getenv("HARD_TAKE_PROFIT_PCT", "0.06"))
HARD_TAKE_PROFIT_AGED_PCT: float = float(
    os.getenv("HARD_TAKE_PROFIT_AGED_PCT", "0.02")
)
HARD_TP_AGED_HOURS: float = float(os.getenv("HARD_TP_AGED_HOURS", "8.0"))

COUNTER_BIAS_GUARD_ENABLED: bool = (
    os.getenv("COUNTER_BIAS_GUARD_ENABLED", "1") == "1"
)

MAX_TICKER_NOTIONAL_PCT_CUMULATIVE: float = float(
    os.getenv("MAX_TICKER_NOTIONAL_PCT_CUMULATIVE", "0.08")
)

PRE_LONG_MOMENTUM_GUARD_ENABLED: bool = (
    os.getenv("PRE_LONG_MOMENTUM_GUARD_ENABLED", "1") == "1"
)
PRE_LONG_MIN_MOMENTUM_PCT: float = float(
    os.getenv("PRE_LONG_MIN_MOMENTUM_PCT", "-0.015")
)

DRAWDOWN_ALERT_PCT: float = float(os.getenv("DRAWDOWN_ALERT_PCT", "-0.01"))

FORCE_REGIME_NAME: str = os.getenv("FORCE_REGIME_NAME", "NORMAL").strip().upper()

ADAPTIVE_NORMAL_SIZE_MULT: float = 1.0
ADAPTIVE_CAUTIOUS_SIZE_MULT: float = 0.7
ADAPTIVE_DEFENSIVE_SIZE_MULT: float = 0.4
ADAPTIVE_CRISIS_SIZE_MULT: float = 0.25

ADAPTIVE_NORMAL_HARD_SL_PCT: float | None = None
ADAPTIVE_CAUTIOUS_HARD_SL_PCT: float | None = 0.012
ADAPTIVE_DEFENSIVE_HARD_SL_PCT: float | None = 0.010
ADAPTIVE_CRISIS_HARD_SL_PCT: float | None = 0.005

ADAPTIVE_NORMAL_HARD_TP_PCT: float | None = None
ADAPTIVE_CAUTIOUS_HARD_TP_PCT: float | None = 0.012
ADAPTIVE_DEFENSIVE_HARD_TP_PCT: float | None = 0.008
ADAPTIVE_CRISIS_HARD_TP_PCT: float | None = 0.005

ENTRY_BLACKOUT_AFTER_OPEN_MIN: int = 5
CONFLUENCE_MIN_SOURCES: int = 1

RAG_CONSENSUS_ENABLED: bool = os.getenv("RAG_CONSENSUS_ENABLED", "1") == "1"
RAG_PERSIST_DIR = DATA_DIR / "rag"
RAG_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
RAG_LLM_BACKEND: str = os.getenv("RAG_LLM_BACKEND", "polza")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_MAX_AGE_HOURS: int = int(os.getenv("RAG_MAX_AGE_HOURS", "48"))
RAG_EMBED_MODEL: str = os.getenv("RAG_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
RAG_PRUNE_HOURS: int = int(os.getenv("RAG_PRUNE_HOURS", "168"))
SCHEDULE_MORNING_CONSENSUS_MSK: str = os.getenv("SCHEDULE_MORNING_CONSENSUS_MSK", "08:30")
RAG_API_KEY: str = os.getenv("RAG_API_KEY", "")
RAG_MODEL_REASONING: str = os.getenv("RAG_MODEL_REASONING", "qwen/qwen3.5-plus-20260420")
RAG_MODEL_REACTIVE: str = os.getenv("RAG_MODEL_REACTIVE", "deepseek/deepseek-v4-flash")
RAG_MODEL_EMBEDDING: str = os.getenv("RAG_MODEL_EMBEDDING", "qwen/qwen3-embedding-8b")
GEMINI_API_KEY = RAG_API_KEY
GEMINI_MODEL_REASONING = RAG_MODEL_REASONING
GEMINI_MODEL_REACTIVE = RAG_MODEL_REACTIVE
GEMINI_MODEL_EMBEDDING = RAG_MODEL_EMBEDDING
CONSENSUS_MATCH_MAGNITUDE_BUMP: float = float(os.getenv("CONSENSUS_MATCH_MAGNITUDE_BUMP", "1.5"))
CONSENSUS_CONTRADICT_REVERSE: bool = os.getenv("CONSENSUS_CONTRADICT_REVERSE", "1") == "1"
CONSENSUS_MIN_NEWS_PER_TICKER: int = int(os.getenv("CONSENSUS_MIN_NEWS_PER_TICKER", "3"))
CONSENSUS_DROP_STRENGTH_FLOOR: float = float(os.getenv("CONSENSUS_DROP_STRENGTH_FLOOR", "0.5"))
CONSENSUS_EARNINGS_MIN_MAGNITUDE: float = float(
    os.getenv("CONSENSUS_EARNINGS_MIN_MAGNITUDE", "0.4")
)

STRATEGY_CAPITAL_ALLOCATION: dict[str, float] = {
    "TA": float(os.getenv("STRATEGY_ALLOC_TA", "0.50")),
    "NEWS": float(os.getenv("STRATEGY_ALLOC_NEWS", "0.25")),
    "ANOMALY": float(os.getenv("STRATEGY_ALLOC_ANOMALY", "0.15")),
    "MEAN_REV": float(os.getenv("STRATEGY_ALLOC_MEAN_REV", "0.10")),
}
STRATEGY_ALLOCATION_FLOOR_PCT: float = float(os.getenv("STRATEGY_ALLOCATION_FLOOR_PCT", "0.05"))
STRATEGY_DYNAMIC_REBALANCE: bool = os.getenv("STRATEGY_DYNAMIC_REBALANCE", "1") == "1"
STRATEGY_REBALANCE_LOOKBACK_DAYS: int = int(os.getenv("STRATEGY_REBALANCE_LOOKBACK_DAYS", "14"))
SCHEDULE_STRATEGY_REBALANCE_MSK: str = os.getenv("SCHEDULE_STRATEGY_REBALANCE_MSK", "04:00")

def get_strategy_allocation(source: str) -> float:
    """Return per-strategy deposit cap, falling back to floor when unknown.

    Args:
        source: SignalSource value (TA / NEWS / ANOMALY / MEAN_REV / PAIR)
    Returns:
        float: fraction of deposit (0.0–1.0)
    """
    cap = STRATEGY_CAPITAL_ALLOCATION.get(source.upper())
    if cap is None:
        return STRATEGY_ALLOCATION_FLOOR_PCT
    return max(STRATEGY_ALLOCATION_FLOOR_PCT, float(cap))

def reload_strategy_allocation_from_overrides() -> bool:
    """Load `strategy_capital_allocation` from data/runtime_overrides.json.

    This is called on startup so any change written by the Sunday
    rebalance cron is picked up by the next process. Returns True if the
    in-memory `STRATEGY_CAPITAL_ALLOCATION` was updated, False otherwise.
    """
    import json as _json

    path = DATA_DIR / "runtime_overrides.json"
    if not path.exists():
        return False
    try:
        data = _json.loads(path.read_text())
    except Exception:
        return False
    alloc = data.get("strategy_capital_allocation")
    if not isinstance(alloc, dict):
        return False
    cleaned = {str(k).upper(): float(v) for k, v in alloc.items() if isinstance(v, (int, float))}
    if not cleaned:
        return False
    STRATEGY_CAPITAL_ALLOCATION.update(cleaned)
    return True

with contextlib.suppress(Exception):
    reload_strategy_allocation_from_overrides()

ADAPTIVE_NOISE_FILTER_ENABLED: bool = os.getenv("ADAPTIVE_NOISE_FILTER_ENABLED", "1") == "1"
NOISE_WR_THRESHOLD: float = float(os.getenv("NOISE_WR_THRESHOLD", "0.35"))
NOISE_MIN_TRADES: int = int(os.getenv("NOISE_MIN_TRADES", "10"))
NOISE_LOOKBACK_DAYS: int = int(os.getenv("NOISE_LOOKBACK_DAYS", "7"))
SCHEDULE_NOISE_REVIEW_MSK: str = os.getenv("SCHEDULE_NOISE_REVIEW_MSK", "03:30")

SESSION_PROFILE_ENABLED: bool = os.getenv("SESSION_PROFILE_ENABLED", "0") == "1"
SESSION_PROFILE_VERSION: str = os.getenv(
    "SESSION_PROFILE_VERSION",
    "v1_compromise",
)
SESSION_AB_TEST_ENABLED: bool = os.getenv("SESSION_AB_TEST_ENABLED", "0") == "1"
SESSION_PROFILE_OVERRIDE_PATH = DATA_DIR / "runtime_overrides.json"

PER_TICKER_SESSION_BIAS_ENABLED: bool = (
    os.getenv("PER_TICKER_SESSION_BIAS_ENABLED", "1") == "1"
)

PER_TICKER_SESSION_BIAS: dict[tuple[str, str], float] = {
    ("SBER", "morning_open"): 1.50,
    ("X5", "morning_open"): 1.49,
    ("X5", "morning"): 1.47,
    ("VTBR", "morning"): 1.44,
    ("ALRS", "closing"): 1.40,
    ("GAZP", "closing"): 1.39,
    ("MOEX", "premarket"): 1.36,
    ("AFLT", "closing"): 1.31,
    ("MOEX", "morning_open"): 1.30,
    ("LKOH", "premarket"): 1.30,
    ("GMKN", "premarket"): 1.26,
    ("PLZL", "evening"): 1.22,
    ("SBER", "morning"): 1.19,
    ("SNGSP", "morning_open"): 0.83,
    ("GMKN", "morning_open"): 0.83,
    ("MOEX", "evening"): 0.82,
    ("PLZL", "premarket"): 0.79,
    ("X5", "closing"): 0.78,
    ("ALRS", "premarket"): 0.77,
    ("VTBR", "morning_open"): 0.75,
    ("ALRS", "morning"): 0.75,
    ("SBER", "evening"): 0.73,
    ("GMKN", "evening"): 0.72,
    ("AFLT", "premarket"): 0.72,
    ("CHMF", "premarket"): 0.71,
    ("PLZL", "morning_open"): 0.71,
    ("GAZP", "evening"): 0.71,
    ("MOEX", "closing"): 0.65,
    ("NVTK", "morning"): 0.57,
    ("ALRS", "morning_open"): 0.57,
}

def get_ticker_session_mult(ticker: str, session: str) -> float:
    """Return the per-(ticker, session) magnitude multiplier.

    Pure lookup: returns ``1.0`` (no change) when:
      - the toggle ``PER_TICKER_SESSION_BIAS_ENABLED`` is False,
      - either argument is empty,
      - the cell is not present in ``PER_TICKER_SESSION_BIAS``.

    Args:
        ticker: instrument code (case-insensitive).
        session: MSK session label (e.g. ``"midday"``, ``"closing"``).
    Returns:
        float: multiplier in ``[0.5, 1.5]`` to apply to ``combined_magnitude``.
    """
    if not PER_TICKER_SESSION_BIAS_ENABLED:
        return 1.0
    if not ticker or not session:
        return 1.0
    key = (ticker.upper(), session.lower())
    return float(PER_TICKER_SESSION_BIAS.get(key, 1.0))

CONFLUENCE_TIERED_BOOST: bool = os.getenv("CONFLUENCE_TIERED_BOOST", "1") == "1"

RISK_PARITY_VOL_NORM: bool = os.getenv("RISK_PARITY_VOL_NORM", "1") == "1"

ADAPTIVE_META_THRESHOLD: bool = os.getenv("ADAPTIVE_META_THRESHOLD", "1") == "1"

ENTRY_GUARD_ENABLED: bool = os.getenv("ENTRY_GUARD_ENABLED", "1") == "1"

ENTRY_GUARD_MAX_SPREAD_MULT: float = float(
    os.getenv("ENTRY_GUARD_MAX_SPREAD_MULT", "1.5")
)

ENTRY_GUARD_OFI_REVERSE_THRESHOLD: float = float(
    os.getenv("ENTRY_GUARD_OFI_REVERSE_THRESHOLD", "0.4")
)

ENTRY_GUARD_PRICE_SPIKE_MAX: float = float(
    os.getenv("ENTRY_GUARD_PRICE_SPIKE_MAX", "0.005")
)

TRAILING_STOP_ENABLED: bool = os.getenv("TRAILING_STOP_ENABLED", "1") == "1"

TRAILING_STOP_R_TO_BE: float = float(os.getenv("TRAILING_STOP_R_TO_BE", "1.0"))

TRAILING_STOP_R_TO_R1: float = float(os.getenv("TRAILING_STOP_R_TO_R1", "2.0"))

MAGNITUDE_NONLINEAR_SCALING: bool = (
    os.getenv("MAGNITUDE_NONLINEAR_SCALING", "1") == "1"
)

SECTOR_CAP_OVERRIDE_ENABLED: bool = (
    os.getenv("SECTOR_CAP_OVERRIDE_ENABLED", "1") == "1"
)

SECTOR_CAP_OVERRIDE: dict[str, float] = {
    "telecom":      0.10,
    "it":           0.10,
    "other":        0.10,
}

SECTOR_GROUP_CAPS_ENABLED: bool = (
    os.getenv("SECTOR_GROUP_CAPS_ENABLED", "1") == "1"
)

SECTOR_GROUP_CAPS: dict[str, float] = {
}

def get_sector_cap_pct(sector: str) -> float:
    """Return the cap for ``sector``, falling back to default 30%.

    Honours ``SECTOR_CAP_OVERRIDE_ENABLED``; when disabled, always returns
    the default `MAX_SECTOR_EXPOSURE_PCT`.

    Args:
        sector: sector key (e.g. ``"telecom_it"``); case-insensitive.
    Returns:
        float: per-sector exposure cap fraction.
    """
    if not SECTOR_CAP_OVERRIDE_ENABLED:
        return MAX_SECTOR_EXPOSURE_PCT
    s = (sector or "").lower()
    return float(SECTOR_CAP_OVERRIDE.get(s, MAX_SECTOR_EXPOSURE_PCT))

def get_sector_group_cap(sector_a: str, sector_b: str) -> float | None:
    """Return joint cap for the ``(sector_a, sector_b)`` group or None.

    Lookup is order-insensitive (``"a+b"`` and ``"b+a"`` are equivalent).
    Returns None when no group cap applies — caller should fall back to
    per-sector caps.

    Args:
        sector_a: first sector key.
        sector_b: second sector key.
    Returns:
        float | None: joint exposure cap or None.
    """
    if not SECTOR_GROUP_CAPS_ENABLED:
        return None
    a = (sector_a or "").lower()
    b = (sector_b or "").lower()
    if not a or not b:
        return None
    key1 = f"{a}+{b}"
    key2 = f"{b}+{a}"
    if key1 in SECTOR_GROUP_CAPS:
        return float(SECTOR_GROUP_CAPS[key1])
    if key2 in SECTOR_GROUP_CAPS:
        return float(SECTOR_GROUP_CAPS[key2])
    return None

BROKER_SAFE_MODE_ENABLED: bool = os.getenv("BROKER_SAFE_MODE_ENABLED", "1") == "1"
BROKER_SAFE_MODE_THRESHOLD_SEC: float = float(
    os.getenv("BROKER_SAFE_MODE_THRESHOLD_SEC", "300")
)

EQUITY_HARD_FLOOR_ENABLED: bool = os.getenv("EQUITY_HARD_FLOOR_ENABLED", "1") == "1"
EQUITY_HARD_FLOOR_PCT: float = float(os.getenv("EQUITY_HARD_FLOOR_PCT", "0.50"))
STAGE2_RESET_PEAK_EQUITY: bool = os.getenv("STAGE2_RESET_PEAK_EQUITY", "0") == "1"
EQUITY_FLOOR_STARTING_CAPITAL_RUB: float = float(
    os.getenv("EQUITY_FLOOR_STARTING_CAPITAL_RUB", "1000000")
)
EQUITY_TRAILING_PEAK_ENABLED: bool = (
    os.getenv("EQUITY_TRAILING_PEAK_ENABLED", "1") == "1"
)

SECTOR_CONCENTRATION_HAIRCUT_ENABLED: bool = (
    os.getenv("SECTOR_CONCENTRATION_HAIRCUT_ENABLED", "0") == "1"
)
SECTOR_CONCENTRATION_HAIRCUT_THRESHOLD: int = int(
    os.getenv("SECTOR_CONCENTRATION_HAIRCUT_THRESHOLD", "3")
)
SECTOR_CONCENTRATION_HAIRCUT_MULT: float = float(
    os.getenv("SECTOR_CONCENTRATION_HAIRCUT_MULT", "0.5")
)

DIRECTION_CONCENTRATION_CAP_ENABLED: bool = (
    os.getenv("DIRECTION_CONCENTRATION_CAP_ENABLED", "0") == "1"
)
DIRECTION_CONCENTRATION_THRESHOLD: int = int(
    os.getenv("DIRECTION_CONCENTRATION_THRESHOLD", "5")
)
DIRECTION_CONCENTRATION_META_MIN: float = float(
    os.getenv("DIRECTION_CONCENTRATION_META_MIN", "0.55")
)

STARTING_DEPOSIT_RUB: float = float(os.getenv("STARTING_DEPOSIT_RUB", "1000000.0"))

NEWS_LLM_REACTIVE_MAX_TOKENS: int = int(os.getenv("NEWS_LLM_REACTIVE_MAX_TOKENS", "900"))
NEWS_LLM_SANCTIONS_MAX_TOKENS: int = int(os.getenv("NEWS_LLM_SANCTIONS_MAX_TOKENS", "3000"))
NEWS_LLM_MAGNITUDE_FLOOR: float = float(os.getenv("NEWS_LLM_MAGNITUDE_FLOOR", "0.08"))
NEWS_LLM_DEFAULT_MAGNITUDE: float = float(os.getenv("NEWS_LLM_DEFAULT_MAGNITUDE", "0.5"))
NEWS_LLM_KEYWORD_FALLBACK_FLOOR: float = float(
    os.getenv("NEWS_LLM_KEYWORD_FALLBACK_FLOOR", "0.5")
)
NEWS_LLM_DEFAULT_HORIZON_MIN: int = int(os.getenv("NEWS_LLM_DEFAULT_HORIZON_MIN", "60"))
NEWS_LLM_HORIZON_MIN_MIN: int = int(os.getenv("NEWS_LLM_HORIZON_MIN_MIN", "5"))
NEWS_LLM_HORIZON_MIN_MAX: int = int(os.getenv("NEWS_LLM_HORIZON_MIN_MAX", "300"))
NEWS_LLM_BODY_MAX_CHARS: int = int(os.getenv("NEWS_LLM_BODY_MAX_CHARS", "2000"))
NEWS_LLM_REASON_MAX_CHARS: int = int(os.getenv("NEWS_LLM_REASON_MAX_CHARS", "200"))

NEWS_SOURCE_TIER_S_MULT: float = float(os.getenv("NEWS_SOURCE_TIER_S_MULT", "1.25"))
NEWS_SOURCE_TIER_A_MULT: float = float(os.getenv("NEWS_SOURCE_TIER_A_MULT", "1.12"))
NEWS_SOURCE_TIER_B_MULT: float = float(os.getenv("NEWS_SOURCE_TIER_B_MULT", "1.0"))
NEWS_SOURCE_TIER_C_MULT: float = float(os.getenv("NEWS_SOURCE_TIER_C_MULT", "0.92"))
NEWS_TA_CONFIRM_MULT: float = float(os.getenv("NEWS_TA_CONFIRM_MULT", "1.12"))
NEWS_TA_OPPOSE_MULT: float = float(os.getenv("NEWS_TA_OPPOSE_MULT", "0.88"))
NEWS_HIST_BIAS_BUMP_CAP: float = float(os.getenv("NEWS_HIST_BIAS_BUMP_CAP", "0.18"))
NEWS_HIST_BIAS_SCALE: float = float(os.getenv("NEWS_HIST_BIAS_SCALE", "8.0"))
NEWS_CATBOOST_BASE: float = float(os.getenv("NEWS_CATBOOST_BASE", "0.95"))
NEWS_CATBOOST_BUMP_CAP: float = float(os.getenv("NEWS_CATBOOST_BUMP_CAP", "0.20"))
NEWS_CATBOOST_SCALE: float = float(os.getenv("NEWS_CATBOOST_SCALE", "0.25"))

NEWS_FALLBACK_ATR_PCT: float = float(os.getenv("NEWS_FALLBACK_ATR_PCT", "0.008"))
NEWS_PULLBACK_ATR_MULT: float = float(os.getenv("NEWS_PULLBACK_ATR_MULT", "0.25"))
NEWS_STOP_ATR_MULT: float = float(os.getenv("NEWS_STOP_ATR_MULT", "1.2"))
NEWS_BREAKOUT_ATR_MULT: float = float(os.getenv("NEWS_BREAKOUT_ATR_MULT", "0.15"))
NEWS_DEFAULT_RR_BASE: float = float(os.getenv("NEWS_DEFAULT_RR_BASE", "1.0"))
NEWS_HIST_RR_BUMP_CAP: float = float(os.getenv("NEWS_HIST_RR_BUMP_CAP", "0.4"))
NEWS_HIST_RR_SCALE: float = float(os.getenv("NEWS_HIST_RR_SCALE", "10.0"))
NEWS_RR_FLOOR: float = float(os.getenv("NEWS_RR_FLOOR", "0.8"))
NEWS_RR_CAP: float = float(os.getenv("NEWS_RR_CAP", "3.5"))

DEDUP_SHINGLE_K: int = int(os.getenv("DEDUP_SHINGLE_K", "3"))
DEDUP_JACCARD_THRESHOLD: float = float(os.getenv("DEDUP_JACCARD_THRESHOLD", "0.85"))
DEDUP_NUM_PERM: int = int(os.getenv("DEDUP_NUM_PERM", "128"))
DEDUP_TTL_SEC: int = int(os.getenv("DEDUP_TTL_SEC", "86400"))
DEDUP_MAX_SIZE: int = int(os.getenv("DEDUP_MAX_SIZE", "10000"))

META_VPIN_THRESHOLD: float = float(os.getenv("META_VPIN_THRESHOLD", "0.50"))
META_VPIN_PENALTY: float = float(os.getenv("META_VPIN_PENALTY", "0.10"))
META_HEURISTIC_BASE: float = float(os.getenv("META_HEURISTIC_BASE", "0.40"))
META_HEURISTIC_FLOOR: float = float(os.getenv("META_HEURISTIC_FLOOR", "0.05"))
META_HEURISTIC_CAP: float = float(os.getenv("META_HEURISTIC_CAP", "0.95"))
META_REGIME_CRISIS_PENALTY: float = float(os.getenv("META_REGIME_CRISIS_PENALTY", "0.10"))
META_REGIME_TRENDING_BONUS: float = float(os.getenv("META_REGIME_TRENDING_BONUS", "0.03"))
META_DD_PENALTY_CAP: float = float(os.getenv("META_DD_PENALTY_CAP", "0.15"))
META_DD_PENALTY_SCALE: float = float(os.getenv("META_DD_PENALTY_SCALE", "1.5"))

PIVOT_ARGREL_ORDER: int = int(os.getenv("PIVOT_ARGREL_ORDER", "5"))
PIVOT_MERGE_DISTANCE_ATR: float = float(os.getenv("PIVOT_MERGE_DISTANCE_ATR", "0.2"))
PIVOT_ZIGZAG_ATR_MULT: float = float(os.getenv("PIVOT_ZIGZAG_ATR_MULT", "0.6"))
PIVOT_PROMINENCE_MULT: float = float(os.getenv("PIVOT_PROMINENCE_MULT", "0.2"))
PIVOT_MIN_BARS_BETWEEN: int = int(os.getenv("PIVOT_MIN_BARS_BETWEEN", "2"))

RECONCILER_SYNTH_SL_ATR: float = float(os.getenv("RECONCILER_SYNTH_SL_ATR", "1.75"))
RECONCILER_SYNTH_RR: float = float(os.getenv("RECONCILER_SYNTH_RR", "2.0"))
RECONCILER_SYNTH_ATR_PCT: float = float(os.getenv("RECONCILER_SYNTH_ATR_PCT", "0.015"))
RECONCILER_INTERVAL_SEC: float = float(os.getenv("RECONCILER_INTERVAL_SEC", "300.0"))

ARENAGO_REAUTH_MAX_ATTEMPTS: int = int(os.getenv("ARENAGO_REAUTH_MAX_ATTEMPTS", "5"))
ARENAGO_RECOVERY_STALE_SEC: float = float(os.getenv("ARENAGO_RECOVERY_STALE_SEC", "3600.0"))
ARENAGO_CASH_DROP_ALERT_RUB: float = float(os.getenv("ARENAGO_CASH_DROP_ALERT_RUB", "-50000.0"))

LLM_UNKNOWN_MODEL_PRICE_INPUT_RUB: float = float(
    os.getenv("LLM_UNKNOWN_MODEL_PRICE_INPUT_RUB", "100.0")
)
LLM_UNKNOWN_MODEL_PRICE_OUTPUT_RUB: float = float(
    os.getenv("LLM_UNKNOWN_MODEL_PRICE_OUTPUT_RUB", "100.0")
)

MORNING_PLAN_MAX_TOKENS: int = int(os.getenv("MORNING_PLAN_MAX_TOKENS", "3000"))
REFLECTION_MAX_TOKENS: int = int(os.getenv("REFLECTION_MAX_TOKENS", "4000"))
REFLECTION_DECISIONS_SUMMARY_LIMIT: int = int(
    os.getenv("REFLECTION_DECISIONS_SUMMARY_LIMIT", "30")
)

DUPLICATE_WINDOW_SEC: int = int(os.getenv("DUPLICATE_WINDOW_SEC", "300"))

GAP_UP_THRESHOLD_PCT: float = float(os.getenv("GAP_UP_THRESHOLD_PCT", "0.003"))
GAP_DOWN_THRESHOLD_PCT: float = float(os.getenv("GAP_DOWN_THRESHOLD_PCT", "-0.003"))

RISK_KELLY_FRACTION: float = float(os.getenv("RISK_KELLY_FRACTION", "0.25"))
RISK_MAX_SECTOR_PCT: float = float(os.getenv("RISK_MAX_SECTOR_PCT", "0.30"))
RISK_VOL_HIGH_RATIO: float = float(os.getenv("RISK_VOL_HIGH_RATIO", "1.5"))
RISK_VOL_LOW_RATIO: float = float(os.getenv("RISK_VOL_LOW_RATIO", "0.7"))
RISK_VOL_HIGH_MULT: float = float(os.getenv("RISK_VOL_HIGH_MULT", "0.7"))
RISK_VOL_LOW_MULT: float = float(os.getenv("RISK_VOL_LOW_MULT", "1.2"))
RISK_VOL_LOOKBACK_DAYS: int = int(os.getenv("RISK_VOL_LOOKBACK_DAYS", "60"))
RISK_PARITY_REF_ATR_PCT: float = float(os.getenv("RISK_PARITY_REF_ATR_PCT", "0.015"))
RISK_PARITY_ATR_PCT_FLOOR: float = float(os.getenv("RISK_PARITY_ATR_PCT_FLOOR", "0.005"))
RISK_PARITY_MAX_BOOST: float = float(os.getenv("RISK_PARITY_MAX_BOOST", "2.0"))

KELLY_PWIN_BASE: float = float(os.getenv("KELLY_PWIN_BASE", "0.50"))
KELLY_PWIN_MAG_COEF: float = float(os.getenv("KELLY_PWIN_MAG_COEF", "0.40"))
KELLY_RR_FLOOR: float = float(os.getenv("KELLY_RR_FLOOR", "0.5"))

LOSING_STREAK_SIZE_MULT: float = float(os.getenv("LOSING_STREAK_SIZE_MULT", "0.5"))
WINNING_STREAK_SIZE_MULT: float = float(os.getenv("WINNING_STREAK_SIZE_MULT", "1.5"))
DD_KELLY_ACTIVATION_PCT: float = float(os.getenv("DD_KELLY_ACTIVATION_PCT", "0.02"))
DD_KELLY_SHRINK_FLOOR: float = float(os.getenv("DD_KELLY_SHRINK_FLOOR", "0.1"))

AGG_CONFLUENCE_TIER4_MULT: float = float(os.getenv("AGG_CONFLUENCE_TIER4_MULT", "2.0"))
AGG_CONFLUENCE_TIER3_MULT: float = float(os.getenv("AGG_CONFLUENCE_TIER3_MULT", "1.7"))
AGG_CONFLUENCE_TIER2_MULT: float = float(os.getenv("AGG_CONFLUENCE_TIER2_MULT", "1.3"))
AGG_CONFLUENCE_TIER1_MULT: float = float(os.getenv("AGG_CONFLUENCE_TIER1_MULT", "1.0"))
AGG_ANOMALY_ONLY_WEIGHT_FACTOR: float = float(
    os.getenv("AGG_ANOMALY_ONLY_WEIGHT_FACTOR", "0.7")
)

PAIR_LEG_SL_PCT: float = float(os.getenv("PAIR_LEG_SL_PCT", "0.02"))
PAIR_LEG_TP_PCT: float = float(os.getenv("PAIR_LEG_TP_PCT", "0.04"))
PAIR_MAG_CAP: float = float(os.getenv("PAIR_MAG_CAP", "0.85"))
PAIR_MAG_BASE: float = float(os.getenv("PAIR_MAG_BASE", "0.45"))
PAIR_MAG_Z_SCALE: float = float(os.getenv("PAIR_MAG_Z_SCALE", "0.15"))
PAIR_RAW_CONFIDENCE: float = float(os.getenv("PAIR_RAW_CONFIDENCE", "0.65"))
PAIR_HORIZON_MIN: int = int(os.getenv("PAIR_HORIZON_MIN", "180"))
PAIR_EXPECTED_RR: float = float(os.getenv("PAIR_EXPECTED_RR", "1.0"))
PAIR_BETA_FLOOR: float = float(os.getenv("PAIR_BETA_FLOOR", "0.1"))
PAIR_EXIT_MAGNITUDE: float = float(os.getenv("PAIR_EXIT_MAGNITUDE", "0.50"))

ANOMALY_MAGNITUDE_DAMP: float = float(os.getenv("ANOMALY_MAGNITUDE_DAMP", "0.7"))
ANOMALY_SL_ATR_MULT: float = float(os.getenv("ANOMALY_SL_ATR_MULT", "1.5"))
ANOMALY_TP_ATR_MULT: float = float(os.getenv("ANOMALY_TP_ATR_MULT", "2.0"))
ANOMALY_SIGNAL_HORIZON_MIN: int = int(os.getenv("ANOMALY_SIGNAL_HORIZON_MIN", "15"))
ANOMALY_OBSTATS_TTL_SEC: float = float(os.getenv("ANOMALY_OBSTATS_TTL_SEC", "90.0"))

MEAN_REV_BB_PERIOD: int = int(os.getenv("MEAN_REV_BB_PERIOD", "20"))
MEAN_REV_BB_STD: float = float(os.getenv("MEAN_REV_BB_STD", "2.0"))
MEAN_REV_RSI_PERIOD: int = int(os.getenv("MEAN_REV_RSI_PERIOD", "14"))
MEAN_REV_RSI_OVERSOLD: float = float(os.getenv("MEAN_REV_RSI_OVERSOLD", "30.0"))
MEAN_REV_RSI_OVERBOUGHT: float = float(os.getenv("MEAN_REV_RSI_OVERBOUGHT", "70.0"))
MEAN_REV_ATR_STOP_MULT: float = float(os.getenv("MEAN_REV_ATR_STOP_MULT", "1.5"))
MEAN_REV_TARGET_ATR_MULT: float = float(os.getenv("MEAN_REV_TARGET_ATR_MULT", "1.5"))
MEAN_REV_MAX_HOLD_BARS: int = int(os.getenv("MEAN_REV_MAX_HOLD_BARS", "6"))
MEAN_REV_MAG_CAP: float = float(os.getenv("MEAN_REV_MAG_CAP", "0.85"))
MEAN_REV_MAG_BASE: float = float(os.getenv("MEAN_REV_MAG_BASE", "0.50"))
MEAN_REV_MAG_RSI_SCALE: float = float(os.getenv("MEAN_REV_MAG_RSI_SCALE", "0.30"))
MEAN_REV_RAW_CONFIDENCE: float = float(os.getenv("MEAN_REV_RAW_CONFIDENCE", "0.60"))

HMM_CRISIS_SIZE_MULT: float = float(os.getenv("HMM_CRISIS_SIZE_MULT", "0.85"))
HMM_MR_SIZE_MULT: float = float(os.getenv("HMM_MR_SIZE_MULT", "0.85"))
HMM_TRENDING_SIZE_MULT: float = float(os.getenv("HMM_TRENDING_SIZE_MULT", "1.0"))
HMM_CRISIS_SIGNAL_MULT: float = float(os.getenv("HMM_CRISIS_SIGNAL_MULT", "0.5"))
HMM_TRENDING_CONT_MULT: float = float(os.getenv("HMM_TRENDING_CONT_MULT", "1.2"))
HMM_TRENDING_REV_MULT: float = float(os.getenv("HMM_TRENDING_REV_MULT", "0.7"))
HMM_MR_REV_MULT: float = float(os.getenv("HMM_MR_REV_MULT", "1.2"))
HMM_MR_CONT_MULT: float = float(os.getenv("HMM_MR_CONT_MULT", "0.7"))
HMM_FIT_LOOKBACK_DAYS: int = int(os.getenv("HMM_FIT_LOOKBACK_DAYS", "60"))

CONSENSUS_WEAK_ATTENUATION_FACTOR: float = float(
    os.getenv("CONSENSUS_WEAK_ATTENUATION_FACTOR", "0.85")
)
CONSENSUS_MIN_PASSTHROUGH: float = float(os.getenv("CONSENSUS_MIN_PASSTHROUGH", "0.10"))
CONSENSUS_RAG_MAX_AGE_HOURS: int = int(os.getenv("CONSENSUS_RAG_MAX_AGE_HOURS", "24"))
CONSENSUS_LLM_MAX_TOKENS: int = int(os.getenv("CONSENSUS_LLM_MAX_TOKENS", "400"))
CONSENSUS_LLM_TEMPERATURE: float = float(os.getenv("CONSENSUS_LLM_TEMPERATURE", "0.15"))

HIST_EDGE_WIN_BONUS: float = float(os.getenv("HIST_EDGE_WIN_BONUS", "1.3"))
HIST_EDGE_LOSS_PENALTY: float = float(os.getenv("HIST_EDGE_LOSS_PENALTY", "0.7"))
HIST_EDGE_WR_HIGH: float = float(os.getenv("HIST_EDGE_WR_HIGH", "0.7"))
HIST_EDGE_WR_LOW: float = float(os.getenv("HIST_EDGE_WR_LOW", "0.3"))
HIST_EDGE_MIN_SAMPLES: int = int(os.getenv("HIST_EDGE_MIN_SAMPLES", "3"))
HIST_EDGE_LOOKBACK_DAYS: int = int(os.getenv("HIST_EDGE_LOOKBACK_DAYS", "30"))

ADAPTIVE_END_OF_SESSION_SECS: float = float(
    os.getenv("ADAPTIVE_END_OF_SESSION_SECS", "1800.0")
)

DISPATCHER_EMPTY_POLLS_ALERT_THRESHOLD: int = int(
    os.getenv("DISPATCHER_EMPTY_POLLS_ALERT_THRESHOLD", "3")
)
DISPATCHER_CYCLE_BUDGET_OVERSHOOT_MULT: float = float(
    os.getenv("DISPATCHER_CYCLE_BUDGET_OVERSHOOT_MULT", "1.1")
)
DISPATCHER_SUPERCANDLES_CONCURRENCY: int = int(
    os.getenv("DISPATCHER_SUPERCANDLES_CONCURRENCY", "4")
)

POSITION_BOOK_REFRESH_INTERVAL_SEC: int = int(
    os.getenv("POSITION_BOOK_REFRESH_INTERVAL_SEC", "30")
)
MTM_PRICE_TTL_SEC: float = float(os.getenv("MTM_PRICE_TTL_SEC", "60.0"))

NEWS_BUS_MAX_QUEUE_SIZE: int = int(os.getenv("NEWS_BUS_MAX_QUEUE_SIZE", "4000"))
NEWS_BUS_HIGH_WATERMARK: int = int(os.getenv("NEWS_BUS_HIGH_WATERMARK", "3200"))
NEWS_BUS_PRIORITY_QUEUE_SIZE: int = int(os.getenv("NEWS_BUS_PRIORITY_QUEUE_SIZE", "200"))
NEWS_BUS_RECENT_BUFFER_SIZE: int = int(os.getenv("NEWS_BUS_RECENT_BUFFER_SIZE", "500"))

PRETRADE_GATE_ENABLED: bool = os.getenv("PRETRADE_GATE_ENABLED", "0") == "1"
PRETRADE_GATE_TIERS: str = os.getenv("PRETRADE_GATE_TIERS", "Tier1,Tier2")
PRETRADE_GATE_MAX_TOKENS: int = int(os.getenv("PRETRADE_GATE_MAX_TOKENS", "200"))

MARKET_MOOD_ENABLED: bool = os.getenv("MARKET_MOOD_ENABLED", "1") == "1"
MARKET_MOOD_INTERVAL_SEC: int = int(os.getenv("MARKET_MOOD_INTERVAL_SEC", "3600"))
MARKET_MOOD_MAX_TOKENS: int = int(os.getenv("MARKET_MOOD_MAX_TOKENS", "500"))

POST_TRADE_REFLECTION_ENABLED: bool = os.getenv("POST_TRADE_REFLECTION_ENABLED", "1") == "1"
POST_TRADE_REFLECTION_MAX_TOKENS: int = int(os.getenv("POST_TRADE_REFLECTION_MAX_TOKENS", "300"))
POST_TRADE_REFLECTION_MIN_PNL_RUB: float = float(
    os.getenv("POST_TRADE_REFLECTION_MIN_PNL_RUB", "100.0")
)

MULTI_MODEL_CONSENSUS_ENABLED: bool = os.getenv("MULTI_MODEL_CONSENSUS_ENABLED", "0") == "1"
MULTI_MODEL_CONSENSUS_TIERS: str = os.getenv("MULTI_MODEL_CONSENSUS_TIERS", "Tier1")
MULTI_MODEL_CONSENSUS_BOOST: float = float(os.getenv("MULTI_MODEL_CONSENSUS_BOOST", "1.2"))
MULTI_MODEL_CONSENSUS_VETO_THRESHOLD: int = int(
    os.getenv("MULTI_MODEL_CONSENSUS_VETO_THRESHOLD", "2")
)

SANCTIONS_DEEP_DIVE_ENABLED: bool = os.getenv("SANCTIONS_DEEP_DIVE_ENABLED", "1") == "1"
SANCTIONS_DEEP_MAX_TOKENS: int = int(os.getenv("SANCTIONS_DEEP_MAX_TOKENS", "6000"))

USE_QWEN_FOR_MORNING_BRIEF: bool = os.getenv("USE_QWEN_FOR_MORNING_BRIEF", "1") == "1"
