"""Контракты данных (UnifiedSignal, Decision, TradeRequest)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Optional
from uuid import uuid4

try:
    from pydantic import BaseModel, Field, field_validator  # type: ignore

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False
    BaseModel = object  # type: ignore

class SignalSource(StrEnum):
    """Signal Source."""

    TA = "TA"
    ANOMALY = "ANOMALY"
    NEWS = "NEWS"
    PAIR = "PAIR"
    MEAN_REV = "MEAN_REV"

class Direction(StrEnum):
    """Direction."""

    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class DecisionAction(StrEnum):
    """Decision Action."""

    EXECUTE = "EXECUTE"
    VETO = "VETO"
    NO_TRADE = "NO_TRADE"
    WAIT_CONFLUENCE = "WAIT_CONFLUENCE"

class DecisionTier(StrEnum):
    """Decision Tier."""

    TIER1 = "1"
    TIER2 = "2"
    TIER3 = "3"
    NONE = "NONE"

class RiskCheckResult(StrEnum):
    """Risk Check Result."""

    PASSED = "PASSED"
    REJECTED_HARD_CAP = "REJECTED_HARD_CAP"
    REJECTED_MAX_POSITIONS = "REJECTED_MAX_POSITIONS"
    REJECTED_SECTOR_LIMIT = "REJECTED_SECTOR_LIMIT"
    REJECTED_CORRELATION = "REJECTED_CORRELATION"
    REJECTED_CASH_RESERVE = "REJECTED_CASH_RESERVE"
    REJECTED_HOLDING_PERIOD = "REJECTED_HOLDING_PERIOD"
    REJECTED_CIRCUIT_BREAKER = "REJECTED_CIRCUIT_BREAKER"
    REJECTED_MARKET_CLOSED = "REJECTED_MARKET_CLOSED"
    REJECTED_DAILY_LIMIT = "REJECTED_DAILY_LIMIT"
    REJECTED_STRATEGY_ALLOCATION = "REJECTED_STRATEGY_ALLOCATION"

class ArenaGoError(StrEnum):
    """Arena Go Error."""

    MARKET_CLOSED = "MARKET CLOSED"
    NOT_VALID_SECID = "NOT VALID SECID"
    INSUFFICIENT_CASH = "INSUFFICIENT CASH"
    DAILY_TRADE_LIMIT = "HAS REACHED DAILY TRADE LIMIT"
    UNKNOWN = "UNKNOWN"

if _HAS_PYDANTIC:

    class UnifiedSignal(BaseModel):
        """Signal from any model adapter routed to Dispatcher."""

        signal_id: str = Field(default_factory=lambda: uuid4().hex)
        source: SignalSource
        detector: str
        ticker: str
        direction: Direction
        magnitude: float
        raw_confidence: float
        horizon_min: int
        timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
        price: float
        entry_level: float | None = None
        stop_level: float | None = None
        target_level: float | None = None
        expected_rr: float = 0.0
        atr: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            """Return serializable dict.

            Returns:
                dict[str, Any]: JSON-mode dump
            """
            return self.model_dump(mode="json")

        @field_validator("magnitude", "raw_confidence")
        @classmethod
        def clamp_01(cls, v: float) -> float:
            """Clamp 01."""
            return max(0.0, min(1.0, v))

        @field_validator("ticker")
        @classmethod
        def upper_ticker(cls, v: str) -> str:
            """Upper ticker."""
            return v.upper()

    class Decision(BaseModel):
        """Dispatcher decision produced from one or more UnifiedSignals."""

        decision_id: str
        cycle_id: str
        ticker: str
        action: DecisionAction
        tier: DecisionTier = DecisionTier.NONE
        direction: Direction = Direction.NEUTRAL
        combined_magnitude: float = 0.0
        signals: list[UnifiedSignal] = Field(default_factory=list)
        risk_check: RiskCheckResult = RiskCheckResult.PASSED
        trade_request: TradeRequest | None = None
        expected_holding_min: int = 0
        stop_loss: float | None = None
        take_profit: float | None = None
        take_profit_1: float | None = None
        take_profit_2: float | None = None
        expected_rr: float = 0.0
        rationale: str = ""
        git_commit: str = ""
        prompt_versions: dict[str, str] = Field(default_factory=dict)
        created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
        executed_at: datetime | None = None
        pnl_rub: float | None = None
        reflection_status: str = "PENDING"

        meta_score: float | None = None
        meta_threshold: float | None = None

        gate_reason: str | None = None

        dominant_source: str | None = None

        @classmethod
        def make_id(cls, cycle_id: str, ticker: str, signal_ids: list[str]) -> str:
            """Build deterministic decision_id from cycle+ticker+signals.

            Args:
                cycle_id: dispatcher cycle id
                ticker: instrument code
                signal_ids: contributing signal ids
            Returns:
                str: 16-char SHA1 hex digest
            """
            key = f"{cycle_id}:{ticker}:{':'.join(sorted(signal_ids))}"
            return hashlib.sha1(key.encode()).hexdigest()[:16]

        def to_dict(self) -> dict[str, Any]:
            """Return serializable dict.

            Returns:
                dict[str, Any]: JSON-mode dump
            """
            return self.model_dump(mode="json")

    class TradeRequest(BaseModel):
        """Execution payload sent to ArenaGo."""

        decision_id: str
        ticker: str
        direction: Direction
        quantity: int
        bot: str
        price_at_signal: float

        def to_arena_order(self) -> dict[str, Any]:
            """Return the exact ArenaGo POST body.

            Returns:
                dict[str, Any]: ArenaGo order payload
            """
            direction_map = {Direction.BUY: "B", Direction.SELL: "S"}
            return {
                "direction": direction_map[self.direction],
                "secid": self.ticker,
                "quantity": self.quantity,
                "bot": self.bot,
            }

    class ConfluenceResult(BaseModel):
        """Result of anomaly verify_signal cross-check."""

        ticker: str
        direction: Direction
        matching_count: int
        opposing_count: int
        multiplier: float

        @property
        def is_confirmed(self) -> bool:
            """True when ≥2 matches and no opposers.

            Returns:
                bool: confirmation status
            """
            return self.matching_count >= 2 and self.opposing_count == 0

        @property
        def is_vetoed(self) -> bool:
            """True when an opposer is present and no matches.

            Returns:
                bool: veto status
            """
            return self.opposing_count >= 1 and self.matching_count == 0

    Decision.model_rebuild(_types_namespace={"TradeRequest": TradeRequest, "Optional": Optional})

else:

    class UnifiedSignal:  # type: ignore
        """Unified Signal."""

        pass

    class Decision:  # type: ignore
        """Decision."""

        pass

    class TradeRequest:  # type: ignore
        """Trade Request."""

        pass

    class ConfluenceResult:  # type: ignore
        """Confluence Result."""

        pass
