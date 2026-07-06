"""Точка входа автономного трейдера."""

from __future__ import annotations

import asyncio
import os
import contextlib
import signal
import sys
from datetime import UTC, datetime

import app.config as cfg
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

async def bootstrap() -> None:
    """Create SQLite schemas if missing."""
    from scripts.bootstrap_db import main as _bootstrap_main

    try:
        _bootstrap_main()
    except Exception as exc:
        logger.warning("bootstrap_db failed (may already exist)", extra={"error": str(exc)})

async def main() -> None:
    """Main process entry point."""
    setup_logging()
    logger.info(
        "Запуск автономного трейдера",
        extra={
            "version": "1.5.0",
            "run_mode": cfg.RUN_MODE,
            "live_sizing": cfg.LIVE_SIZING,
            "strict_mode": cfg.STRICT_MODE,
            "adaptive_risk_regime": "enabled",
            "deposit_total": 1_000_000,
            "tickers": len(cfg.TICKERS),
        },
    )
    logger.info(
        "Снимок эффективной конфигурации",
        extra={
            "tier1_min_mag": cfg.TIER1_MIN_MAGNITUDE,
            "tier1_min_rr": cfg.TIER1_MIN_RR,
            "tier2_min_mag": cfg.TIER2_MIN_MAGNITUDE,
            "tier2_min_rr": cfg.TIER2_MIN_RR,
            "tier3_min_mag": cfg.TIER3_MIN_MAGNITUDE,
            "tier3_min_rr": cfg.TIER3_MIN_RR,
            "tier1_size_pct": cfg.TIER1_SIZE_PCT,
            "tier2_size_pct": cfg.TIER2_SIZE_PCT,
            "tier3_size_pct": cfg.TIER3_SIZE_PCT,
            "force_regime_name": getattr(cfg, "FORCE_REGIME_NAME", "—"),
            "disable_regime_dd_check": getattr(cfg, "DISABLE_REGIME_DD_CHECK", False),
            "hard_stop_loss_pct": getattr(cfg, "HARD_STOP_LOSS_PCT", None),
            "hard_take_profit_pct": getattr(cfg, "HARD_TAKE_PROFIT_PCT", None),
            "hard_take_profit_aged_pct": getattr(cfg, "HARD_TAKE_PROFIT_AGED_PCT", None),
            "hard_tp_aged_hours": getattr(cfg, "HARD_TP_AGED_HOURS", None),
            "hard_time_stop_hours": getattr(cfg, "HARD_TIME_STOP_HOURS", None),
            "sl_monitor_interval_sec": getattr(cfg, "SL_MONITOR_INTERVAL_SEC", None),
            "min_notional_rub": getattr(cfg, "MIN_NOTIONAL_RUB", None),
            "counter_bias_guard": getattr(cfg, "COUNTER_BIAS_GUARD_ENABLED", False),
            "max_ticker_pct_cumulative": getattr(cfg, "MAX_TICKER_NOTIONAL_PCT_CUMULATIVE", None),
            "pre_long_momentum_guard": getattr(cfg, "PRE_LONG_MOMENTUM_GUARD_ENABLED", False),
            "pre_long_min_momentum": getattr(cfg, "PRE_LONG_MIN_MOMENTUM_PCT", None),
            "drawdown_alert_pct": getattr(cfg, "DRAWDOWN_ALERT_PCT", None),
            "meta_min_proba": getattr(cfg, "META_MIN_PROBA", None),
            "pair_leg_tp_pct": getattr(cfg, "PAIR_LEG_TP_PCT", None),
            "pair_leg_sl_pct": getattr(cfg, "PAIR_LEG_SL_PCT", None),
            "max_open_positions": getattr(cfg, "MAX_OPEN_POSITIONS", None),
            "n_gold_tickers": sum(
                1 for v in cfg.PER_TICKER_POLICY.values() if v == "GOLD"
            ),
            "n_whitelist_tickers": sum(
                1 for v in cfg.PER_TICKER_POLICY.values() if v == "WHITELIST_ONLY"
            ),
            "n_disabled_tickers": sum(
                1 for v in cfg.PER_TICKER_POLICY.values() if v == "DISABLED"
            ),
            "build_time": os.getenv("BUILD_TIME", "—"),
        },
    )

    await bootstrap()

    try:
        import uvloop  # type: ignore

        uvloop.install()
        logger.info("uvloop установлен для asyncio")
    except (ImportError, AttributeError):
        pass

    from app.data.algopack_client import get_algopack_client
    from app.data.candle_store import get_candle_store
    from app.data.iss_client import get_iss_client
    from app.execution.arenago_client import get_arenago_client
    from app.llm.polza_client import get_polza_client

    iss = get_iss_client()
    algopack = get_algopack_client()
    arenago = get_arenago_client()
    polza = get_polza_client()

    startup_results = await asyncio.gather(
        iss.startup(),
        algopack.startup(),
        arenago.startup(),
        polza.startup(),
        return_exceptions=True,
    )
    for name, result in zip(("iss", "algopack", "arenago", "polza"), startup_results, strict=False):
        if isinstance(result, Exception):
            logger.error(
                "Data/exec client startup failed (continuing degraded)",
                extra={"client": name, "error": str(result)},
            )

    store = get_candle_store()
    if cfg.RUN_MODE != "backtest":
        try:
            await store.warm_up(intervals=(1, 5, 10))
        except Exception as exc:
            logger.warning("Candle warm-up failed", extra={"error": str(exc)})

    from app.agents.anomaly_detector import get_anomaly_agent
    from app.agents.mean_reversion import get_mean_reversion
    from app.agents.news_llm import get_news_llm
    from app.agents.pair_trader import get_pair_trader
    from app.agents.ta_trader import get_ta_trader

    ta = get_ta_trader()
    anomaly = get_anomaly_agent()
    news = get_news_llm()
    pair = get_pair_trader()
    mean_rev = get_mean_reversion()

    try:
        from app.memory.reflexive_overrides import apply_saved_reflexive_overrides

        restored_reflexive = apply_saved_reflexive_overrides()
        if restored_reflexive.get("applied"):
            logger.info("Reflexive runtime params restored", extra=restored_reflexive)
    except Exception as exc:
        logger.warning("Reflexive runtime restore skipped", extra={"error": str(exc)})

    agent_results = await asyncio.gather(
        ta.startup(),
        anomaly.startup(),
        news.startup(),
        pair.startup(),
        mean_rev.startup(),
        return_exceptions=True,
    )
    for name, result in zip(
        ("ta", "anomaly", "news", "pair", "mean_rev"), agent_results, strict=False
    ):
        if isinstance(result, Exception):
            logger.error(
                "Agent startup failed (continuing degraded)",
                extra={"agent": name, "error": str(result)},
            )

    try:
        await ta.fit_hmm_from_iss(days=60)
    except Exception as exc:
        logger.warning("HMM warm-up failed (will fit on schedule)", extra={"error": str(exc)})

    try:
        await pair.refit_all_pairs()
    except Exception as exc:
        logger.warning("Pair refit on startup failed", extra={"error": str(exc)})

    from app.risk.circuit_breakers import get_circuit_breaker
    from app.risk.position_book import get_position_book

    cb = get_circuit_breaker()
    await cb.load()

    book = get_position_book()
    await book.start()

    from app.execution.order_manager import get_order_manager

    _om = get_order_manager()
    try:
        recon = await _om.reconcile_pending_decisions(lookback_hours=4)
        logger.info(
            "Сверка orphan-ордеров завершена",
            extra=recon,
        )
        if recon.get("orphans_found", 0) > 0:
            logger.critical(
                "Reconcile flipped pending decisions to executed — capital "
                "was at risk of orphan-order leak",
                extra=recon,
            )
    except Exception as exc:
        logger.error("Сверка orphan-ордеров не удалась", extra={"error": str(exc)})

    reconciler = None
    try:
        from app.execution.broker_reconciler import get_broker_reconciler

        reconciler = get_broker_reconciler()
        startup_report = await reconciler.reconcile_once()
        if startup_report.has_mismatch:
            logger.info(
                "Startup broker reconciliation converged state",
                extra={
                    "synthetic_added": startup_report.synthetic_added,
                    "marked_closed": startup_report.marked_closed,
                    "cash_divergence_rub": round(startup_report.cash_divergence_rub, 4),
                    "fetched_positions": startup_report.fetched_positions,
                    "fetched_trades": startup_report.fetched_trades,
                },
            )
        await book.refresh()
        await reconciler.start_periodic()
    except Exception as exc:
        logger.warning(
            "Broker reconciliation startup failed (continuing degraded)",
            extra={"error": str(exc)},
        )

    from app.recovery import get_recovery_manager

    recovery = get_recovery_manager()
    prior_snap = recovery.load()
    if prior_snap is not None:
        logger.info(
            "Восстановление состояния сессии",
            extra={
                "hmm_regime": prior_snap.hmm_regime,
                "last_n_decisions": len(prior_snap.last_decision_ids),
                "n_open_positions_hint": len(prior_snap.open_positions),
                "n_open_positions_broker": book.n_open_positions,
                "n_trades_today_carryover": prior_snap.n_trades_today,
            },
        )

        snap_tickers = {
            p.get("ticker", "").upper() for p in prior_snap.open_positions if p.get("ticker")
        }
        broker_tickers = set(book.positions.keys())
        only_in_snap = snap_tickers - broker_tickers
        only_in_broker = broker_tickers - snap_tickers
        if only_in_snap or only_in_broker:
            logger.critical(
                "Position divergence on recovery — broker is source of truth",
                extra={
                    "only_in_snapshot": sorted(only_in_snap),
                    "only_in_broker": sorted(only_in_broker),
                    "broker_count": len(broker_tickers),
                    "snapshot_count": len(snap_tickers),
                },
            )

        with contextlib.suppress(Exception):
            _om.recent_decision_ids.extend(prior_snap.last_decision_ids[-100:])

        if prior_snap.n_trades_today > cb.state.n_trades_today:
            cb.state.n_trades_today = prior_snap.n_trades_today

        with contextlib.suppress(Exception):
            peak = prior_snap.extras.get("equity_peak_rub") if prior_snap.extras else None
            if peak is not None:
                from app.risk.equity_floor import get_equity_floor

                get_equity_floor().load_peak(float(peak))
                logger.info(
                    "Восстановлен trailing-peak equity",
                    extra={"peak_equity_rub": round(float(peak), 2)},
                )

        if getattr(cfg, "STAGE2_RESET_PEAK_EQUITY", False):
            from app.risk.equity_floor import get_equity_floor

            start_dep = float(getattr(cfg, "STARTING_DEPOSIT_RUB", 1_000_000.0))

            get_equity_floor().reset_for_stage(label="STAGE2_RESET_PEAK_EQUITY=1")

            cb.state.blocked_until_iso = None
            cb.state.block_reason = ""
            cb.state.peak_equity_rub = start_dep
            cb.state.current_drawdown_pct = 0.0
            cb.state.max_drawdown_pct = 0.0
            cb.state.losing_streak = 0
            cb.state.winning_streak = 0
            cb.state.daily_pnl_rub = 0.0
            cb.state.n_trades_today = 0
            cb.state.current_equity_rub = start_dep
            with contextlib.suppress(Exception):
                await cb._persist()

            try:
                book.cash_balance = start_dep
                if hasattr(book, "_positions"):
                    book._positions.clear()
                logger.critical(
                    "STAGE2_RESET: position_book cleared in-memory",
                    extra={"cash_balance_rub": start_dep, "positions_cleared": True},
                )
            except Exception as exc:
                logger.error("STAGE2_RESET: position_book clear failed", extra={"error": str(exc)})

            logger.critical(
                "STAGE2_RESET: circuit_breaker + equity_floor + position_book reset",
                extra={
                    "action": "stage2_reset",
                    "starting_capital_rub": start_dep,
                    "components_reset": [
                        "equity_floor.peak", "equity_floor.breach",
                        "cb.blocked_until", "cb.peak_equity", "cb.drawdown",
                        "cb.losing_streak", "cb.winning_streak",
                        "cb.daily_pnl", "cb.n_trades_today",
                        "position_book.positions", "position_book.cash",
                    ],
                    "preserved_in_data": [
                        "decisions.db", "trades.db",
                        "feeds.db", "models/", "logs/",
                    ],
                },
            )

    rss_manager = None
    sanctions_parser = None
    moex_parser = None
    if cfg.RUN_MODE != "backtest":
        try:
            from app.news.parsers.moex_iss_parser import get_moex_iss_parser
            from app.news.parsers.rss_parser import get_rss_manager
            from app.news.parsers.sanctions_parser import get_sanctions_parser

            rss_manager = get_rss_manager()
            rss_manager.start()

            sanctions_parser = get_sanctions_parser()
            await sanctions_parser.start()

            moex_parser = get_moex_iss_parser()
            await moex_parser.start()
        except Exception as exc:
            logger.error("News ingestion startup failed", extra={"error": str(exc)})

    from app.agents.meta_classifier import get_meta_classifier
    from app.agents.microstructure_gates import get_microstructure_gates
    from app.dispatcher.aggregator import SignalAggregator
    from app.dispatcher.dispatcher import Dispatcher, set_active_dispatcher

    meta_classifier = get_meta_classifier()
    micro_gates = get_microstructure_gates()
    aggregator = SignalAggregator(
        anomaly_agent=anomaly,
        meta_classifier=meta_classifier,
        microstructure_gates=micro_gates,
    )

    dispatcher = Dispatcher(
        adapters=[ta, anomaly, news, mean_rev],
        aggregator=aggregator,
        cycle_seconds=cfg.DISPATCHER_CYCLE_SECONDS,
        poll_timeout_seconds=cfg.POLL_TIMEOUT_SECONDS,
    )
    set_active_dispatcher(dispatcher)

    try:
        news.set_dispatcher_trigger(dispatcher.priority_event)
    except Exception as exc:
        logger.warning("Failed to wire news -> dispatcher trigger", extra={"error": str(exc)})

    rag_store = None
    comparator = None
    if cfg.RAG_CONSENSUS_ENABLED:
        try:
            from app.agents.consensus_compare import ConsensusComparator
            from app.memory.rag_store import get_rag_store
            from app.news.consensus_rag import build_morning_consensus, load_consensus

            rag_store = get_rag_store()
            consensus_today = load_consensus()
            if not consensus_today:
                try:
                    consensus_today = await build_morning_consensus(rag_store, cfg.RAG_LLM_BACKEND)
                except Exception as exc:
                    logger.warning(
                        "Initial consensus build failed (will retry at schedule)",
                        extra={"error": str(exc)},
                    )
                    consensus_today = {}
            comparator = ConsensusComparator(
                rag=rag_store,
                consensus_today=consensus_today,
                llm_backend=cfg.RAG_LLM_BACKEND,
            )
            news.attach_comparator(comparator)
            logger.info(
                "RAG consensus подключён",
                extra={
                    "backend": cfg.RAG_LLM_BACKEND,
                    "consensus_tickers": len(consensus_today),
                    "rag_records": len(rag_store),
                },
            )
        except Exception as exc:
            logger.warning(
                "RAG consensus wiring failed (continuing without)",
                extra={"error": str(exc)},
            )

    from app.execution.turnover_tracker import get_turnover_tracker
    from app.memory.morning_plan import get_morning_planner
    from app.memory.reflection import get_reflection_engine
    from app.utils.scheduler import scheduler

    morning = get_morning_planner()
    reflection = get_reflection_engine()
    turnover = get_turnover_tracker()

    @scheduler.daily(cfg.SCHEDULE_MORNING_BRIEF_MSK)
    async def _morning_task():
        """Morning task."""
        try:
            await morning.generate()
        except Exception as exc:
            logger.exception("Morning plan failed", extra={"error": str(exc)})

    @scheduler.daily(cfg.SCHEDULE_EVENING_REFLECTION_MSK)
    async def _reflection_task():
        """Reflection task."""
        try:
            await reflection.run_today()
        except Exception as exc:
            logger.exception("Reflection failed", extra={"error": str(exc)})

    from app.training.evening_pipeline import get_evening_pipeline

    evening_pipeline = get_evening_pipeline()

    @scheduler.daily(cfg.SCHEDULE_EVENING_PIPELINE_MSK)
    async def _evening_pipeline_task():
        """Evening pipeline task."""
        try:
            await evening_pipeline.run()
        except Exception as exc:
            logger.exception("Evening pipeline failed", extra={"error": str(exc)})

    @scheduler.weekly(
        cfg.SCHEDULE_WEEKLY_FULL_RETRAIN_DAY_OF_WEEK,
        cfg.SCHEDULE_WEEKLY_FULL_RETRAIN_MSK,
    )
    async def _weekly_full_retrain_task():
        """Weekly full retrain task."""
        try:
            await evening_pipeline.run_weekly_full()
        except Exception as exc:
            logger.exception("Weekly full retrain failed", extra={"error": str(exc)})

    @scheduler.daily(cfg.SCHEDULE_TURNOVER_CHECK_MSK)
    async def _turnover_task():
        """Turnover task."""
        try:
            await turnover.run_check()
        except Exception as exc:
            logger.exception("Turnover check failed", extra={"error": str(exc)})

    @scheduler.daily(cfg.SCHEDULE_DAILY_CIRCUIT_BREAKER_RESET_MSK)
    async def _cb_reset_task():
        """Cb reset task."""
        try:
            await cb.reset_daily()
        except Exception as exc:
            logger.exception("Circuit breaker reset failed", extra={"error": str(exc)})

    if cfg.RAG_CONSENSUS_ENABLED and rag_store is not None and comparator is not None:

        @scheduler.daily(cfg.SCHEDULE_MORNING_CONSENSUS_MSK)
        async def _morning_consensus_task():
            """Morning consensus task."""
            try:
                from app.news.consensus_rag import build_morning_consensus

                pruned = rag_store.prune_older_than(hours=cfg.RAG_PRUNE_HOURS)
                consensus_today = await build_morning_consensus(rag_store, cfg.RAG_LLM_BACKEND)
                comparator.update_consensus(consensus_today)
                logger.info(
                    "Morning consensus rebuilt",
                    extra={
                        "pruned": pruned,
                        "consensus_tickers": len(consensus_today),
                        "n_non_neutral": sum(
                            1 for e in consensus_today.values() if e.direction != "NEUTRAL"
                        ),
                    },
                )
            except Exception as exc:
                logger.exception(
                    "Morning consensus rebuild failed",
                    extra={"error": str(exc)},
                )

    if cfg.ADAPTIVE_NOISE_FILTER_ENABLED:

        @scheduler.daily(cfg.SCHEDULE_NOISE_REVIEW_MSK)
        async def _noise_review_task():
            """Noise review task."""
            try:
                from app.agents.ta_patterns.noise_blacklist import (
                    update_dynamic_overrides,
                )
                from scripts.noise_review import review

                patterns = review(cfg.NOISE_LOOKBACK_DAYS)
                update_dynamic_overrides(patterns)
                logger.info(
                    "Adaptive noise review complete",
                    extra={"n_patterns": len(patterns), "lookback_days": cfg.NOISE_LOOKBACK_DAYS},
                )
            except Exception as exc:
                logger.exception(
                    "Adaptive noise review failed",
                    extra={"error": str(exc)},
                )

    if cfg.STRATEGY_DYNAMIC_REBALANCE:

        @scheduler.daily(cfg.SCHEDULE_STRATEGY_REBALANCE_MSK)
        async def _strategy_rebalance_task():
            """Strategy rebalance task."""
            try:
                from datetime import datetime as _dt

                now = _dt.now()
                if now.weekday() != 6:
                    return
                from app.data.iss_client import get_iss_client
                from scripts.strategy_backtest import (
                    _aggregate,
                    _anomaly_trades_one_ticker,
                    _fetch_candles,
                    _mean_rev_trades_one_ticker,
                    _news_trades_from_db,
                    _ta_trades_from_rankings,
                    compute_optimal_allocation,
                )

                days = cfg.STRATEGY_REBALANCE_LOOKBACK_DAYS
                rankings = cfg.DATA_DIR / "training_cache" / "detector_rankings_after.json"
                ta_trades = _ta_trades_from_rankings(rankings)
                anomaly_trades: list = []
                mr_trades: list = []
                try:
                    iss = get_iss_client()
                    await iss.startup()
                    for ticker in cfg.TICKERS:
                        df = await _fetch_candles(iss, ticker, 10, days)
                        if df is None:
                            continue
                        anomaly_trades.extend(_anomaly_trades_one_ticker(df, ticker))
                        mr_trades.extend(_mean_rev_trades_one_ticker(df, ticker))
                    await iss.shutdown()
                except Exception:
                    pass
                news_trades = _news_trades_from_db(
                    cfg.DATA_DIR / "feeds.db",
                    days=days,
                )
                metrics = {
                    "TA": _aggregate(ta_trades, bars_per_day=6.5),
                    "ANOMALY": _aggregate(anomaly_trades, bars_per_day=39.0),
                    "NEWS": _aggregate(news_trades, bars_per_day=4.0),
                    "MEAN_REV": _aggregate(mr_trades, bars_per_day=39.0),
                }
                alloc = compute_optimal_allocation(
                    metrics,
                    floor_pct=cfg.STRATEGY_ALLOCATION_FLOOR_PCT,
                )
                cfg.STRATEGY_CAPITAL_ALLOCATION.update(alloc["final"])
                ov_path = cfg.DATA_DIR / "runtime_overrides.json"
                payload: dict = {}
                if ov_path.exists():
                    try:
                        payload = __import__("json").loads(ov_path.read_text())
                    except Exception:
                        payload = {}
                payload["strategy_capital_allocation"] = dict(alloc["final"])
                payload["strategy_alloc_updated_at_utc"] = datetime.now(tz=UTC).isoformat()
                ov_path.write_text(__import__("json").dumps(payload, indent=2))
                logger.info(
                    "Strategy rebalance complete",
                    extra={"allocation": alloc["final"]},
                )
            except Exception as exc:
                logger.exception(
                    "Strategy rebalance failed",
                    extra={"error": str(exc)},
                )

    @scheduler.daily(cfg.SCHEDULE_DATA_FEEDS_MSK)
    async def _dividend_calendar_task():
        """Dividend calendar task."""
        try:
            from app.news.parsers.dividend_calendar_parser import (
                fetch_smartlab_dividends,
                save_to_feeds_db,
            )

            events = await fetch_smartlab_dividends()
            saved = save_to_feeds_db(events)
            logger.info(
                "Dividend calendar refreshed",
                extra={
                    "events": len(events),
                    "saved": saved,
                    "tickers": [e.ticker for e in events],
                },
            )
        except Exception as exc:
            logger.exception("Dividend calendar refresh failed", extra={"error": str(exc)})

    try:
        from app.news.parsers.dividend_calendar_parser import (
            fetch_smartlab_dividends,
            save_to_feeds_db,
        )

        startup_events = await fetch_smartlab_dividends()
        startup_saved = save_to_feeds_db(startup_events)
        logger.info(
            "Календарь дивидендов загружен при старте",
            extra={"events": len(startup_events), "saved": startup_saved},
        )
    except Exception as exc:
        logger.warning(
            "Dividend calendar bootstrap failed (will retry on schedule)",
            extra={"error": str(exc)},
        )

    scheduler.start()

    async def _recovery_loop() -> None:
        """Save recovery snapshot every cfg.RECOVERY_SAVE_INTERVAL_SEC."""
        from dataclasses import asdict

        from app.recovery import RecoveryStateManager

        last_ids: list[str] = []
        meta_hist: list[float] = []
        if prior_snap is not None:
            last_ids = list(prior_snap.last_decision_ids)
            meta_hist = list(prior_snap.meta_score_history)
        while True:
            try:
                await asyncio.sleep(cfg.RECOVERY_SAVE_INTERVAL_SEC)

                try:
                    recent = list(getattr(dispatcher.orders, "recent_decision_ids", []) or [])
                    if recent:
                        last_ids = (last_ids + recent)[-100:]
                except Exception:
                    pass
                open_positions_snap: list[dict] = []
                try:
                    for ticker, pos in book.positions.items():
                        open_positions_snap.append(
                            {
                                "ticker": ticker,
                                "quantity": int(pos.quantity),
                                "avg_price": float(pos.avg_price),
                                "bot": str(pos.bot),
                                "entry_ts": float(pos.entry_ts),
                            }
                        )
                except Exception:
                    pass
                extras_dict: dict = {}
                try:
                    from app.risk.equity_floor import get_equity_floor

                    floor_inst = get_equity_floor()
                    current_eq = float(book.total_equity())
                    floor_inst.update_peak(current_eq)
                    extras_dict["equity_peak_rub"] = floor_inst.peak_equity_rub
                except Exception:  # pragma: no cover — never block save loop
                    pass
                snap = RecoveryStateManager.build_snapshot(
                    circuit_state_dict=asdict(cb.state),
                    hmm_regime=getattr(ta.hmm, "_current_label", "unknown")
                    if hasattr(ta, "hmm")
                    else "unknown",
                    last_decision_ids=last_ids,
                    meta_score_history=meta_hist,
                    daily_turnover_rub=getattr(turnover, "today_volume", 0.0),
                    n_trades_today=cb.state.n_trades_today,
                    open_positions=open_positions_snap,
                    extras=extras_dict,
                )
                await recovery.save_atomic(snap)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Recovery save iteration failed", extra={"error": str(exc)})

    recovery_task = asyncio.create_task(_recovery_loop(), name="recovery_save_loop")

    online_retrain_task = None
    try:
        from app.training.online_retrain import start_background_loop

        online_retrain_task = asyncio.create_task(
            start_background_loop(interval_seconds=60),
            name="online_retrain_loop",
        )
        logger.info("Запущен background-цикл online retrain", extra={"interval_sec": 60})
    except Exception as exc:
        logger.warning("Online retrain loop start failed", extra={"error": str(exc)})

    metrics_task = None
    snapshot_task = None
    sl_monitor = None
    try:
        from app.dashboard.metrics_writer import metrics_writer_loop

        metrics_task = asyncio.create_task(metrics_writer_loop(), name="metrics_writer_loop")
    except Exception as exc:
        logger.warning("metrics_writer not started", extra={"error": str(exc)})

    try:
        from app.risk.stop_loss_monitor import get_stop_loss_monitor

        sl_monitor = get_stop_loss_monitor()
        await sl_monitor.start()
    except Exception as exc:
        logger.warning("stop_loss_monitor not started", extra={"error": str(exc)})

    mood_scanner = None
    if getattr(cfg, "MARKET_MOOD_ENABLED", False):
        try:
            from app.agents.market_mood import get_market_mood_scanner

            mood_scanner = get_market_mood_scanner()
            await mood_scanner.start()
        except Exception as exc:
            logger.warning("market_mood_scanner not started", extra={"error": str(exc)})

    if cfg.DASHBOARD_SNAPSHOT_ENABLED:
        try:
            from app.dashboard.snapshot_renderer import snapshot_loop

            snapshot_task = asyncio.create_task(snapshot_loop(), name="snapshot_loop")
        except Exception as exc:
            logger.warning("snapshot_loop not started", extra={"error": str(exc)})

    try:
        from app.dashboard.static_server import start_static_server

        start_static_server(port=8501)
    except Exception as exc:
        logger.error("static_server start failed", extra={"error": str(exc)})

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _signal_handler(signum, _frame):
        """Handle SIGTERM/SIGINT by stopping dispatcher."""
        logger.info(f"Received signal {signum}, shutting down")
        stop_event.set()
        dispatcher.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _signal_handler(s, None))
        except NotImplementedError:
            signal.signal(sig, _signal_handler)

    dispatcher_task = asyncio.create_task(dispatcher.run(), name="dispatcher_loop")

    try:
        await stop_event.wait()
    finally:
        logger.info("Завершение работы...")
        if sl_monitor is not None:
            with contextlib.suppress(Exception):
                await sl_monitor.stop()
        if reconciler is not None:
            with contextlib.suppress(Exception):
                await reconciler.stop()
        for bg in (recovery_task, metrics_task, snapshot_task):
            if bg is None:
                continue
            try:
                bg.cancel()
                await asyncio.wait_for(bg, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                pass
        with contextlib.suppress(Exception):
            scheduler.shutdown()
        dispatcher.stop()
        try:
            await asyncio.wait_for(dispatcher_task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            dispatcher_task.cancel()

        if rss_manager:
            with contextlib.suppress(Exception):
                await rss_manager.stop()
        if sanctions_parser:
            with contextlib.suppress(Exception):
                await sanctions_parser.stop()
        if moex_parser:
            with contextlib.suppress(Exception):
                await moex_parser.stop()

        await asyncio.gather(
            ta.shutdown(),
            anomaly.shutdown(),
            news.shutdown(),
            pair.shutdown(),
            mean_rev.shutdown(),
            return_exceptions=True,
        )
        await book.stop()

        await asyncio.gather(
            iss.shutdown(),
            algopack.shutdown(),
            arenago.shutdown(),
            polza.shutdown(),
            return_exceptions=True,
        )

        logger.info(
            "Работа завершена",
            extra={"dispatcher_stats": dispatcher.stats},
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
