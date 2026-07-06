# 404: Loss Not Found — автономный ML-трейдер MOEX

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()
[![MOEX](https://img.shields.io/badge/exchange-MOEX-red.svg)](https://www.moex.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![LLM: open-source](https://img.shields.io/badge/LLM-MIT%20%2F%20Apache--2.0-orange.svg)]()
[![Hackathon](https://img.shields.io/badge/MOEX%20AI%20Hackathon-2026-purple.svg)]()

Полностью автономный (без ручного управления) торговый агент для платформы ArenaGo
Московской биржи. Управляет портфелем 1 000 000 ₽ виртуальных средств на 21 тикере
голубых фишек MOEX. Принимает решения BUY / SELL / HOLD на основе пяти независимых
ML-моделей, объединяемых единым диспетчером, риск-менеджером и системой ретробучения.

Проект разработан командой **404: Loss Not Found** для MOEX AI Hackathon 2026
(Этап 2: 28 мая — 10 июня 2026, 14 дней автономной торговли).

## Содержание

1. [Архитектура](#архитектура)
2. [Быстрый старт](#быстрый-старт)
3. [Переменные окружения](#переменные-окружения)
4. [Стратегии](#стратегии)
5. [Risk Management](#risk-management)
6. [Модели и данные](#модели-и-данные)
7. [LLM и лицензии](#llm-и-лицензии)
8. [Dashboard](#dashboard)
9. [CI/CD](#cicd)
10. [Тесты](#тесты)
11. [Переобучение](#переобучение)
12. [Этап 2 — отказоустойчивость](#этап-2--отказоустойчивость)
13. [Структура репозитория](#структура-репозитория)
14. [Команда](#команда)

## Архитектура

```
   MOEX ISS  ─────────┐
   AlgoPack  ────┐    │
   50+ RSS   ────┤    │      ┌────────────────────────────────────┐
   Polza LLM ────┤    └─────►│  5 ML-адаптеров                    │
                 │           │  • TA Trader (87 паттернов)        │
                 │           │  • Anomaly Detector (6)            │
                 │           │  • News LLM (LLM + RAG)            │
                 │           │  • Pair Trader (6 пар)             │
                 │           │  • Mean Reversion (BB+RSI)         │
                 │           └──────────┬─────────────────────────┘
                 │                      │ UnifiedSignal
                 │                      ▼
                 │           ┌──────────────────────┐
                 │           │  Dispatcher          │
                 │           │   Aggregator         │
                 │           │   Tier classifier    │
                 │           │   Meta classifier    │
                 │           └──────────┬───────────┘
                 │                      │ Decision
                 │                      ▼
                 │           ┌──────────────────────────────────┐
                 │           │  Risk Manager (15 слоёв)         │
                 │           └──────────┬───────────────────────┘
                 │                      ▼
                 │           ┌──────────────────────┐
                 │           │  ArenaGo execution   │
                 │           └──────────────────────┘
                 ▼
        /data (persistent):
          decisions.db, trades.db, feeds.db,
          recovery_state.json (atomic 5s),
          models/, logs/, rag/
```

## Быстрый старт

### Docker (рекомендуется)
```bash
docker compose up -d
open http://localhost:8501
```

### Локально
```bash
pip install -e .
cp .env.example .env
python -m app.main
```

Дашборд: `http://localhost:8501`. Healthcheck: `http://localhost:8501/_stcore/health`.

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `SANDBOX_API_KEY` | — | Ключ ArenaGo (без `Bearer`) |
| `ALGOPACK_TOKEN` | — | JWT-токен MOEX AlgoPack |
| `POLZA_API_KEY` | — | polza.ai LLM router |
| `RUN_MODE` | `paper` | `paper` / `backtest` / `live` |
| `LIVE_SIZING` | `1` | Боевой sizing T1/T2/T3 = 3.5%/2.2%/0% |
| `RAG_LLM_BACKEND` | `polza` | Бэкенд RAG-consensus |
| `POLZA_MODEL_REACTIVE` | `deepseek/deepseek-v4-flash` | MIT — news reactive |
| `POLZA_MODEL_FALLBACK` | `qwen/qwen3.5-plus-20260420` | Apache-2.0 — budget fallback |
| `POLZA_MODEL_SANCTIONS` | `deepseek/deepseek-r1-0528` | MIT — reasoning санкций |
| `POLZA_MODEL_MORNING_BRIEF` | `qwen/qwen3.7-max` | Apache-2.0, 1M-контекст |
| `FORCE_REGIME_NAME` | `NORMAL` | Фиксация regime (NORMAL/CAUTIOUS/DEFENSIVE/CRISIS) |
| `DISABLE_REGIME_DD_CHECK` | `1` | Игнорировать DD в regime detection |
| `MIN_CASH_RESERVE_PCT` | `0.05` | Минимальный cash reserve |
| `MAX_TICKER_NOTIONAL_PCT_CUMULATIVE` | `0.08` | Кумулятивный cap позиции |
| `HARD_STOP_LOSS_PCT` | `0.015` | -1.5% безусловный stop-loss |
| `HARD_TAKE_PROFIT_PCT` | `0.06` | +6% take-profit (свежие) |
| `HARD_TAKE_PROFIT_AGED_PCT` | `0.02` | +2% для позиций ≥8ч |
| `HARD_TIME_STOP_HOURS` | `12` | Закрыть стейл-позицию через 12ч |
| `COUNTER_BIAS_GUARD_ENABLED` | `1` | Soft penalty counter-bias сигналов |
| `PRE_LONG_MOMENTUM_GUARD_ENABLED` | `1` | Блок LONG при -1.5%+ momentum |
| `STAGE2_RESET_PEAK_EQUITY` | `0` | Сбросить peak equity (для Stage 2) |

## Стратегии

5 независимых ML-адаптеров работают параллельно в 30-сек цикле Dispatcher.

### 1. Technical Pattern Trader
87 паттерн-детекторов на 10-мин фрейме:
- 8 reversal (Double Top/Bottom, H&S, Wedge, Megaphone, Diamond, Rounding, Cup&Handle)
- 7 continuation (Flag, Pennant, Triangle, Rectangle, Box, Wedge, Compression)
- 6 harmonic (Gartley, Bat, Butterfly, Crab, Cypher, Shark)
- 5 SMC (Order Block, FVG, Liquidity Sweep, BOS, CHOCH)
- 61 candle pattern (TA-Lib)

Фильтрация: CatBoost (99 features) + HMM regime gate.

### 2. Anomaly Detector
6 микроструктурных детекторов на AlgoPack supercandles:
Volume Z-score / Price spikes / Absorption / VWAP crosses / OFI spikes / ATR reversion.

### 3. News LLM
50+ RSS источников в 7 tiers. Pipeline: RSS → Dedup → NER → Material filter →
LLM (DeepSeek v4-flash) → RAG consensus.

### 4. Pair Trader
6 коинтегрированных пар (SBER/SBERP, SNGS/SNGSP, ROSN/LKOH, NLMK/CHMF, VTBR/SBER).
Engle-Granger 2-step + ADF (p<0.05). Daily refit. Entry |z|>1.5σ, exit |z|<0.3.

### 5. Mean Reversion
Bollinger Bands (20, 2σ) + RSI(14) на 5-мин барах. Активен в `mean_reverting` regime.

## Risk Management

15 защитных слоёв:

### PRE-открытие (9 проверок)
1. PER_TICKER_POLICY (GOLD / WHITELIST_ONLY / DISABLED)
2. counter_bias_guard (×0.7 penalty)
3. pre_long_momentum_guard (-1.5%)
4. MTF counter-trend VETO
5. microstructure gates (OFI / Kyle / VPIN)
6. Meta-classifier (CatBoost, ≥0.35)
7. cash reserve floor (5%)
8. concentration cap (8% cumulative)
9. entry cooldown (12 мин)

### POST-открытие (мониторинг каждые 5 сек)
10. HARD_STOP_LOSS -1.5%
11. HARD_TIME_STOP 12h (pnl<+0.5%)
12. HARD_TAKE_PROFIT +6%/+2% старые
13. trailing stop (R-based)
14. TP1/TP2 partial exits
15. circuit breakers (daily -1.5%, equity floor 50%)

### Адаптивный режим
| Regime | Триггер | Size multiplier |
|---|---|---|
| NORMAL | default | 1.0 |
| CAUTIOUS | DD≥0.3% / streak≥2 / pnl≤-0.3% | 0.7 |
| DEFENSIVE | DD≥1.5% / streak≥3 / pnl≤-0.8% | 0.4 |
| CRISIS | DD≥6% / streak≥5 / pnl≤-2.5% | 0.25 |

## Модели и данные

- **CatBoost**: 99 features, triple-barrier labeling, purged k-fold CV, test_acc 73.9%
- **HMM**: GaussianHMM(3) на 60d IMOEX, состояния {trending, mean_reverting, crisis}
- **Meta-classifier**: secondary scoring (López de Prado meta-labeling)
- **Pair Trading**: Engle-Granger + ADF, β/μ/σ refit daily
- **RAG**: ChromaDB + Qwen3-embedding-8b, 1041+ historical events с post-hoc PnL

## LLM и лицензии

Все LLM имеют открытые лицензии (MIT / Apache-2.0):

| Модель | Лицензия | Применение |
|---|---|---|
| `deepseek/deepseek-v4-flash` | MIT | News reactive |
| `deepseek/deepseek-r1-0528` | MIT | Sanctions reasoning |
| `qwen/qwen3.5-plus-20260420` | Apache-2.0 | Consensus + reasoning |
| `qwen/qwen3.7-max` | Apache-2.0 | Morning brief (1M context) |
| `qwen/qwen3-embedding-8b` | Apache-2.0 | Embeddings |
| `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | Fallback embeddings |

LLM-router: **polza.ai** (OpenAI-compatible API, Apache-2.0 SDK).

## Dashboard

Static HTTP-сервер (stdlib `http.server`), порт 8501:

| Endpoint | Описание |
|---|---|
| `/` | HTML snapshot (KPIs, equity, decisions) |
| `/_stcore/health`, `/healthz`, `/livez`, `/readyz` | Probes → `ok` |
| `/metrics.json` | Live KPIs (JSON) |

Snapshot регенерируется каждые 5 мин.

## CI/CD

GitLab Pipeline stages:
1. `test` — ruff lint + pytest unit
2. `build` — Docker image
3. `deploy` — Yandex Cloud
4. `ops` (manual) — bot_restart, attach_disk, detach_disk

Pre-commit: ruff, ruff-format, check-yaml, check-toml, large-files (2MB), private-keys.

## Тесты

```bash
pytest tests/unit -v --asyncio-mode=auto
```

Покрытие: TA patterns, anomaly, dispatcher, risk_manager, order_manager idempotency,
news pipeline, RAG, CatBoost loader, HMM, pair trading.

## Переобучение

```bash
python scripts/train_catboost.py --days 180 --interval 10
python scripts/train_hmm.py --days 120
python scripts/refit_pairs.py --days 90
python scripts/train_meta_v2.py
```

Online retrain (background) каждые 60 сек при ≥50 новых сделок.

## Этап 2 — отказоустойчивость

### Защиты на 14 дней
1. **Recovery state** — atomic snapshot каждые 5 сек
2. **Broker reconciler** — синхронизация с ArenaGo каждые 5 мин
3. **Orphan-order reconcile** — на startup
4. **Stop-loss monitor retry** — при `INSUFFICIENT_CASH` retry-on-failure
5. **Docker restart policy** `unless-stopped`
6. **k8s liveness probe** на `/_stcore/health`
7. **/data persistent volume** — все БД в volume, переживают restart

### Stage 2 startup
- `/data/decisions.db`, `/data/trades.db`, `/data/feeds.db` — persistent
- ENV `SANDBOX_API_KEY` — обновится организаторами автоматически
- ENV `STAGE2_RESET_PEAK_EQUITY=1` — установить при первом restart 28 мая

## Структура репозитория

```
.
├── app/                # Main application code (Python 3.11+)
│   ├── agents/         # 5 strategy adapters + market_mood, hmm_regime
│   ├── dashboard/      # static_server, snapshot_renderer, metrics_writer
│   ├── data/           # ISS, AlgoPack, candle_store, supercandles
│   ├── dispatcher/     # dispatcher, aggregator, tier_classifier
│   ├── execution/      # ArenaGo client, order_manager, broker_reconciler
│   ├── llm/            # polza_client
│   ├── memory/         # ChromaDB, reflection, morning_plan
│   ├── news/           # parsers, RAG, ticker_tagger, material_filter
│   ├── recovery/       # state_manager
│   ├── risk/           # risk_manager, position_book, stop_loss_monitor
│   ├── training/       # labeling, cross_validation, meta_features
│   └── utils/          # logging, sessions, scheduler
├── scripts/            # 8 critical scripts (bootstrap_db, train_*, refit_pairs)
├── tests/              # Unit + integration tests
├── notebooks/          # Analysis notebooks (A/B/C)
├── data/               # /data volume mount
├── Dockerfile          # python:3.11-slim
├── docker-compose.yml
├── start.sh            # bootstrap → run app
├── .gitlab-ci.yml      # CI/CD
├── pyproject.toml
└── README.md
```

## Команда

**404: Loss Not Found** — single-developer team.
MOEX AI Hackathon 2026 · автономная торговля · 28 мая — 10 июня.

## Лицензия

MIT (см. `LICENSE`).
