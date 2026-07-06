FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ ca-certificates curl tzdata bash \
    && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        httpx 'pydantic>=2.7' feedparser selectolax \
        'numpy>=1.24' 'pandas>=2.0' \
        aiosqlite python-json-logger apscheduler python-dotenv \
        'datasketch>=1.6' orjson msgspec eval_type_backport \
        'openai>=1.0' streamlit yfinance && \
    pip install --no-cache-dir \
        scipy statsmodels && \
    pip install --no-cache-dir \
        hmmlearn catboost && \
    pip install --no-cache-dir \
        pandas-ta-classic natasha ta-lib && \
    pip install --no-cache-dir \
        plotly streamlit-autorefresh streamlit-aggrid streamlit-extras && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        'torch>=2.0,<2.4' && \
    pip install --no-cache-dir \
        'transformers>=4.30,<5.0'

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUN_MODE=paper

ARG BUILD_TIME=2026-05-27T14:35:00Z
ENV BUILD_TIME=${BUILD_TIME}
ARG APP_VERSION=v1.5.0
ENV APP_VERSION=${APP_VERSION}

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY start.sh ./start.sh
RUN chmod +x /app/start.sh

COPY .streamlit/ ./.streamlit/

COPY data/models/ /opt/seed_models/
COPY data/feeds.db /opt/seed_data/feeds.db

VOLUME /data
ENV DATA_DIR=/data

EXPOSE 8501

CMD ["/app/start.sh"]
