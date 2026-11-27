FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf

WORKDIR /app

COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r deploy/requirements.txt

# Downloaded at build time so the running server never needs to reach HuggingFace.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY scripts/causal_gold_articles_with_topics_bert.csv scripts/
COPY redflag_market_kit redflag_market_kit
COPY deploy deploy
RUN python deploy/build.py

# Only when the server itself should log in to Refinitiv (SERVER_REFINITIV=1 with platform credentials).
ARG INSTALL_REFINITIV=0
RUN if [ "$INSTALL_REFINITIV" = "1" ]; then pip install lseg-data; fi

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 10001 app && mkdir -p /app/deploy/runtime && chown -R app /app/.hf /app/deploy/runtime
USER app

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn server:app --app-dir deploy --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
