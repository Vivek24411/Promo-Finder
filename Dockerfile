# Promo Finder web app, for a shared server. Public mode is on: visitors can't read server folders,
# see other visitors' runs, clear the cache, or point the server's API key at a custom URL.
#   docker build -t promo-finder .
#   docker run -d --name promo-finder --restart unless-stopped --memory 1g --cpus 1 \
#     --env-file .env -p 127.0.0.1:8511:8501 \
#     -v promo-finder-cache:/app/.cache -v promo-finder-outputs:/app/outputs promo-finder
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROMO_FINDER_PUBLIC=1

WORKDIR /app
COPY requirements.txt .
# Same versions the tests ran against.
RUN pip install -r requirements.txt "streamlit==1.67.0" "anthropic==2.6.0"

COPY . .
RUN useradd --create-home --uid 10001 app && mkdir -p .cache/llm outputs && chown -R app .cache outputs
USER app

EXPOSE 8501
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
