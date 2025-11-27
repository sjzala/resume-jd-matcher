FROM python:3.11-slim AS base
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

COPY data ./data
COPY app.py ./app.py

EXPOSE 7860
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["streamlit", "run", "src/matcher/ui/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
