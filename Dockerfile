FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MODEL_PATH=/app/models/best.onnx \
    MODEL_VERSION=yolo26n-hammer-v1.0

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY static ./static
COPY models ./models

RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=45s --retries=4 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health', timeout=2)" || exit 1

CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log"]
