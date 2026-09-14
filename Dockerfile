FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app

# Build tools are needed by some wheels; removed again to keep the image small
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so this layer is cached when only code changes
COPY requirements.txt .

# CPU-only torch first. sentence-transformers depends on torch, and the default
# PyPI wheel for Linux is the CUDA build: ~2GB of nvidia packages this image
# has no GPU to use. Installing the CPU wheel up front satisfies that dependency
# so the next step skips them entirely.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y

COPY src/ ./src/
COPY db/ ./db/
COPY config.yaml ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]