FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools are needed by some wheels, then removed to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache the dependency layer.
COPY requirements.txt .

# The CPU wheel index for torch, which sentence-transformers pulls in. Without
# it pip takes the default build, which carries the CUDA runtime and about two
# gigabytes of NVIDIA libraries into a container that has no GPU. The first
# build of this image died inside that install with pip exit code 2, out of
# memory on a machine with under a gigabyte free. CI already used this index;
# the Dockerfile did not, and an image that only builds on the CI runner is
# not a deployable image.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y

COPY src/ ./src/
COPY db/ ./db/
COPY config.yaml ./
EXPOSE 8000

# Give the API enough time to start before health checks begin.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]